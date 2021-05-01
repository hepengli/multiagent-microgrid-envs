"""
Oct 12, 2020
Created by Hepeng Li

Network Core Functions.
"""

import numpy as np
from collections import deque
from pypower.api import ppoption, runpf, rundcpf
from pypower.idx_brch import F_BUS, T_BUS, TAP, RATE_A, BR_STATUS, PF, PT, QF, QT
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS
from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PQ, REF

# state of agents
class AgentState(object):
    def __init__(self):
        super(AgentState, self).__init__()
        # unit commitment
        self.uc = 1
        self.P = 0
        self.Q = 0

# action of the agent
class Action(object):
    def __init__(self):
        # discreate action
        self.d = None
        # continuous action
        self.c = None
        # control categories (d) and range (c)
        self.ncats = None
        self.dim_d = 0
        self.range = None
        self.dim_c = 0

# properties of agent entities
class Agent(object):
    def __init__(self):
        super(Agent, self).__init__()
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

# properties of distributed generator agents
class DG(Agent):
    def __init__(self, name, bus, min_p_mw, max_p_mw, sn_mva=None, min_pf=None, control_q=False, type='DG', 
        leading=None, min_q_mvar=None, max_q_mvar=None, ramp_up=1e10, ramp_down=1e10, dt=1, 
        cost_curve_coefs=[0,0,0], startup_time=None, shutdown_time=None, startup_cost=0, 
        shutdown_cost=0):
        super(DG, self).__init__()
        # properties
        self.type = type
        self.name = name
        self.bus = bus
        self.sn_mva = sn_mva
        self.min_p_mw = min_p_mw
        self.max_p_mw = max_p_mw
        if self.type == 'DG':
            if sn_mva is not None:
                self.min_pf = max_p_mw / sn_mva if min_pf is None else min_pf
                self.min_q_mvar = - np.sqrt(sn_mva**2 - max_p_mw**2)
                self.max_q_mvar = np.sqrt(sn_mva**2 - max_p_mw**2)
            else:
                self.min_q_mvar = 0
                self.max_q_mvar = 0
                if min_q_mvar is not None:
                    self.min_q_mvar = min_q_mvar
                if max_q_mvar is not None:
                    self.max_q_mvar = max_q_mvar
                    self.sn_mva = np.sqrt(max_p_mw**2 + max_q_mvar**2)
                self.min_pf = max_p_mw / self.sn_mva if min_pf is None else min_pf
        elif self.type == 'CL':
            if sn_mva is not None:
                self.min_pf = - min_p_mw / sn_mva if min_pf is None else min_pf
        self.leading = leading
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.dt = dt
        self.cost_curve_coefs = cost_curve_coefs
        self.startup_time = startup_time
        self.shutdown_time = shutdown_time
        self.startup_cost = startup_cost
        self.shutdown_cost = shutdown_cost
        # state variables
        self.state.P = 0
        self.state.Q = 0
        self.state.shutting = 0
        self.state.starting = 0
        # cost and safety
        self.uc_cost = 0
        self.safety = 0
        # action spaces
        if control_q:
            self.action.dim_c = 2
            self.action.range = np.array([[self.min_p_mw, self.min_q_mvar], [self.max_p_mw, self.max_q_mvar]])
        else:
            self.action.dim_c = 1
            self.action.range = np.array([[self.min_p_mw], [self.max_p_mw]])
        if startup_time is not None:
            self.action.ncats = 2
            self.action.dim_d = 1

    def update_state(self):
        # update uc status
        if self.action.d is not None:
            self.update_uc_state()
        # update P and Q
        self.state.P = self.action.c[0]
        if self.action.dim_c > 1:
            self.state.Q = self.action.c[1]
        elif self.action.dim_c == 1:
            if self.type == 'DG':
                self.state.Q = np.sqrt((self.state.P / self.min_pf)**2 - self.state.P**2)
            if self.type == 'CL':
                if self.leading is True:
                    self.state.Q = np.sqrt((self.state.P / self.min_pf)**2 - self.state.P**2)
                elif self.leading is False:
                    self.state.Q = - np.sqrt((self.state.P / self.min_pf)**2 - self.state.P**2)

    def update_cost_safety(self):
        # quadratic cost
        if len(self.cost_curve_coefs) == 3: 
            a, b, c = self.cost_curve_coefs
            fuel_cost = a * (self.state.P**2) + b * self.state.P + c # ($)
        # piecewise linear
        else:
            assert len(self.cost_curve_coefs) % 2 == 0
            p0, f0 = self.max_p_mw, 0
            for i in range(0, len(self.cost_curve_coefs), 2):
                p1, f1 = self.cost_curve_coefs[i:i+2]
                if self.state.P <= p1:
                    fuel_cost = f0 + (self.state.P-p0) * (f1-f0)/(p1-p0) # ($)
                    break
                else:
                    p0, f0 = p1, f1
        # step cost
        self.cost = (self.state.uc * fuel_cost + self.uc_cost) * self.dt
        # step safety
        if self.action.dim_c > 1:
            S = np.sqrt(self.state.P**2 + self.state.Q**2)
            # see https://www.ny-engineers.com/blog/diesel-genset-specifications-kw-kva-and-power-factor
            if S > 0 and abs(self.state.P / S) < self.min_pf:
                self.safety = self.min_pf - abs(self.state.P / S)
            else:
                self.safety = 0

    def update_uc_state(self):
        # cannot start up and shut down at the same time
        assert not (self.state.shutting and self.state.starting)
        if not self.state.shutting and not self.state.starting:
            # shut down
            if self.state.uc == 1 and self.action.d == 0:
                self.state.shutting += 1
            # start up
            if self.state.uc == 0 and self.action.d == 1:
                self.state.starting += 1
        if self.state.starting:
            assert self.state.uc == 0
            if self.state.starting > self.startup_time:
                self.state.uc = 1
                self.state.starting = 0
                self.uc_cost = 0
            else:
                self.state.starting += 1
                self.uc_cost += self.startup_cost
        if self.state.shutting:
            assert self.state.uc == 1
            if self.state.shutting > self.shutdown_time:
                self.state.uc = 0
                self.state.shutting = 0
                self.uc_cost = 0
            else:
                self.state.shutting += 1
                self.uc_cost += self.shutdown_cost

    def reset(self, rnd):
        # uc status
        self.state.uc = 1
        self.state.shutting = 0
        self.state.starting = 0
        # generation
        self.state.P = 0
        self.state.Q = 0
        # raise NotImplementedError()

# properties of energy storage agents
class ESS(Agent):
    def __init__(self, name, bus, min_p_mw, max_p_mw, max_e_mwh, sn_mva=None, init_soc=None, min_e_mwh=0.0, 
            min_q_mvar=None, max_q_mvar=None, ch_eff=0.98, dsc_eff=0.98, cost_curve_coefs=[0,0,0], dt=1):
        super(ESS, self).__init__()
        # properties
        self.type = 'ESS'
        self.name = name
        self.bus = bus
        self.min_e_mwh = min_e_mwh
        self.max_e_mwh = max_e_mwh
        self.sn_mva = sn_mva
        self.min_p_mw = min_p_mw
        self.max_p_mw = max_p_mw
        self.min_q_mvar = min_q_mvar
        self.max_q_mvar = max_q_mvar
        self.dt = dt
        self.ch_eff = ch_eff
        self.dsc_eff = dsc_eff
        self.cost_curve_coefs = cost_curve_coefs
        self.init_soc = init_soc
        self.min_soc = min_e_mwh / max_e_mwh
        self.max_soc = 1
        # cost and safety
        self.cost = 0
        self.safety = 0
        self.state.soc=0.5
        # action spaces
        if sn_mva is None:
            self.action.dim_c = 1
            self.action.range = np.array([[min_p_mw], [max_p_mw]])
        else:
            self.action.dim_c = 2
            self.action.range = np.array([[min_p_mw, min_q_mvar], [max_p_mw, max_q_mvar]])

    def update_state(self):
        # self.feasible_action()
        # compute ture soc
        self.state.P = P = self.action.c[0]
        self.state.Q = self.action.c[1] if self.action.dim_c > 1 else 0
        if P >= 0: # charging
            self.state.soc = self.state.soc + P * self.ch_eff * self.dt / self.max_e_mwh
        elif P < 0: # discharging
            self.state.soc = self.state.soc + P / self.dsc_eff * self.dt / self.max_e_mwh

    def update_cost_safety(self):
        # quadratic cost
        a, b, c, *_ = self.cost_curve_coefs
        self.cost = a * abs(self.state.P)
        # step safety
        safety = 0
        if self.sn_mva is not None:
            S = np.sqrt(self.state.P**2 + self.state.Q**2)
            if S > self.sn_mva:
                safety += S - self.sn_mva
        # restrict soc
        if self.state.soc > self.max_soc:
            safety += self.state.soc - self.max_soc
        if self.state.soc < self.min_soc:
            safety += self.min_soc - self.state.soc

        self.safety = safety

    def reset(self, rnd):
        # generation
        self.state.soc = rnd.uniform(self.min_soc, self.max_soc) if self.init_soc is None else self.init_soc

    def feasible_action(self):
        max_discharge_power = (self.state.soc - self.min_soc) * self.max_e_mwh * self.dsc_eff / self.dt
        max_discharge_power = min(max_discharge_power, -self.min_p_mw)

        max_charge_power = (self.max_soc - self.state.soc) * self.max_e_mwh / self.ch_eff / self.dt
        max_charge_power = min(max_charge_power, self.max_p_mw)

        low, high = np.array([-max_discharge_power]), np.array([max_charge_power])
        if len(self.action.c) > 1:
            low = np.append(low, self.min_q_mvar)
            high = np.append(high, self.max_q_mvar)
        self.action.c = np.clip(self.action.c, low, high)

# properties of renewable energy resource agents
class RES(Agent):
    def __init__(self, name, source, bus, sn_mva, max_q_mvar=0, min_q_mvar=0, control_q=False, dt=1, cost_curve_coefs=[0,0,0]):
        super(RES, self).__init__()
        # properties
        assert source in ['SOLAR', 'WIND']
        self.type = source
        self.name = name
        self.bus = bus
        self.sn_mva = sn_mva
        self.max_q_mvar = max_q_mvar
        self.min_q_mvar = min_q_mvar
        self.control_q = control_q
        self.dt = dt
        self.cost_curve_coefs = cost_curve_coefs
        # state
        self.state.P = 0
        self.state.Q = 0
        # cost and safety
        self.cost = 0
        self.safety = 0
        # action spaces
        if control_q:
            self.action.range = np.array([[min_q_mvar], [max_q_mvar]])
            self.action.dim_c = 1
        else:
            self.action_callback = True

    def update_state(self, scaling):
        assert 0<= scaling <= 1
        # update state
        self.state.P = self.sn_mva * scaling
        if self.action.dim_c > 0:
            self.state.Q = self.action.c[0]

    def update_cost_safety(self):
        # update safety
        if self.action.dim_c > 0:
            S = np.sqrt(self.state.P**2 + self.state.Q**2)
            if S > self.sn_mva:
                self.safety = S - self.sn_mva
            else:
                self.safety = 0

    def reset(self, rnd):
        self.state.P = 0
        self.state.Q = 0

# properties of capacitor agents
class Shunt(Agent):
    def __init__(self, name, bus, q_mvar, max_step=1):
        super(Shunt, self).__init__()
        # properties
        self.type = 'SCB'
        self.name = name
        self.bus = bus
        self.q_mvar = q_mvar
        self.max_step = max_step
        # cost and safety
        self.cost = 0
        self.safety = 0
        # action spaces
        self.action.ncats = max_step + 1
        self.action.dim_d = 1

    def update_state(self):
        self.state.step = self.action.d[0]

    def update_cost_safety(self):
        pass

    def reset(self, rnd):
        self.state.Q = 0

# properties of substation agents
class Transformer(Agent):
    def __init__(self, name, type, fbus, tbus, sn_mva=None, tap_max=None, tap_min=None, dt=1):
        super(Transformer, self).__init__()
        # properties
        self.type = type # 'TAP' or 'Trafo'
        self.name = name
        self.fbus = fbus
        self.tbus = tbus
        self.sn_mva = sn_mva
        self.tap_max = tap_max
        self.tap_min = tap_min
        self.dt = dt
        # state variables
        self.state.loading = 0
        self.state.tap_position = 0
        # cost and safety
        self.cost = 0
        self.safety = 0
        # action spaces
        if tap_max is not None:
            self.action.ncats = tap_max - tap_min + 1
            self.action.dim_d = 1
        else:
            self.action_callback = True

    def update_state(self):
        if self.tap_max is not None:
            self.state.tap_position = self.action.d[0] + self.tap_min

    def update_cost_safety(self, loading):
        # update state
        self.state.loading = loading
        # update safety
        if loading > 100:
            self.safety = (loading - 100) / 100
        else:
            self.safety = 0

    def reset(self, rnd):
        self.state.loading = 0
        self.state.tap_position = 0

# properties of main grid agent
class Grid(Agent):
    def __init__(self, name, bus, sn_mva, sell_discount=1, dt=1):
        super(Grid, self).__init__()
        # properties
        self.type = 'GRID'
        self.name = name
        self.bus = bus
        self.sn_mva = sn_mva
        self.sell_discount = sell_discount
        self.dt = dt
        # state variables
        self.state.P = 0
        self.state.Q = 0
        self.state.price = 0
        # cost and safety
        self.cost = 0
        self.safety = 0
        # not need action
        self.action_callback = True

    def update_state(self, price, P, Q=0):
        # update state
        self.state.P = P
        self.state.Q = Q
        self.state.price = price

    def update_cost_safety(self):
        # update cost
        if self.state.P > 0:
            self.cost = self.state.P * self.state.price * self.dt
        else:
            self.cost = self.state.P * self.state.price * self.sell_discount * self.dt
        # update safety
        S = np.sqrt(self.state.P**2 + self.state.Q**2)
        if S > self.sn_mva:
            self.safety = (S - self.sn_mva) / self.sn_mva
        else:
            self.safety = 0

    def reset(self, rnd):
        self.state.P = 0
        self.state.Q = 0
        self.state.price = 0
