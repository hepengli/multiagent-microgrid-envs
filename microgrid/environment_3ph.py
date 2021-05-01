import numpy as np
import pandas as pd
import pandapower as pp
from collections import deque
from copy import deepcopy

import time
import gym
from gym import spaces
from gym.utils import seeding
from microgrid.read_data import read_pickle_data
from microgrid.core import DG, ESS, RES, Shunt, Transformer, Grid
from microgrid.scenarios.case34_3ph import case34_3ph

def isNAN(num):
    return num != num

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Environment(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, train):
        # simulation timestep
        self.dt = 1
        # timestep counter
        self.t = 0
        # running mode: train or test
        self.train = train
        # simulation data: load, solar power, wind pwoer, price
        self.data = read_pickle_data()['train'] if train else read_pickle_data()['test']
        # days
        self.days = self.data['solar_0'].size//24
        # network architecture
        self.net = net = case34_3ph()
        # how much history to store
        self.past_t = 24
        # seed
        self.seed()
        # list of agents
        DG_1 = DG('DG 1', bus='Bus 848', min_p_mw=0, max_p_mw=0.66, sn_mva=0.825, control_q=True, cost_curve_coefs=[100, 71.6, 0.4615])
        DG_2 = DG('DG 2', bus='Bus 890', min_p_mw=0, max_p_mw=0.50, sn_mva=0.625, control_q=True, cost_curve_coefs=[100, 62.4, 0.5011])
        PV_1 = RES('PV 1', source='SOLAR', bus='Bus 822', sn_mva=0.2, control_q=False)
        PV_2 = RES('PV 2', source='SOLAR', bus='Bus 856', sn_mva=0.2, control_q=False)
        PV_3 = RES('PV 3', source='SOLAR', bus='Bus 838', sn_mva=0.2, control_q=False)
        WP_1 = RES('WP_1', source='WIND', bus='Bus 822', sn_mva=0.3, control_q=False)
        WP_2 = RES('WP_2', source='WIND', bus='Bus 826', sn_mva=0.3, control_q=False)
        WP_3 = RES('WP_3', source='WIND', bus='Bus 838', sn_mva=0.3, control_q=False)
        SUB = Transformer('Substation', type='TAP', fbus='Bus0', tbus='Bus 800', sn_mva=2.5, tap_max=2, tap_min=-2)
        TF = Transformer('TF', type='Trafo', fbus='Bus 832', tbus='Bus 888', sn_mva=0.5)
        TAP_1 = Transformer('TAP 1', type='TAP', fbus='Bus 814', tbus='Bus 850', sn_mva=2.5, tap_max=10, tap_min=-16)
        TAP_2 = Transformer('TAP 2', type='TAP', fbus='Bus 852', tbus='Bus 832', sn_mva=2.5, tap_max=10, tap_min=-16)
        SCB_1 = Shunt('SCB 1', bus='Bus 840', q_mvar=0.12, max_step=4)
        SCB_2 = Shunt('SCB 2', bus='Bus 864', q_mvar=0.12, max_step=4)
        ESS_1 = ESS('Storage', bus='Bus 810', min_p_mw=-0.2, max_p_mw=0.2, max_e_mwh=1.0, min_e_mwh=0.2)
        GRID = Grid('GRID', bus='Bus 800', sn_mva=2.5)
        self.agents = [DG_1, DG_2, PV_1, PV_2, PV_3, WP_1, WP_2, WP_3, TF, SUB, TAP_1, TAP_2, SCB_1, SCB_2, ESS_1, GRID]
        # reset
        ob = self.reset()

        # configure spaces
        action_space, action_shape = [], 0
        for agent in self.policy_agents:
            total_action_space = []
            # continuous action space
            if agent.action.range is not None:
                low, high = agent.action.range
                u_action_space = spaces.Box(low=low, high=high, dtype=np.float32)
                total_action_space.append(u_action_space)
                action_shape += u_action_space.shape[-1]
            # discrete action space
            if agent.action.ncats is not None:
                if isinstance(agent.action.ncats, list):
                    u_action_space = spaces.MultiDiscrete(agent.action.ncats)
                    action_shape += u_action_space.nvec.shape[-1]
                elif isinstance(agent.action.ncats, int):
                    u_action_space = spaces.Discrete(agent.action.ncats)
                    action_shape += 1
                else:
                    raise NotImplementedError()
                total_action_space.append(u_action_space)

            action_space.extend(total_action_space)

        # total action space
        if len(action_space) > 1:
            # all action spaces are discrete, so simplify to MultiDiscrete action space
            self.action_space = spaces.Tuple(action_space)

            self.action_space.shape = (action_shape,)
        else:
            self.action_space = action_space[0]
        # observation space
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(ob.shape[0],), dtype=np.float32)

        # reward
        self.reward_range = (-200.0, 200.0)

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    @property
    def resource_agents(self):
        return [agent for agent in self.agents if agent.type in ['GRID', 'DG', 'CL', 'ESS', 'SCB', 'SOLAR', 'WIND']]

    @property
    def grid_agent(self):
        return [agent for agent in self.agents if agent.type in ['GRID']]

    @property
    def dg_agents(self):
        return [agent for agent in self.agents if agent.type in ['DG']]

    @property
    def cl_agents(self):
        return [agent for agent in self.agents if agent.type in ['CL']]

    @property
    def res_agents(self):
        return [agent for agent in self.agents if agent.type in ['SOLAR', 'WIND']]

    @property
    def ess_agents(self):
        return [agent for agent in self.agents if agent.type in ['ESS']]

    @property
    def tap_agents(self):
        return [agent for agent in self.agents if agent.type in ['TAP']]

    @property
    def trafo_agents(self):
        return [agent for agent in self.agents if agent.type in ['Trafo']]

    @property
    def shunt_agents(self):
        return [agent for agent in self.agents if agent.type in ['SCB']]

    def seed(self, seed=None):
        self.rnd, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        net = deepcopy(self.net)
        # set action for each agent
        s_index, t_index = 0, 0
        for agent in self.policy_agents:
            t_index += agent.action.dim_c + agent.action.dim_d
            self._set_action(action[s_index:t_index], agent)
            s_index = t_index
        # update agent state
        for agent in self.agents:
            self._update_agent_state(agent)
        # update load info at all buses
        # area = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
        net.asymmetric_load.scaling.iloc[:5] = self.data['load_0'][self.t]
        net.asymmetric_load.scaling.iloc[5:9] = self.data['load_1'][self.t]
        net.asymmetric_load.scaling.iloc[9:-1] = self.data['load_2'][self.t]
        net.asymmetric_load.scaling.iloc[-1] = self.data['load_1'][self.t]
        net.sgen.p_mw = [agent.state.P for agent in self.dg_agents + self.res_agents]
        net.sgen.q_mvar = [agent.state.Q for agent in self.dg_agents + self.res_agents]
        net.trafo.tap_pos[:3] = [agent.state.tap_position for agent in self.tap_agents]
        net.shunt.step = [agent.state.step for agent in self.shunt_agents]
        net.storage.p_mw = [agent.state.P for agent in self.ess_agents]

        # # runpf
        # pp.pf.runpp_3ph.runpp_3ph(net)
        # converged = net["converged"]
        # nan = isNAN(net.res_ext_grid_3ph.p_a_mw.values)
        # assert converged and (not nan)

        # runpf
        try:
            pp.pf.runpp_3ph.runpp_3ph(net)
        except:
            pass
        converged = net["converged"]
        nan = isNAN(net.res_ext_grid_3ph.p_a_mw.values)
        assert converged and (not nan)

        if converged and not nan:
            # update grid state
            for agent in self.grid_agent:
                pgrid = net.res_ext_grid_3ph[['p_a_mw','p_b_mw','p_c_mw']].values.sum()
                qgrid = net.res_ext_grid_3ph[['q_a_mvar','q_b_mvar','q_c_mvar']].values.sum()
                agent.update_state(self.data['price'][self.t], pgrid, qgrid)
            # update transormers/voltage regulators cost and safety
            for agent in self.trafo_agents+self.tap_agents:
                trafo_loading = net.res_trafo_3ph.iloc[0][['loading_a_percent', 'loading_b_percent', 'loading_c_percent']].max()
                agent.update_cost_safety(trafo_loading)
            # update resource agents' cost and safety
            for agent in self.resource_agents:
                agent.update_cost_safety()
            # update power flow safety
            line_loading_percent = net.res_line_3ph[['loading_a_percent', 'loading_b_percent', 'loading_c_percent']].values
            bus_vm_pu = net.res_bus_3ph[['vm_a_pu', 'vm_b_pu', 'vm_c_pu']].values
            overloading = np.maximum(line_loading_percent - 100, 0).max()
            overvoltage = np.maximum(bus_vm_pu - 1.05, 0).max()
            undervoltage = np.maximum(0.95 - bus_vm_pu, 0).max()
            reward, safety = 0, 0
            safety += overloading / 100 + overvoltage + undervoltage
            for agent in self.agents:
                reward -= agent.cost
                safety += agent.safety
                # if agent.safety > 0:
                #     print(agent.name, agent.safety)
        else:
            reward = -200
            safety = 2

        # update past observation
        data = self.data
        self.past_load_0.append(data['load_0'][self.t])
        self.past_load_1.append(data['load_1'][self.t])
        self.past_load_2.append(data['load_2'][self.t])
        self.past_wind_0.append(data['wind_0'][self.t])
        self.past_wind_1.append(data['wind_1'][self.t])
        self.past_wind_2.append(data['wind_2'][self.t])
        self.past_solar_0.append(data['solar_0'][self.t])
        self.past_solar_1.append(data['solar_1'][self.t])
        self.past_solar_2.append(data['solar_2'][self.t])
        self.past_price.append(data['price'][self.t])

        # timestep counter
        self.t += 1
        if self.t >= self.data['solar_0'].size:
            self.t = 0

        # info
        info = {'s': safety * 100}

        return self._get_obs(), reward, False, info

    # set env action for a particular agent
    def _set_action(self, action, agent, time=None):
        index = 0
        # continuous actions
        if agent.action.dim_c > 0:
            agent.action.c = action[index:(index+agent.action.dim_c)]
            index += agent.action.dim_c
        # discrete actions
        if agent.action.dim_d > 0:
            agent.action.d = action[index:(index+agent.action.dim_d)]
            index += agent.action.dim_d
        # make sure we used all elements of action\
        assert index == len(action)

    def _update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.type in ['DG', 'CL']:
            agent.update_state()
        elif agent.type == 'ESS':
            agent.update_state()
        elif agent.type == 'SOLAR':
            if agent.name == 'PV 1':
                agent.update_state(self.data['solar_0'][self.t])
            elif agent.name == 'PV 2':
                agent.update_state(self.data['solar_1'][self.t])
            elif agent.name == 'PV 3':
                agent.update_state(self.data['solar_2'][self.t])
        elif agent.type == 'WIND':
            if agent.name == 'WP 1':
                agent.update_state(self.data['wind_0'][self.t])
            elif agent.name == 'WP 2':
                agent.update_state(self.data['wind_1'][self.t])
            elif agent.name == 'WP 3':
                agent.update_state(self.data['wind_2'][self.t])
        elif agent.type in ['TAP', 'Trafo']:
            agent.update_state()
        elif agent.type in ['SCB']:
            agent.update_state()
        else:
            pass

    def reset(self, day=None, seed=None):
        # which day
        if day is None:
            day = self.rnd.randint(self.days-1)
        # which hour
        self.t = day * 24
        # reset all agents
        if seed is not None:
            self.rnd, seed = seeding.np_random(seed)
        for agent in self.agents:
            agent.reset(self.rnd)

        t, past_t, data = self.t, self.past_t, self.data
        if t-past_t >= 0:
            self.past_load_0 = deque(data['load_0'][t-past_t:t], maxlen=past_t)
            self.past_load_1 = deque(data['load_1'][t-past_t:t], maxlen=past_t)
            self.past_load_2 = deque(data['load_2'][t-past_t:t], maxlen=past_t)
            self.past_wind_0 = deque(data['wind_0'][t-past_t:t], maxlen=past_t)
            self.past_wind_1 = deque(data['wind_1'][t-past_t:t], maxlen=past_t)
            self.past_wind_2 = deque(data['wind_2'][t-past_t:t], maxlen=past_t)
            self.past_solar_0 = deque(data['solar_0'][t-past_t:t], maxlen=past_t)
            self.past_solar_1 = deque(data['solar_1'][t-past_t:t], maxlen=past_t)
            self.past_solar_2 = deque(data['solar_2'][t-past_t:t], maxlen=past_t)
            self.past_price = deque(data['price'][t-past_t:t], maxlen=past_t)
        else:
            self.past_load_0 = deque(np.hstack([data['load_0'][t-past_t:], data['load_0'][:t]]), maxlen=past_t)
            self.past_load_1 = deque(np.hstack([data['load_1'][t-past_t:], data['load_1'][:t]]), maxlen=past_t)
            self.past_load_2 = deque(np.hstack([data['load_2'][t-past_t:], data['load_2'][:t]]), maxlen=past_t)
            self.past_wind_0 = deque(np.hstack([data['wind_0'][t-past_t:], data['wind_0'][:t]]), maxlen=past_t)
            self.past_wind_1 = deque(np.hstack([data['wind_1'][t-past_t:], data['wind_1'][:t]]), maxlen=past_t)
            self.past_wind_2 = deque(np.hstack([data['wind_2'][t-past_t:], data['wind_2'][:t]]), maxlen=past_t)
            self.past_solar_0 = deque(np.hstack([data['solar_0'][t-past_t:], data['solar_0'][:t]]), maxlen=past_t)
            self.past_solar_1 = deque(np.hstack([data['solar_1'][t-past_t:], data['solar_1'][:t]]), maxlen=past_t)
            self.past_solar_2 = deque(np.hstack([data['solar_2'][t-past_t:], data['solar_2'][:t]]), maxlen=past_t)
            self.past_price = deque(np.hstack([data['price'][t-past_t:], data['price'][:t]]), maxlen=past_t)

        return self._get_obs()

    def _get_obs(self):
        internal_state = []
        internal_state.append(np.array([(self.t%24) / 24.]))
        # for agent in self.dg_agents:
            # internal_state.append(np.arange(2)==agent.state.uc)
            # internal_state.append(np.arange(agent.startup_time+1)==agent.starting)
            # internal_state.append(np.arange(agent.shutdown_time+1)==agent.shutting)
            # internal_state.append(np.array([agent.state.P])*1e-3)
        for agent in self.ess_agents:
            internal_state.append(np.array([agent.state.soc]))
        if len(internal_state) > 0:
            internal_state = np.hstack(internal_state)

        # past_t, net = self.past_t, deepcopy(self.net)
        # # past demand at every bus
        # df = pd.DataFrame(columns=['bus','area']+['{}'.format(t) for t in range(past_t)])
        # df.bus = net.load.bus
        # df.area = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
        # df.at[df[df.area==0].index, df.columns[2:]] = net.load[df.area==0].p_mw.values[:,None] * self.past_load_0
        # df.at[df[df.area==1].index, df.columns[2:]] = net.load[df.area==1].p_mw.values[:,None] * self.past_load_1
        # df.at[df[df.area==2].index, df.columns[2:]] = net.load[df.area==2].p_mw.values[:,None] * self.past_load_2
        # dfn = net.bus.join(df.set_index('bus'))
        # past_p_mw = dfn.groupby(dfn.index).agg(dict(zip(['{}'.format(t) for t in range(past_t)], [sum]*past_t))).values
        # # subtract PV generation
        # # past_p_mw[net.sgen[net.sgen.name=='PV 1'].bus] -= self.past_solar_0 * net.sgen[net.sgen.name=='PV 1'].p_mw.values
        # # past_p_mw[net.sgen[net.sgen.name=='PV 2'].bus] -= self.past_solar_1 * net.sgen[net.sgen.name=='PV 2'].p_mw.values
        # # past_p_mw[net.sgen[net.sgen.name=='PV 3'].bus] -= self.past_solar_2 * net.sgen[net.sgen.name=='PV 3'].p_mw.values
        # past_p_mw[net.sgen.iloc[2].bus] -= np.array(list(self.past_solar_0)) * net.sgen.iloc[2].p_mw
        # past_p_mw[net.sgen.iloc[3].bus] -= np.array(list(self.past_solar_1)) * net.sgen.iloc[3].p_mw
        # past_p_mw[net.sgen.iloc[4].bus] -= np.array(list(self.past_solar_2)) * net.sgen.iloc[4].p_mw
        # # subtract WP generation
        # # past_p_mw[net.sgen[net.sgen.name=='WP 1'].bus] -= self.past_wind_0 * net.sgen[net.sgen.name=='WP 1'].p_mw.values
        # # past_p_mw[net.sgen[net.sgen.name=='WP 2'].bus] -= self.past_wind_1 * net.sgen[net.sgen.name=='WP 2'].p_mw.values
        # # past_p_mw[net.sgen[net.sgen.name=='WP 3'].bus] -= self.past_wind_2 * net.sgen[net.sgen.name=='WP 3'].p_mw.values
        # past_p_mw[net.sgen.iloc[5].bus] -= np.array(list(self.past_wind_0)) * net.sgen.iloc[5].p_mw
        # past_p_mw[net.sgen.iloc[6].bus] -= np.array(list(self.past_wind_1)) * net.sgen.iloc[6].p_mw
        # past_p_mw[net.sgen.iloc[7].bus] -= np.array(list(self.past_wind_2)) * net.sgen.iloc[7].p_mw
        # # append price data
        # past_p_mw = np.append(past_p_mw, [list(self.past_price)], axis=0)

        external_state = np.hstack([
            np.array(list(self.past_solar_0)),
            np.array(list(self.past_solar_1)),
            np.array(list(self.past_solar_2)),
            np.array(list(self.past_wind_0)),
            np.array(list(self.past_wind_1)),
            np.array(list(self.past_wind_2)),
            np.array(list(self.past_load_0)),
            np.array(list(self.past_load_1)),
            np.array(list(self.past_load_2)),
            np.array(list(self.past_price))
        ])

        return np.hstack([internal_state, external_state]).astype('float32')
