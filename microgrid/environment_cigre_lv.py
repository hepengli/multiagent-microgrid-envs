import time
import gym
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
from gym.utils import seeding
from collections import deque
from microgrid.read_data import read_pickle_data
from microgrid.scenarios.cigre_network_lv import create_cigre_network_lv
from microgrid.core import DG, ESS, RES, Shunt, Transformer, Grid

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Environment_CIGRE_LV(gym.Env):
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
        # time steps
        self.total_timesteps = self.data['solar'].size
        # days
        self.days = self.data['solar'].size//24
        self.day = -2
        # network architecture
        self.net = create_cigre_network_lv()
        # how much history to store
        self.past_t = 24
        # seed
        self.seed()
        # list of agents
        MT_1 = DG('MT 1', bus='Bus R15', min_p_mw=0, max_p_mw=0.03, sn_mva=0.0375, control_q=True, min_pf=1e-10, cost_curve_coefs=[100, 75.7, 0.4712])
        FC_1 = DG('FC 1', bus='Bus R18', min_p_mw=0, max_p_mw=0.05, sn_mva=0.0625, control_q=True, min_pf=1e-10, cost_curve_coefs=[100, 51.6, 0.5011])
        MT_2 = DG('MT 2', bus='Bus C17', min_p_mw=0, max_p_mw=0.04, sn_mva=0.05, control_q=True, min_pf=1e-10, cost_curve_coefs=[100, 62.4, 0.4615])
        PV_1 = RES('PV 1', source='SOLAR', bus='Bus R16', sn_mva=0.01, control_q=False)
        PV_2 = RES('PV 2', source='SOLAR', bus='Bus R17', sn_mva=0.01, control_q=False)
        PV_3 = RES('PV 3', source='SOLAR', bus='Bus C19', sn_mva=0.01, control_q=False)
        WKA_1 = RES('WKA 1', source='WIND', bus='Bus R16', sn_mva=0.01, control_q=False)
        WKA_2 = RES('WKA 2', source='WIND', bus='Bus C20', sn_mva=0.01, control_q=False)
        BAT_1 = ESS('Battery 1', bus='Bus R11', min_p_mw=-0.06, max_p_mw=0.06, max_e_mwh=0.3, min_e_mwh=0.03, cost_curve_coefs=[0,0,0])
        BAT_2 = ESS('Battery 2', bus='Bus C13', min_p_mw=-0.06, max_p_mw=0.06, max_e_mwh=0.3, min_e_mwh=0.03, cost_curve_coefs=[0,0,0])
        GRID = Grid('GRID', bus='0', sn_mva=0.5, sell_discount=0.8)
        self.agents = [MT_1, FC_1, MT_2, PV_1, PV_2, PV_3, WKA_1, WKA_2, BAT_1, BAT_2, GRID]
        # reset
        ob = self.reset()

        # configure spaces
        action_space, action_shape = [], 0
        for agent in self.policy_agents:
            total_action_space = []
            # continuous action space
            if agent.action.range is not None:
                low, high = agent.action.range
                u_action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
                total_action_space.append(u_action_space)
                action_shape += u_action_space.shape[-1]

            action_space.extend(total_action_space)

        low = np.concatenate([ac.low for ac in action_space])
        high = np.concatenate([ac.high for ac in action_space])
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # observation space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=+np.inf, shape=(ob.shape[0],), dtype=np.float32)

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
        net = self.net
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
        net.load.scaling = self.data['load'][self.t]
        # update sgen info at all buese
        net.sgen.p_mw = [agent.state.P for agent in self.dg_agents + self.res_agents]
        net.sgen.q_mvar = [agent.state.Q for agent in self.dg_agents + self.res_agents]
        net.storage.p_mw = [agent.state.P for agent in self.ess_agents]

        # runpf
        try:
            pp.runpp(net)
        except:
            pass
        converge = net["converged"]
        if converge:
            # update grid state
            for agent in self.grid_agent:
                pgrid = net.res_ext_grid.p_mw.values[0]
                qgrid = net.res_ext_grid.q_mvar.values[0]
                agent.update_state(self.data['price'][self.t], pgrid, qgrid)
            # update resource agents' cost and safety
            for agent in self.resource_agents:
                agent.update_cost_safety()
            # update power flow safety
            overloading = np.maximum(net.res_line.loading_percent.values - 100, 0).sum()
            overvoltage = np.maximum(net.res_bus.vm_pu.values - 1.05, 0).sum()
            undervoltage = np.maximum(0.95 - net.res_bus.vm_pu.values, 0).sum()
            # if overvoltage > 0:
            #     print(overvoltage)
            # if undervoltage > 0:
            #     print(undervoltage)
            # reward and safety
            reward, safety = 0, 0
            safety += overvoltage + undervoltage + overloading / 100
            for agent in self.agents:
                reward -= agent.cost
                safety += agent.safety
                # print(agent.name, agent.state.P, agent.cost, agent.safety)
                # if agent.safety > 0:
                #     print(agent.name, agent.safety)
                assert agent.cost is not np.nan
                assert agent.safety is not np.nan
        else:
            reward = -200.0
            safety = 2.0
            print('Doesn\'t converge!')

        # update past observation
        self.past_load.append(self.data['load'][self.t])
        self.past_wind.append(self.data['wind'][self.t])
        self.past_solar.append(self.data['solar'][self.t])
        self.past_price.append(self.data['price_sigmoid'][self.t])

        # timestep counter
        self.t += 1
        if self.t >= self.total_timesteps:
            self.t = 0

        # info
        info = {
            's': safety * 10000, 
            'loading': net.res_line.loading_percent.values,
            'voltage': net.res_bus.vm_pu.values}
        reward -= safety * 1000

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
            agent.action.d = action[index:(index+agent.action.dim_d)].round()
            index += agent.action.dim_d
        # make sure we used all elements of action
        assert index == len(action)

    def _update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.type in ['DG', 'CL']:
            agent.update_state()
        elif agent.type == 'ESS':
            agent.update_state()
        elif agent.type == 'SOLAR':
            agent.update_state(self.data['solar'][self.t])
        elif agent.type == 'WIND':
            agent.update_state(self.data['wind'][self.t])
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
            self.past_load = deque(data['load'][t-past_t:t], maxlen=past_t)
            self.past_wind = deque(data['wind'][t-past_t:t], maxlen=past_t)
            self.past_solar = deque(data['solar'][t-past_t:t], maxlen=past_t)
            self.past_price = deque(data['price_sigmoid'][t-past_t:t], maxlen=past_t)

            self.past_load = deque(data['load'][t-past_t:t], maxlen=past_t)
            self.past_wind = deque(data['wind'][t-past_t:t], maxlen=past_t)
            self.past_solar = deque(data['solar'][t-past_t:t], maxlen=past_t)
            self.past_price = deque(data['price_sigmoid'][t-past_t:t], maxlen=past_t)
        else:
            self.past_load = deque(np.hstack([data['load'][t-past_t:], data['load'][:t]]), maxlen=past_t)
            self.past_wind = deque(np.hstack([data['wind'][t-past_t:], data['wind'][:t]]), maxlen=past_t)
            self.past_solar = deque(np.hstack([data['solar'][t-past_t:], data['solar'][:t]]), maxlen=past_t)
            self.past_price = deque(np.hstack([data['price_sigmoid'][t-past_t:], data['price_sigmoid'][:t]]), maxlen=past_t)

        return self._get_obs()

    def _get_obs(self):
        nongraph_state = []
        nongraph_state.append(np.array([(self.t%24) / 24.]))
        nongraph_state.append(np.array(self.past_price))
        # for agent in self.dg_agents:
            # internal_state.append(np.arange(2)==agent.state.uc)
            # internal_state.append(np.arange(agent.startup_time+1)==agent.starting)
            # internal_state.append(np.arange(agent.shutdown_time+1)==agent.shutting)
            # internal_state.append(np.array([agent.state.P])*1e-3)
        for agent in self.ess_agents:
            nongraph_state.append(np.array([agent.state.soc]))
        if len(nongraph_state) > 0:
            nongraph_state = np.hstack(nongraph_state)

        # net = self.net
        # net.bus['bus'] = net.bus.index

        # bus_group = net.load.groupby('bus').sum()
        # bus_group = pd.merge(net.bus, bus_group, on="bus", how="outer").fillna(0.0)
        # past_pload = bus_group.p_mw.values[:,None] * np.array(self.past_load)
        # past_qload = bus_group.q_mvar.values[:,None] * np.array(self.past_load)

        # bus_group = pd.merge(net.bus, net.sgen[:9], on="bus", how="outer").fillna(0.0)
        # pv_bus = bus_group.type_y=='PV'
        # wp_bus = bus_group.type_y=='WP'
        # bus_group = bus_group.groupby('bus').sum()
        # past_psolar = (bus_group.p_mw * pv_bus).values[:,None] * np.array(self.past_solar)
        # past_pwind = (bus_group.p_mw * wp_bus).values[:,None] * np.array(self.past_wind)
        # past_pload = past_pload - past_psolar - past_pwind
        # graph_state = np.hstack([past_pload, past_qload,]).ravel()

        # return np.concatenate([graph_state, nongraph_state]).astype('float32')


        # # # past demand at every bus
        # # df = pd.DataFrame(columns=['bus','area']+['{}'.format(t) for t in range(past_t)])
        # # df.bus = net.load.bus
        # # df.area = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
        # # df.at[df[df.area==0].index, df.columns[2:]] = net.load[df.area==0].p_mw.values[:,None] * self.past_load_0
        # # df.at[df[df.area==1].index, df.columns[2:]] = net.load[df.area==1].p_mw.values[:,None] * self.past_load_1
        # # df.at[df[df.area==2].index, df.columns[2:]] = net.load[df.area==2].p_mw.values[:,None] * self.past_load_2
        # # dfn = net.bus.join(df.set_index('bus'))
        # # past_p_mw = dfn.groupby(dfn.index).agg(dict(zip(['{}'.format(t) for t in range(past_t)], [sum]*past_t))).values
        # # # subtract PV generation
        # # # past_p_mw[net.sgen[net.sgen.name=='PV 1'].bus] -= self.past_solar_0 * net.sgen[net.sgen.name=='PV 1'].p_mw.values
        # # # past_p_mw[net.sgen[net.sgen.name=='PV 2'].bus] -= self.past_solar_1 * net.sgen[net.sgen.name=='PV 2'].p_mw.values
        # # # past_p_mw[net.sgen[net.sgen.name=='PV 3'].bus] -= self.past_solar_2 * net.sgen[net.sgen.name=='PV 3'].p_mw.values
        # # past_p_mw[net.sgen.iloc[2].bus] -= np.array(list(self.past_solar_0)) * net.sgen.iloc[2].p_mw
        # # past_p_mw[net.sgen.iloc[3].bus] -= np.array(list(self.past_solar_1)) * net.sgen.iloc[3].p_mw
        # # past_p_mw[net.sgen.iloc[4].bus] -= np.array(list(self.past_solar_2)) * net.sgen.iloc[4].p_mw
        # # # subtract WP generation
        # # # past_p_mw[net.sgen[net.sgen.name=='WP 1'].bus] -= self.past_wind_0 * net.sgen[net.sgen.name=='WP 1'].p_mw.values
        # # # past_p_mw[net.sgen[net.sgen.name=='WP 2'].bus] -= self.past_wind_1 * net.sgen[net.sgen.name=='WP 2'].p_mw.values
        # # # past_p_mw[net.sgen[net.sgen.name=='WP 3'].bus] -= self.past_wind_2 * net.sgen[net.sgen.name=='WP 3'].p_mw.values
        # # past_p_mw[net.sgen.iloc[5].bus] -= np.array(list(self.past_wind_0)) * net.sgen.iloc[5].p_mw
        # # past_p_mw[net.sgen.iloc[6].bus] -= np.array(list(self.past_wind_1)) * net.sgen.iloc[6].p_mw
        # # past_p_mw[net.sgen.iloc[7].bus] -= np.array(list(self.past_wind_2)) * net.sgen.iloc[7].p_mw
        # # # append price data
        # # past_p_mw = np.append(past_p_mw, [list(self.past_price)], axis=0)

        external_state = np.hstack([
            np.array(self.past_solar),
            np.array(self.past_wind),
            np.array(self.past_load),
            # np.array(self.past_price),
        ])

        return np.hstack([nongraph_state, external_state]).astype('float32')
