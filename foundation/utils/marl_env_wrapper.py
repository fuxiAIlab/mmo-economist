# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
Wrapper for making the gather-trade-build environment an OpenAI compatible environment.
This can then be used with reinforcement learning frameworks such as RLlib.
"""

import os
import pickle
import random
import warnings

import numpy as np
from foundation.scenarios.utils import social_metrics
import foundation
from gym import spaces
from gym.spaces import Box,Dict as GymDict
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv

_BIG_NUMBER = 1e20



def recursive_list_to_np_array(d):
    if isinstance(d, dict):
        new_d = {}
        for k, v in d.items():
            if isinstance(v, list):
                new_d[k] = np.array(v)
            elif isinstance(v, dict):
                new_d[k] = recursive_list_to_np_array(v)
            elif isinstance(v, (float, int, np.floating, np.integer)):
                new_d[k] = np.array([v])
            elif isinstance(v, np.ndarray):
                new_d[k] = v
            else:
                raise AssertionError
        return new_d
    raise AssertionError


def pretty_print(dictionary):
    for key in dictionary:
        print("{:15s}: {}".format(key, dictionary[key].shape))
    print("\n")

import yaml
class MarlEnvWrapper(MultiAgentEnv):
    """
    Environment wrapper for RLlib. It sub-classes MultiAgentEnv.
    This wrapper adds the action and observation space to the environment,
    and adapts the reset and step functions to run with RLlib.
    """

    def __init__(self, env_config, verbose=False):
        self.env_config_dict = env_config#["env_config_dict"]
        # Adding env id in the case of multiple environments
        if hasattr(env_config, "worker_index"):
            self.env_id = (
                1*
                # env_config["num_envs_per_worker"] *
                (env_config.worker_index - 1)
            ) + env_config.vector_index
        else:
            self.env_id = None

        # config_path =  '/data/private/yuanxi/mmo-economist/experiments/marl_config_50_50.yaml'
        # config_path =  '/home/game/env/mmo-economist/experiments/marl_config_80_20.yaml'
        path=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(path, 'experiments', 'marl_config_80_20.yaml')
        with open(config_path, "r") as f:
            run_configuration = yaml.safe_load(f)

        trainer_config = run_configuration.get("trainer")
        env_config = {
            'disable_env_checking': True,
            "env_config_dict": run_configuration.get("env"),
            "num_envs_per_worker": trainer_config.get("num_envs_per_worker"),
        }
        self.env_config_dict=env_config
        self.env_config_dict=run_configuration.get("env_args")
        # self.env_config_dict=run_configuration.get("env")
        self.env = foundation.make_env_instance(**self.env_config_dict)
        self.verbose = verbose
        self.sample_agent_idx = str(self.env.all_agents[0].idx)

        obs = self.env.reset()
        self.observation_space =GymDict({'obs':spaces.Box(low=-1e20,high=1e20,
                                                          shape= obs["0"].shape,
                                                          dtype=np.dtype("float64"))})
        self.observation_space_pl = GymDict({'obs': spaces.Box(low=-1e20, high=1e20,
                                                            shape=obs["p"].shape,
                                                            dtype=np.dtype("float64"))})
        '''
        # used for same obs space of planner and agent 
        
        self.observation_space=GymDict({"obs":
                                            spaces.Box(low=-1e20,high=1e20,shape= (1069,),
                                                       dtype=np.dtype("float64"))
                                        })
        '''

        if self.env.world.agents[0].multi_action_mode:
            self.action_space = spaces.MultiDiscrete(
                self.env.get_agent(self.sample_agent_idx).action_spaces
            )
            self.action_space.dtype = np.int64
            self.action_space.nvec = self.action_space.nvec.astype(np.int64)

        else:
            self.action_space = spaces.Discrete(
                self.env.get_agent(self.sample_agent_idx).action_spaces
            )
            self.action_space.dtype = np.int64

        if self.env.world.planner.multi_action_mode:
            self.action_space_pl = spaces.MultiDiscrete(
                self.env.get_agent("p").action_spaces
            )
            self.action_space_pl.dtype = np.int64
            self.action_space_pl.nvec = self.action_space_pl.nvec.astype(
                np.int64)

        else:
            self.action_space_pl = spaces.Discrete(
                self.env.get_agent("p").action_spaces
            )
            self.action_space_pl.dtype = np.int64

        #marllib init
        self.env_config=self.env_config_dict
        self.num_agents=self.n_agents+1

        self.agents=['a_'+str(i) for i in range(self.n_agents)]+['p_0']
        self.ori_agents = [str(i) for i in range(10)]+['p']

        self.policy_mapping_dict = {
            "all_scenario": {
                "description": "policy mapping for agent and planner",
                "team_prefix":('p',''),
                "all_agents_one_policy": True,
                "one_agent_one_policy": True,
            },

        }

        self._seed = None
        # if self.verbose:
        #     print("[EnvWrapper] Spaces")
        #     print("[EnvWrapper] Obs (a)   ")
        #     print(self.observation_space)
        #     print("[EnvWrapper] Obs (p)   ")
        #     print(self.observation_space_pl)
        #     print("[EnvWrapper] Action (a)", self.action_space)
        #     print("[EnvWrapper] Action (p)", self.action_space_pl)


    def close(self):
        pass
        # self.env.close()
    def render(self, mode=None):
        return False
        # self.env.render()
        # time.sleep(0.05)
        # return True

    def get_env_info(self):
        env_info = {
            "space_obs": {'p':self.observation_space_pl,'a':self.observation_space,},#self.observation_space,
            "space_act":  {'p':self.action_space_pl,'a':self.action_space,},#self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 3000,
            "policy_mapping_info": self.policy_mapping_dict
        }
        return env_info


    @property
    def pickle_file(self):
        if self.env_id is None:
            return "game_object.pkl"
        return "game_object_{:03d}.pkl".format(self.env_id)

    def save_game_object(self, save_dir):
        assert os.path.isdir(save_dir)
        path = os.path.join(save_dir, self.pickle_file)
        with open(path, "wb") as F:
            pickle.dump(self.env, F)

    def load_game_object(self, save_dir):
        assert os.path.isdir(save_dir)
        path = os.path.join(save_dir, self.pickle_file)
        with open(path, "rb") as F:
            self.env = pickle.load(F)

    @property
    def n_agents(self):
        return self.env.n_agents

    @property
    def summary(self):
        last_completion_metrics = self.env.previous_episode_metrics
        if last_completion_metrics is None:
            return {}
        last_completion_metrics["completions"] = int(self.env._completions)
        return last_completion_metrics

    def get_seed(self):
        return int(self._seed)

    def seed(self, seed):
        # Using the seeding utility from OpenAI Gym
        # https://github.com/openai/gym/blob/master/gym/utils/seeding.py
        _, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as an uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31

        if self.verbose:
            print(
                "[EnvWrapper] twisting seed {} -> {} -> {} (final)".format(
                    seed, seed1, seed2
                )
            )

        seed = int(seed2)
        np.random.seed(seed2)
        random.seed(seed2)
        self._seed = seed2

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        if 'p' in obs.keys(): _=obs.pop('p')
        return {k: {'obs': obs[k]} for k in obs.keys()}

    def step(self, action_dict):
        if 'p' in action_dict.keys(): _ = action_dict.pop('p')
        obs, rew, done, info = self.env.step(action_dict)
        if 'p' in obs.keys(): _ = obs.pop('p')
        if 'p' in rew.keys(): _ = rew.pop('p')
        if 'p' in done.keys(): _ = done.pop('p')
        if 'p' in info.keys(): _ = info.pop('p')

        # print({k:round(rew[k],3) for k in rew.keys()})
        info = {k: {'res': np.array([-1.0, -1.0, -1.0]), "training_enabled": True} for k in action_dict.keys()}
        if self.env._steps_in_period == 0:
            metrics = self.env.scenario_metrics()
            profit, equality, capability = metrics['social/profitability'], \
                                           metrics['social/equality'], \
                                           metrics['social/capability_avg']
            info['0']['res'] = np.array([profit, equality, capability])
        else:
            capability_endowments = np.array(
                [agent.endogenous['Capability'] for agent in self.env.world.agents])
            equality = social_metrics.get_equality(
                capability_endowments)
            info['0']['res'] = np.array([-1.0, equality, -1.0])
        return {k: {'obs': obs[k]} for k in obs.keys()}, rew, done, info #{}
