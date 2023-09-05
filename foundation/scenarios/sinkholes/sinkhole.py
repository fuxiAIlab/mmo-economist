# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from copy import deepcopy

import numpy as np

from foundation.base.base_env import BaseEnvironment, scenario_registry, resource_registry
from foundation.scenarios.utils import rewards, social_metrics


@scenario_registry.add
class Sinkhole(BaseEnvironment):
    name = "scenarios/sinkhole"
    agent_subclasses = ["BasicPlayer", "BasicPlanner"]
    required_entities = ["Exp", "Mat", "Token",
                         "Currency", "Labor", "Capability"]

    def __init__(
        self,
        *base_env_args,
        planner_gets_spatial_info=True,
        full_observability=False,
        player_observation_range=5,
        base_launch_plan={
            "Exp": 5,
            "Mat": 5,
            "Token": 5
        },
        timesteps_for_force_refresh_launch=1000,
        adjustemt_type='none',
        normal_wear_and_tear_rate=0.05,
        checker_source_blocks=False,
        starting_player_token=0,
        starting_player_currency=0,
        isoelastic_eta=0.23,
        energy_cost=0.21,
        energy_warmup_constant=0,
        energy_warmup_method="decay",
        player_monetary_cost_dist='pareto',
        player_nonmonetary_cost_dist='normal',
        player_utility_income_fxrate=0.1,
        mixing_weight_gini_vs_coin=0.0,
        **base_env_kwargs
    ):
        super().__init__(*base_env_args, **base_env_kwargs)

        # Whether agents receive spatial information in their observation tensor
        self._planner_gets_spatial_info = bool(planner_gets_spatial_info)

        # Whether the (non-planner) agents can see the whole world map
        self._full_observability = bool(full_observability)

        self._player_observation_range = int(
            player_observation_range)

        self._durations = 0
        self._steps_in_duration = 0
        self._checker_source_blocks = bool(checker_source_blocks)
        c, r = np.meshgrid(
            np.arange(self.world_size[1]) % 2, np.arange(
                self.world_size[0]) % 2
        )
        self._checker_mask = (r + c) == 1
        m = 2 if self._checker_source_blocks else 1

        if not isinstance(base_launch_plan, dict):
            base_launch_plan = eval(base_launch_plan)
        assert isinstance(base_launch_plan, dict)
        self._base_launch_plan = base_launch_plan

        self._curr_launch_plan = deepcopy(self._base_launch_plan)
        self._last_launch_plan = deepcopy(self._base_launch_plan)

        self._source_maps = self.init_map_layout()
        self._normal_wear_and_tear_rate = normal_wear_and_tear_rate

        # How much coin do agents begin with at upon reset
        self.starting_player_token = float(starting_player_token)
        assert self.starting_player_token >= 0.0

        self.starting_player_currency = float(starting_player_currency)
        assert self.starting_player_currency >= 0.0

        # Controls the diminishing marginal utility of coin.
        # isoelastic_eta=0 means no diminishing utility.
        self.isoelastic_eta = float(isoelastic_eta)
        assert 0.0 <= self.isoelastic_eta <= 1.0

        # The amount that labor is weighted in utility computation
        # (once annealing is finished)
        self.energy_cost = float(energy_cost)
        assert self.energy_cost >= 0

        # What value to use for calculating the progress of energy annealing
        # If method = 'decay': #completed episodes
        # If method = 'auto' : #timesteps where avg. agent reward > 0
        self.energy_warmup_method = energy_warmup_method.lower()
        assert self.energy_warmup_method in ["decay", "auto"]
        # Decay constant for annealing to full energy cost
        # (if energy_warmup_constant == 0, there is no annealing)
        self.energy_warmup_constant = float(energy_warmup_constant)
        assert self.energy_warmup_constant >= 0
        self._auto_warmup_integrator = 0

        # Which social welfare function to use
        # self.planner_reward_type = str(planner_reward_type).lower()

        # How much to weight equality if using SWF=eq*prod:
        # 0 -> SWF=eq*prod
        # 1 -> SWF=prod
        self.mixing_weight_gini_vs_coin = float(mixing_weight_gini_vs_coin)
        assert 0 <= self.mixing_weight_gini_vs_coin <= 1.0

        # Use this to calculate marginal changes and deliver that as reward
        self.init_optimization_metric = {
            agent.idx: 0 for agent in self.all_agents}
        self.prev_optimization_metric = {
            agent.idx: 0 for agent in self.all_agents}
        self.curr_optimization_metric = {
            agent.idx: 0 for agent in self.all_agents}
    
        self._timesteps_for_force_refresh_launch = timesteps_for_force_refresh_launch

        # Initialize the agent's monetary and nonmonetary utility cost sensitivity
        if player_monetary_cost_dist.lower() == 'pareto':
            pareto_samples = np.random.pareto(4, size=(100000, self.n_agents))
            clipped_samples = np.minimum(1, pareto_samples)
            sorted_clipped_samples = np.sort(clipped_samples, axis=1)
            average_ranked_samples = sorted_clipped_samples.mean(axis=0)
            np.random.shuffle(average_ranked_samples)
            self._player_monetary_cost_sensitivities = 1-average_ranked_samples
        elif player_monetary_cost_dist.lower() == 'normal':
            normal_samples = np.random.normal((1+0)/2, (1-0)/3, self.n_agents)
            self._player_monetary_cost_sensitivities = np.clip(
                normal_samples, 0, 1)
        else:
            raise NotImplementedError

        if player_nonmonetary_cost_dist.lower() == 'pareto':
            pareto_samples = np.random.pareto(4, size=(100000, self.n_agents))
            clipped_samples = np.minimum(1, pareto_samples)
            sorted_clipped_samples = np.sort(clipped_samples, axis=1)
            average_ranked_samples = sorted_clipped_samples.mean(axis=0)
            np.random.shuffle(average_ranked_samples)
            self._player_nonmonetary_cost_sensitivities = 1-average_ranked_samples
        elif player_nonmonetary_cost_dist.lower() == 'normal':
            normal_samples = np.random.normal((1+0)/2, (1-0)/3, self.n_agents)
            self._player_nonmonetary_cost_sensitivities = np.clip(
                normal_samples, 0, 1)
        else:
            raise NotImplementedError

        for agent in self.world.agents:
            agent.set_cost_sensitivity(
                monetary_cost_sensitivity=self._player_monetary_cost_sensitivities[agent.idx],
                nonmonetary_cost_sensitivity=self._player_nonmonetary_cost_sensitivities[agent.idx]
            )

        self._player_utility_income_fxrate = player_utility_income_fxrate
        assert isinstance(adjustemt_type, str)

        self._adjustemt_type = adjustemt_type.lower()
        assert self._adjustemt_type in ['none', 'planner']

    @property
    def base_launch_plan(self):
        return self._base_launch_plan

    @property
    def curr_launch_plan(self):
        return self._curr_launch_plan

    @property
    def last_launch_plan(self):
        return self._last_launch_plan

    # 初始化地图
    def init_map_layout(self):
        self._launch_plan = {
            k: (int(v * self.world_size[0] *
                    self.world_size[1]) if isinstance(v, float) and
                (v >= 0 and v <= 1) else v)
            for k, v in self._curr_launch_plan.items()
        }

        layouts = {k: [] for k in self._launch_plan.keys()}
        used_loc = []
        for landmark, n in self._launch_plan.items():
            while n > 0:
                r = np.random.randint(0, self.world_size[0])
                c = np.random.randint(0, self.world_size[1])
                if (r, c) not in used_loc:
                    n -= 1
                    used_loc.append((r, c))
                    layouts[landmark].append((r, c))
        _source_maps = {}
        for k, v in layouts.items():
            v_ = np.zeros((self.world_size[0], self.world_size[1]))
            if len(v) == 0:
                continue
            v = np.array(v)
            v_[v[:, 0], v[:, 1]] = 1
            _source_maps[k] = v_
            self.world.maps.set(str(k), v_)
            self.world.maps.set(str(k)+"SourceBlock", v_)
        return _source_maps

    @property
    def energy_weight(self):
        """
        Energy annealing progress. Multiply with self.energy_cost to get the
        effective energy coefficient.
        """
        if self.energy_warmup_constant <= 0.0:
            return 1.0

        if self.energy_warmup_method == "decay":
            return float(1.0 - np.exp(-self._completions / self.energy_warmup_constant))

        if self.energy_warmup_method == "auto":
            return float(
                1.0
                - np.exp(-self._auto_warmup_integrator /
                         self.energy_warmup_constant)
            )

        raise NotImplementedError

    def get_current_optimization_metrics(self):
        """
        Compute optimization metrics based on the current state. Used to compute reward.

        Returns:
            curr_optimization_metric (dict): A dictionary of {agent.idx: metric}
                with an entry for each agent (including the planner) in the env.
        """
        curr_optimization_metric = {}
        # (for agents)
        for agent in self.world.agents:
            curr_optimization_metric[agent.idx] = rewards.isoelastic_utility_for_player(
                income=agent.state["endogenous"]["Capability"],
                monetary_cost=self.starting_player_currency -
                agent.total_endowment("Currency"),
                nonmonetary_cost=agent.state["endogenous"]["Labor"],
                isoelastic_eta=self.isoelastic_eta,
                labor_coefficient=self.energy_weight * self.energy_cost,
                income_exchange_rate=self._player_utility_income_fxrate,
                monetary_cost_sensitivity=agent.monetary_cost_sensitivity,
                nonmonetary_cost_sensitivity=agent.nonmonetary_cost_sensitivity
            )

        # (for the planner)
        curr_optimization_metric[self.world.planner.idx] = rewards.utility_for_planner(
            monetary_incomes=np.array(
                [self.starting_player_currency - agent.total_endowment("Currency")
                 for agent in self.world.agents]
            ),
            nonmonetary_incomes=np.array(
                [agent.state["endogenous"]["Capability"]
                 for agent in self.world.agents]
            ),
            equality_weight=1 - self.mixing_weight_gini_vs_coin
        )

        return curr_optimization_metric

    # The following methods must be implemented for each scenario
    # -----------------------------------------------------------

    def reset_starting_layout(self):
        """
        Part 1/2 of scenario reset. This method handles resetting the state of the
        environment managed by the scenario (i.e. resource & landmark layout).

        Here, generate a resource source layout consistent with target parameters.
        """
        self.world.maps.clear()
        self._source_maps = self.init_map_layout()

    def reset_agent_states(self):
        """
        Part 2/2 of scenario reset. This method handles resetting the state of the
        agents themselves (i.e. inventory, locations, etc.).

        Here, empty inventories, give mobile agents any starting coin, and place them
        in random accessible locations to start.
        """
        self.world.clear_agent_locs()

        for agent in self.world.agents:
            # Clear everything to start with
            agent.state["inventory"] = {k: 0 for k in agent.inventory.keys()}
            agent.state["escrow"] = {k: 0 for k in agent.inventory.keys()}
            agent.state["endogenous"] = {k: 0 for k in agent.endogenous.keys()}
            # Add starting coin
            agent.state["inventory"]["Token"] = float(
                self.starting_player_token)
            agent.state["inventory"]["Currency"] = float(
                self.starting_player_currency)

        # Clear everything for the planner
        self.world.planner.state["inventory"] = {
            k: 0 for k in self.world.planner.inventory.keys()
        }
        self.world.planner.state["escrow"] = {
            k: 0 for k in self.world.planner.escrow.keys()
        }

        # Place the agents randomly in the world
        for agent in self.world.get_random_order_agents():
            r = np.random.randint(0, self.world_size[0])
            c = np.random.randint(0, self.world_size[1])
            n_tries = 0
            while not self.world.can_agent_occupy(r, c, agent):
                r = np.random.randint(0, self.world_size[0])
                c = np.random.randint(0, self.world_size[1])
                n_tries += 1
                if n_tries > 200:
                    raise TimeoutError
            self.world.set_agent_loc(agent, r, c)

    def scenario_step(self):
        """
        Update the state of the world according to whatever rules this scenario
        implements.

        This gets called in the 'step' method (of base_env) after going through each
        component step and before generating observations, rewards, etc.

        In this class of scenarios, the scenario step handles stochastic resource
        regeneration.
        """

        resources = [
            x for x in self.world.resources if resource_registry.get(x).collectible]

        all_resource_map = np.zeros(self.world.maps.size)

        for resource in resources:
            resource_map = self.world.maps.get(resource)
            resource_source_blocks = self.world.maps.get(
                resource + "SourceBlock")

            self.world.maps.set(
                resource,
                np.minimum(resource_map, resource_source_blocks)
            )
            self.world.maps.set(
                str(resource)+"SourceBlock",
                np.minimum(resource_map, resource_source_blocks)
            )

            all_resource_map += np.minimum(resource_map,
                                           resource_source_blocks)

        self._steps_in_duration += 1
        total_launch = sum([v for k, v in self._launch_plan.items()])

        # 投放全部被获取 or 超过一定steps 开始新的投放周期
        if np.sum(all_resource_map) <= max(1, self._normal_wear_and_tear_rate * total_launch) or \
            self._steps_in_duration >= self._timesteps_for_force_refresh_launch:
                
            self._durations += 1
            self._steps_in_duration = 0
            self._last_launch_plan = deepcopy(self._curr_launch_plan)
            if self._adjustemt_type == 'planner':
                self._curr_launch_plan = {k: int(self._base_launch_plan[k]*v) for k, v in self.get_component(
                        "LaunchReadjustment").launch_adjustment.items()}
            elif self._adjustemt_type == 'none':
                pass
            else:
                raise NotImplementedError
            self.get_component("LaunchReadjustment").start_new_launch_adjustment()
            self.reset_starting_layout()

    def generate_observations(self):
        """
        Generate observations associated with this scenario.

        A scenario does not need to produce observations and can provide observations
        for only some agent types; however, for a given agent type, it should either
        always or never yield an observation. If it does yield an observation,
        that observation should always have the same structure/sizes!

        Returns:
            obs (dict): A dictionary of {agent.idx: agent_obs_dict}. In words,
                return a dictionary with an entry for each agent (which can including
                the planner) for which this scenario provides an observation. For each
                entry, the key specifies the index of the agent and the value contains
                its associated observation dictionary.

        Here, non-planner agents receive spatial observations (depending on the env
        config) as well as the contents of their inventory and endogenous quantities.
        The planner also receives spatial observations (again, depending on the env
        config) as well as the inventory of each of the mobile agents.
        """
        obs = {}
        curr_map = self.world.maps.state

        # owner_map = self.world.maps.owner_state
        loc_map = self.world.loc_map
        # agent_idx_maps = np.concatenate([owner_map, loc_map[None, :, :]], axis=0)
        agent_idx_maps = np.concatenate([loc_map[None, :, :]], axis=0)
        agent_idx_maps += 2
        agent_idx_maps[agent_idx_maps == 1] = 0

        agent_locs = {
            str(agent.idx): {
                "loc-row": agent.loc[0] / self.world_size[0],
                "loc-col": agent.loc[1] / self.world_size[1],
            }
            for agent in self.world.agents
        }
        agent_invs = {
            str(agent.idx): {
                "inventory-" + k: v * self.inv_scale for k, v in agent.inventory.items()
            }
            for agent in self.world.agents
        }

        agent_escs = {
            str(agent.idx): {
                "escrow-" + k: v * self.inv_scale for k, v in agent.escrow.items()
            }
            for agent in self.world.agents
        }

        agent_ends = {
            str(agent.idx): {
                "endogenous-" + k: v * self.inv_scale *0.1 for k, v in agent.endogenous.items()
            }
            for agent in self.world.agents
        }

        agent_util = {
            str(agent.idx): {
                "utility-monetary_cost_sensitivity": agent.monetary_cost_sensitivity,
                "utility-nonmonetary_cost_sensitivity": agent.nonmonetary_cost_sensitivity
            }
            for agent in self.world.agents
        }

        obs[self.world.planner.idx] = {
            "inventory-" + k: v * self.inv_scale
            for k, v in self.world.planner.inventory.items()
        }

        obs[self.world.planner.idx].update({
            "escrow-" + k: v * self.inv_scale
            for k, v in self.world.planner.escrow.items()
        })

        obs[self.world.planner.idx].update({
            "endogenous-" + k: v * self.inv_scale *0.1
            for k, v in self.world.planner.endogenous.items()
        })

        if self._planner_gets_spatial_info:
            obs[self.world.planner.idx].update(
                dict(map=curr_map, idx_map=agent_idx_maps)
            )

        # Mobile agents see the full map. Convey location info via one-hot map channels.
        if self._full_observability:
            for agent in self.world.agents:
                my_map = np.array(agent_idx_maps)
                my_map[my_map == int(agent.idx) + 2] = 1
                sidx = str(agent.idx)
                obs[sidx] = {"map": curr_map, "idx_map": my_map}
                obs[sidx].update(agent_invs[sidx])
                obs[sidx].update(agent_escs[sidx])
                obs[sidx].update(agent_ends[sidx])
                obs[sidx].update(agent_util[sidx])

        # Mobile agents only see within a window around their position
        else:
            w = (
                self._player_observation_range
            )  # View halfwidth (only applicable without full observability)

            padded_map = np.pad(
                curr_map,
                [(0, 1), (w, w), (w, w)],
                mode="constant",
                constant_values=[(0, 1), (0, 0), (0, 0)],
            )

            padded_idx = np.pad(
                agent_idx_maps,
                [(0, 0), (w, w), (w, w)],
                mode="constant",
                constant_values=[(0, 0), (0, 0), (0, 0)],
            )

            for agent in self.world.agents:
                r, c = [c + w for c in agent.loc]
                visible_map = padded_map[
                    :, (r - w): (r + w + 1), (c - w): (c + w + 1)
                ]
                visible_idx = np.array(
                    padded_idx[:, (r - w): (r + w + 1), (c - w): (c + w + 1)]
                )

                visible_idx[visible_idx == int(agent.idx) + 2] = 1

                sidx = str(agent.idx)

                obs[sidx] = {"map": visible_map, "idx_map": visible_idx}
                obs[sidx].update(agent_locs[sidx])
                obs[sidx].update(agent_invs[sidx])
                obs[sidx].update(agent_escs[sidx])
                obs[sidx].update(agent_ends[sidx])
                obs[sidx].update(agent_util[sidx])

                # Agent-wise planner info (gets crunched into the planner obs in the
                # base scenario code)
                obs["p" + sidx] = agent_invs[sidx]
                obs["p" + sidx].update(agent_escs[sidx])
                obs["p" + sidx].update(agent_ends[sidx])
                obs["p" + sidx].update(agent_util[sidx])
                if self._planner_gets_spatial_info:
                    obs["p" + sidx].update(agent_locs[sidx])

        return obs

    def compute_reward(self):
        """
        Apply the reward function(s) associated with this scenario to get the rewards
        from this step.

        Returns:
            rew (dict): A dictionary of {agent.idx: agent_obs_dict}. In words,
                return a dictionary with an entry for each agent in the environment
                (including the planner). For each entry, the key specifies the index of
                the agent and the value contains the scalar reward earned this timestep.

        Rewards are computed as the marginal utility (agents) or marginal social
        welfare (planner) experienced on this timestep. Ignoring discounting,
        this means that agents' (planner's) objective is to maximize the utility
        (social welfare) associated with the terminal state of the episode.
        """

        # "curr_optimization_metric" hasn't been updated yet, so it gives us the
        # utility from the last step.
        utility_at_end_of_last_time_step = deepcopy(
            self.curr_optimization_metric)

        # compute current objectives and store the values
        self.curr_optimization_metric = self.get_current_optimization_metrics()

        # reward = curr - prev objectives
        rew = {
            k: float(v - utility_at_end_of_last_time_step[k])
            for k, v in self.curr_optimization_metric.items()
        }

        # store the previous objective values
        self.prev_optimization_metric.update(utility_at_end_of_last_time_step)

        # Automatic Energy Cost Annealing
        # -------------------------------
        avg_agent_rew = np.mean([rew[a.idx] for a in self.world.agents])
        # Count the number of timesteps where the avg agent reward was > 0
        if avg_agent_rew > 0:
            self._auto_warmup_integrator += 1

        return rew

    # Optional methods for customization
    # ----------------------------------

    def additional_reset_steps(self):
        """
        Extra scenario-specific steps that should be performed at the end of the reset
        cycle.

        For each reset cycle...
            First, reset_starting_layout() and reset_agent_states() will be called.

            Second, <component>.reset() will be called for each registered component.

            Lastly, this method will be called to allow for any final customization of
            the reset cycle.

        For this scenario, this method resets optimization metric trackers.
        """
        # compute current objectives
        curr_optimization_metric = self.get_current_optimization_metrics()

        self.curr_optimization_metric = deepcopy(curr_optimization_metric)
        self.init_optimization_metric = deepcopy(curr_optimization_metric)
        self.prev_optimization_metric = deepcopy(curr_optimization_metric)

    def scenario_metrics(self):
        """
        Allows the scenario to generate metrics (collected along with component metrics
        in the 'metrics' property).

        To have the scenario add metrics, this function needs to return a dictionary of
        {metric_key: value} where 'value' is a scalar (no nesting or lists!)

        Here, summarize social metrics, endowments, utilities, and labor cost annealing
        """
        metrics = dict()

        capability_endowments = np.array(
            [agent.endogenous['Capability'] for agent in self.world.agents]
        )

        currency_endowments = np.array(
            [self.starting_player_currency -
                agent.total_endowment("Currency") for agent in self.world.agents]
        )

        metrics["social/productivity"] = social_metrics.get_productivity(
            currency_endowments
        )
        metrics["social/equality"] = social_metrics.get_equality(
            capability_endowments
        )

        metrics["social_welfare/planner"] = rewards.utility_for_planner(
            monetary_incomes=currency_endowments,
            nonmonetary_incomes=capability_endowments,
            equality_weight=1.0
        )

        for agent in self.all_agents:
            for resource, quantity in agent.inventory.items():
                metrics[
                    "endow/{}/{}".format(agent.idx, resource)
                ] = agent.total_endowment(resource)

            if agent.endogenous is not None:
                for resource, quantity in agent.endogenous.items():
                    metrics["endogenous/{}/{}".format(
                        agent.idx, resource)] = quantity

            metrics["util/{}".format(agent.idx)] = self.curr_optimization_metric[
                agent.idx
            ]

        # Labor weight
        metrics["labor/weighted_cost"] = self.energy_cost * self.energy_weight
        metrics["labor/warmup_integrator"] = int(self._auto_warmup_integrator)

        return metrics
