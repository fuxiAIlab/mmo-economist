# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np
from foundation.base.base_component import BaseComponent, component_registry


@component_registry.add
class Upgrade(BaseComponent):
    """
    Allows mobile agents to build house landmarks in the world using stone and wood,
    earning income.

    Can be configured to include heterogeneous building skill where agents earn
    different levels of income when building.

    Args:
        payment (int): Default amount of coin agents earn from building.
            Must be >= 0. Default is 10.
        payment_max_skill_multiplier (int): Maximum skill multiplier that an agent
            can sample. Must be >= 1. Default is 1.
        skill_dist (str): Distribution type for sampling skills. Default ("none")
            gives all agents identical skill equal to a multiplier of 1. "pareto" and
            "lognormal" sample skills from the associated distributions.
        upgrade_labor (float): Labor cost associated with building a house.
            Must be >= 0. Default is 10.
    """

    name = "Upgrade"
    component_type = None
    required_entities = ["Exp", "Mat", "Token", "Labor", "Capability"]
    agent_subclasses = ["BasicPlayer"]

    def __init__(
        self,
        *base_component_args,
        payment_max_skill_multiplier=1,
        skill_dist="none",
        upgrade_income=10.0,
        upgrade_labor=1.0,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        self.payment_max_skill_multiplier = int(payment_max_skill_multiplier)
        assert self.payment_max_skill_multiplier >= 1

        self.resource_cost = {"Exp": 1, "Mat": 1, "Token": 1}

        self.upgrade_income = int(upgrade_income)
        assert self.upgrade_income >= 0

        self.upgrade_labor = float(upgrade_labor)
        assert self.upgrade_labor >= 0

        self.skill_dist = skill_dist.lower()
        assert self.skill_dist in ["none", "pareto", "lognormal"]

        self.sampled_skills = {}

        self.upgrades = []

    def agent_can_upgrade(self, agent):
        """Return True if agent can actually build in its current location."""
        # See if the agent has the resources necessary to complete the action
        for resource, cost in self.resource_cost.items():
            if agent.state["inventory"][resource] < cost:
                return False

        # Do nothing if this spot is already occupied by a landmark or resource
        # if self.world.location_resources(*agent.loc):
        #     return False
        # if self.world.location_landmarks(*agent.loc):
        #     return False
        # If we made it here, the agent can build.
        return True

    # Required methods for implementing components
    # --------------------------------------------

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        Add a single action (build) for mobile agents.
        """
        # This component adds 1 action that mobile agents can take: build a house
        if agent_cls_name == "BasicPlayer":
            return 1

        return None

    def get_additional_state_fields(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        For mobile agents, add state fields for building skill.
        """
        if agent_cls_name not in self.agent_subclasses:
            return {}
        if agent_cls_name == "BasicPlayer":
            return {"upgrade_income": float(self.upgrade_income), "upgrade_skill": float(1)}
        raise NotImplementedError

    def component_step(self):
        """
        See base_component.py for detailed description.

        Convert stone+wood to house+coin for agents that choose to build and can.
        """
        world = self.world
        build = []
        # Apply any building actions taken by the mobile agents
        for agent in world.get_random_order_agents():

            action = agent.get_component_action(self.name)

            # This component doesn't apply to this agent!
            if action is None:
                continue

            # NO-OP!
            if action == 0:
                pass

            # Build! (If you can.)
            elif action == 1:
                if self.agent_can_upgrade(agent):
                    # Remove the resources
                    for resource, cost in self.resource_cost.items():
                        agent.state["inventory"][resource] -= cost

                    # Place a house where the agent is standing
                    # loc_r, loc_c = agent.loc
                    # world.create_landmark("House", loc_r, loc_c, agent.idx)

                    # Receive payment for the house
                    agent.state["endogenous"]["Capability"] += agent.state["upgrade_income"]

                    # Incur the labor cost for building
                    agent.state["endogenous"]["Labor"] += self.upgrade_labor

                    build.append(
                        {
                            "upgrader": agent.idx,
                            "loc": np.array(agent.loc),
                            "income": float(agent.state["upgrade_income"]),
                        }
                    )

            else:
                raise ValueError

        self.upgrades.append(build)

    def generate_observations(self):
        """
        See base_component.py for detailed description.

        Here, agents observe their build skill. The planner does not observe anything
        from this component.
        """

        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[agent.idx] = {
                "upgrade_income": agent.state["upgrade_income"] * self.end_scale,
                "upgrade_skill": self.sampled_skills[agent.idx],
            }

        return obs_dict

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.

        Prevent building only if a landmark already occupies the agent's location.
        """

        masks = {}
        # Mobile agents' build action is masked if they cannot build with their
        # current location and/or endowment
        for agent in self.world.agents:
            masks[agent.idx] = np.array([self.agent_can_upgrade(agent)])

        return masks

    # For non-required customization
    # ------------------------------

    def get_metrics(self):
        """
        Metrics that capture what happened through this component.

        Returns:
            metrics (dict): A dictionary of {"metric_name": metric_value},
                where metric_value is a scalar.
        """
        world = self.world

        upgrade_stats = {a.idx: {"n_upgrades": 0} for a in world.agents}
        for upgrades in self.upgrades:
            for upgrade in upgrades:
                idx = upgrade["upgrader"]
                upgrade_stats[idx]["n_upgrades"] += 1

        out_dict = {}
        for a in world.agents:
            for k, v in upgrade_stats[a.idx].items():
                out_dict["{}/{}".format(a.idx, k)] = v

        return out_dict

    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.

        Re-sample agents' building skills.
        """
        world = self.world

        self.sampled_skills = {agent.idx: 1 for agent in world.agents}

        PMSM = self.payment_max_skill_multiplier

        for agent in world.agents:
            if self.skill_dist == "none":
                sampled_skill = 1
                pay_rate = 1
            elif self.skill_dist == "pareto":
                sampled_skill = np.random.pareto(4)
                pay_rate = np.minimum(PMSM, (PMSM - 1) * sampled_skill + 1)
            elif self.skill_dist == "lognormal":
                sampled_skill = np.random.lognormal(-1, 0.5)
                pay_rate = np.minimum(PMSM, (PMSM - 1) * sampled_skill + 1)
            else:
                raise NotImplementedError

            agent.state["upgrade_income"] = float(
                pay_rate * self.upgrade_income)
            agent.state["upgrade_skill"] = float(sampled_skill)

            self.sampled_skills[agent.idx] = sampled_skill

        self.upgrades = []

    def get_dense_log(self):
        """
        Log builds.

        Returns:
            builds (list): A list of build events. Each entry corresponds to a single
                timestep and contains a description of any builds that occurred on
                that timestep.

        """
        return self.upgrades
