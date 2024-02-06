# SPDX-FileCopyrightText: 2024 by NetEase, Inc., All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from foundation.base.base_component import BaseComponent, component_registry


@component_registry.add
class Upgrade(BaseComponent):
    """
    Upgrade refers to any consuming behavior of players to improve their capabilities
    by consuming corresponding economic resources in the MMOs.

    Args:
        payment_max_skill_multiplier (int): Maximum skill multiplier that a player
            can sample. Must be >= 1. Default is 1.
        upgrade_income (int): Default amount of capability players earn from upgrading.
            Must be >= 0. Default is 10.
        upgrade_labor (float): Labor cost associated with upgrading.
            Must be >= 0. Default is 1.
        skill_dist (str): Distribution type for sampling skills. Default ("none")
            gives all players identical skill equal to a multiplier of 1. "pareto" and
            "lognormal" sample skills from the associated distributions.
    """

    name = "Upgrade"
    component_type = "Consumption"
    required_entities = ["EXP", "MAT", "TOK", "LAB", "CAP"]
    agent_subclasses = ["BasicPlayer"]

    def __init__(
        self,
        *base_component_args,
        payment_max_skill_multiplier=1,
        skill_dist="none",
        upgrade_income=10.0,
        upgrade_labor=1.0,
        **base_component_kwargs,
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        self.payment_max_skill_multiplier = int(payment_max_skill_multiplier)
        assert self.payment_max_skill_multiplier >= 1

        # The resources required to upgrade
        self.resource_cost = {"EXP": 1, "MAT": 1, "TOK": 1}

        self.upgrade_income = int(upgrade_income)
        assert self.upgrade_income >= 0

        self.upgrade_labor = float(upgrade_labor)
        assert self.upgrade_labor >= 0

        self.skill_dist = skill_dist.lower()
        assert self.skill_dist in ["none", "pareto", "lognormal"]

        self.sampled_skills = {}

        self.upgrades = []

    def agent_can_upgrade(self, agent):
        """Return True if player can actually upgrade in its current location."""
        # See if the player has the resources necessary to complete the action
        for resource, cost in self.resource_cost.items():
            if agent.state["inventory"][resource] < cost:
                return False

        # If we made it here, the player can upgrade.
        return True

    # Required methods for implementing components
    # --------------------------------------------

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        Add a single action (upgrade) for players.
        """
        # This component adds 1 action that players can take
        if agent_cls_name == "BasicPlayer":
            return 1

        return None

    def get_additional_state_fields(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        For players, add state fields for upgrading.
        """
        if agent_cls_name not in self.agent_subclasses:
            return {}
        if agent_cls_name == "BasicPlayer":
            return {
                "upgrade_income": float(self.upgrade_income),
                "upgrade_skill": float(1),
            }
        raise NotImplementedError

    def component_step(self):
        """
        See base_component.py for detailed description.

        Convert resources to capabilities for players that choose to upgrade and can.
        """
        world = self.world
        upgrad = []
        # Apply any upgrad actions taken by the players
        for agent in world.get_random_order_agents():
            action = agent.get_component_action(self.name)

            # This component doesn't apply to this player!
            if action is None:
                continue

            # NO-OP!
            if action == 0:
                pass

            # Upgrade! (If you can.)
            elif action == 1:
                if self.agent_can_upgrade(agent):
                    # Remove the resources
                    for resource, cost in self.resource_cost.items():
                        agent.state["inventory"][resource] -= cost

                    # Receive capabilities for upgrading
                    agent.state["endogenous"]["CAP"] += agent.state["upgrade_income"]

                    # Incur the labor cost for upgrading
                    agent.state["endogenous"]["LAB"] += self.upgrade_labor

                    upgrad.append(
                        {
                            "upgrader": agent.idx,
                            "loc": np.array(agent.loc),
                            "income": float(agent.state["upgrade_income"]),
                        }
                    )

            else:
                raise ValueError

        self.upgrades.append(upgrad)

    def generate_observations(self):
        """
        See base_component.py for detailed description.

        Here, players observe their upgrade skill. The planner does not observe anything
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
        """

        masks = {}
        # Players' upgrade action is masked if they cannot upgrade with their
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

        Re-sample players' upgrading skills.
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

            agent.state["upgrade_income"] = float(pay_rate * self.upgrade_income)
            agent.state["upgrade_skill"] = float(sampled_skill)

            self.sampled_skills[agent.idx] = sampled_skill

        self.upgrades = []

    def get_dense_log(self):
        """
        Log upgrades.

        Returns:
            upgrades (list): A list of upgrade events. Each entry corresponds to a single
                timestep and contains a description of any upgrades that occurred on
                that timestep.

        """
        return self.upgrades
