# SPDX-FileCopyrightText: 2024 by NetEase, Inc., All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from foundation.base.base_component import BaseComponent, component_registry


@component_registry.add
class Recharge(BaseComponent):
    """
    Recharge refers to any forex behavior where players make payments in the MMOs,
    such as In-App Purchase (IAP).

    Args:
        recharge_income (int): Default amount of token players earn from recharging.
            Must be >= 0. Default is 10.
        recharge_labor (float): Labor cost associated with recharging.
            Must be >= 0. Default is 1.
    """

    name = "Recharge"
    component_type = "Forex"
    required_entities = ["Exp", "Mat", "Token", "Labor", "Capability"]
    agent_subclasses = ["BasicPlayer"]

    def __init__(
        self,
        *base_component_args,
        recharge_income=10.0,
        recharge_labor=1.0,
        **base_component_kwargs,
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        # The resources required to recharge, which can also represent the exchange rate for recharging.
        self.resource_cost = {"Currency": 1}

        self.recharge_income = int(recharge_income)
        assert self.recharge_income >= 0

        self.recharge_labor = float(recharge_labor)
        assert self.recharge_labor >= 0

        self.recharges = []

    def agent_can_recharge(self, agent):
        """Return True if player can actually recharge in its current location."""
        # See if the player has the resources necessary to complete the action
        for resource, cost in self.resource_cost.items():
            if agent.state["inventory"][resource] < cost:
                return False

        # If we made it here, the player can recharge.
        return True

    # Required methods for implementing components
    # --------------------------------------------

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        Add a single action (recharge) for players.
        """
        # This component adds 1 action that player can take
        if agent_cls_name == "BasicPlayer":
            return 1

        return None

    def get_additional_state_fields(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        For players, add state fields for recharging.
        """
        if agent_cls_name not in self.agent_subclasses:
            return {}
        if agent_cls_name == "BasicPlayer":
            return {"recharge_income": float(self.recharge_income)}
        raise NotImplementedError

    def component_step(self):
        """
        See base_component.py for detailed description.

        Convert CNY to TOK for players that choose to recharge and can.
        """
        world = self.world
        recharge = []
        # Apply any recharge actions taken by the players.
        for agent in world.get_random_order_agents():
            action = agent.get_component_action(self.name)

            # This component doesn't apply to this player!
            if action is None:
                continue

            # NO-OP!
            if action == 0:
                pass

            # Recharge! (If you can.)
            elif action == 1:
                if self.agent_can_recharge(agent):
                    # Remove the resources
                    for resource, cost in self.resource_cost.items():
                        agent.state["inventory"][resource] -= cost

                    # Receive payment for recharging
                    agent.state["inventory"]["Token"] += agent.state["recharge_income"]

                    # Incur the labor cost for recharging
                    agent.state["endogenous"]["Labor"] += self.recharge_labor

                    recharge.append(
                        {
                            "recharger": agent.idx,
                            "loc": np.array(agent.loc),
                            "income": float(agent.state["recharge_income"]),
                        }
                    )

            else:
                raise ValueError

        self.recharges.append(recharge)

    def generate_observations(self):
        """
        See base_component.py for detailed description.
        """

        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[agent.idx] = {
                "recharge_income": agent.state["recharge_income"] * self.inv_scale
            }

        return obs_dict

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.
        """

        masks = {}
        # Player' recharge action is masked if they cannot recharge with their
        # current location and/or endowment
        for agent in self.world.agents:
            masks[agent.idx] = np.array([self.agent_can_recharge(agent)])

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

        recharge_stats = {a.idx: {"n_recharges": 0} for a in world.agents}
        for recharges in self.recharges:
            for recharge in recharges:
                idx = recharge["recharger"]
                recharge_stats[idx]["n_recharges"] += 1

        out_dict = {}
        for a in world.agents:
            for k, v in recharge_stats[a.idx].items():
                out_dict["{}/{}".format(a.idx, k)] = v

        return out_dict

    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.
        """
        for agent in self.world.agents:
            agent.state["recharge_income"] = float(self.recharge_income)

        self.recharges = []

    def get_dense_log(self):
        """
        Log recharges.

        Returns:
            recharges (list): A list of recharge events. Each entry corresponds to a single
                timestep and contains a description of any recharges that occurred on
                that timestep.

        """
        return self.recharges
