# SPDX-FileCopyrightText: 2024 by NetEase, Inc., All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from foundation.base.base_component import BaseComponent, component_registry


@component_registry.add
class Shop(BaseComponent):
    """
    Shop refers to any trade behavior where players directly purchase commodities
    from game malls or non-player characters (NPCs) in the MMOs.

    Args:
        shop_labor (float): Labor cost associated with shopping. Must be >= 0.
            Default is 1.0.
    """

    name = "Shop"
    component_type = "Trade"
    required_entities = ["EXP", "MAT", "TOK", "LAB"]
    agent_subclasses = ["BasicPlayer"]

    def __init__(self, *base_component_args, shop_labor=1.0, **base_component_kwargs):
        super().__init__(*base_component_args, **base_component_kwargs)

        # The commodities that can be shopped for, and their fixed prices in terms of tokens.
        self.commodities = {"EXP": {"TOK": 10}, "MAT": {"TOK": 10}}
        maxes = dict()
        for _, x in self.commodities.items():
            for k, v in x.items():
                if k in maxes.keys():
                    maxes[k].append(v)
                else:
                    maxes[k] = [v]
        self.max_prices = {k: max(v) for k, v in maxes.items()}
        self.shop_labor = float(shop_labor)
        assert self.shop_labor >= 0

        self.shops = []

    def agent_can_shop(self, agent, commodity):
        """Return True if player can actually shop in its current location."""
        # See if the player has the resources necessary to complete the shop action
        for resource, cost in self.commodities[commodity].items():
            if agent.state["inventory"][resource] < cost:
                return False

        # If we made it here, the player can shop.
        return True

    # Required methods for implementing components
    # --------------------------------------------

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.
        """
        if agent_cls_name == "BasicPlayer":
            return len(self.commodities)

        return None

    def get_additional_state_fields(self, agent_cls_name):
        """
        See base_component.py for detailed description.
        """
        if agent_cls_name not in self.agent_subclasses:
            return {}
        if agent_cls_name == "BasicPlayer":
            state = {}
            for commodity in self.commodities.keys():
                state.update({"shop_" + str(commodity) + "_income": float(1.0)})
                state.update(
                    {
                        "shop_" + str(commodity) + "_cost_" + str(resource): float(cost)
                        for resource, cost in self.commodities[commodity].items()
                    }
                )
            return state
        raise NotImplementedError

    def component_step(self):
        """
        See base_component.py for detailed description.
        """
        world = self.world
        shop = []
        # Apply any shop actions taken by the player
        for agent in world.get_random_order_agents():
            action = agent.get_component_action(self.name)

            # This component doesn't apply to this player!
            if action is None:
                continue

            # NO-OP!
            if action == 0:
                pass

            # Shop! (If you can.)
            elif action <= len(self.commodities):
                commodity = list(self.commodities.keys())[int(action - 1)]
                if self.agent_can_shop(agent, commodity):
                    # Remove the resources
                    for resource, cost in self.commodities[commodity].items():
                        agent.state["inventory"][resource] -= cost

                    # Receive commodities from the shop
                    agent.state["inventory"][commodity] += 1

                    # Incur the labor cost for shopping
                    agent.state["endogenous"]["LAB"] += self.shop_labor

                    # Log the shop
                    shop.append(
                        {
                            "shoper": agent.idx,
                            "loc": np.array(agent.loc),
                            "commodity": commodity,
                            "income": float(1),
                            "labor": self.shop_labor,
                            "cost": self.commodities[commodity],
                        }
                    )
            else:
                raise ValueError

        self.shops.append(shop)

    def generate_observations(self):
        """
        See base_component.py for detailed description.
        """
        obs_dict = dict()

        for agent in self.world.agents:
            obs_dict[agent.idx] = dict()
            for commodity in self.commodities.keys():
                obs_dict[agent.idx].update(
                    {"shop_" + str(commodity) + "_income": float(1.0) * self.inv_scale}
                )

                obs_dict[agent.idx].update(
                    {
                        "shop_" + str(commodity) + "_cost_" + str(resource): float(cost)
                        * self.inv_scale
                        for resource, cost in self.commodities[commodity].items()
                    }
                )
        return obs_dict

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.
        """
        masks = {}
        # Players' shop action is masked if they cannot shop with their
        # current location and/or endowment
        for agent in self.world.agents:
            masks[agent.idx] = np.array(
                [
                    self.agent_can_shop(agent, commodity)
                    for commodity in self.commodities.keys()
                ]
            )

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

        shop_stats = {a.idx: {"n_shops": 0} for a in world.agents}
        for shops in self.shops:
            for shop in shops:
                idx = shop["shoper"]
                shop_stats[idx]["n_shops"] += 1

        out_dict = {}
        for a in world.agents:
            for k, v in shop_stats[a.idx].items():
                out_dict["{}/{}".format(a.idx, k)] = v

        return out_dict

    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.

        """
        for agent in self.world.agents:
            for commodity in self.commodities.keys():
                agent.state["shop_" + str(commodity) + "_income"] = float(1)
                agent.state.update(
                    {
                        "shop_" + str(commodity) + "_cost_" + str(resource): float(cost)
                        for resource, cost in self.commodities[commodity].items()
                    }
                )

        self.shops = []

    def get_dense_log(self):
        """
        Log shops.

        Returns:
            shops (list): A list of shop events. Each entry corresponds to a single
                timestep and contains a description of any shops that occurred on
                that timestep.
        """
        return self.shops
