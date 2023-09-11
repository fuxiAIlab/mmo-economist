# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np
from foundation.base.base_component import BaseComponent, component_registry


@component_registry.add
class Shop(BaseComponent):
    name = "Shop"
    component_type = "Trade"
    required_entities = ["Exp", "Mat", "Token", "Labor"]
    agent_subclasses = ["BasicPlayer"]

    def __init__(self,
                 *base_component_args,
                 shop_labor=1.0,
                 **base_component_kwargs):
        super().__init__(*base_component_args, **base_component_kwargs)

        self.commodities = {"Exp": {"Token": 10}, "Mat": {"Token": 10}}
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
        """Return True if agent can actually shop in its current location."""
        # See if the agent has the resources necessary to complete the action
        for resource, cost in self.commodities[commodity].items():
            if agent.state["inventory"][resource] < cost:
                return False

        # If we made it here, the agent can recharge.
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
                state.update(
                    {"shop_" + str(commodity) + "_income": float(1.0)})
                state.update({
                    "shop_" + str(commodity) + "_cost_" + str(resource):
                    float(cost)
                    for resource, cost in self.commodities[commodity].items()
                })
            return state
        raise NotImplementedError

    def component_step(self):
        """
        See base_component.py for detailed description.
        """
        world = self.world
        shop = []
        # Apply any building actions taken by the mobile agents
        for agent in world.get_random_order_agents():

            action = agent.get_component_action(self.name)

            # This component doesn't apply to this agent!
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

                    # Receive payment for the shop
                    agent.state["inventory"][commodity] += 1

                    # Incur the labor cost for shopping
                    agent.state["endogenous"]["Labor"] += self.shop_labor

                    shop.append({
                        "shoper": agent.idx,
                        "loc": np.array(agent.loc),
                        "commodity": commodity,
                        "income": float(1),
                        'labor': self.shop_labor,
                        'cost': self.commodities[commodity]
                    })
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
                obs_dict[agent.idx].update({
                    "shop_" + str(commodity) + "_income":
                    float(1.0) * self.inv_scale
                })

                obs_dict[agent.idx].update({
                    "shop_" + str(commodity) + "_cost_" + str(resource):
                    float(cost) * self.inv_scale
                    for resource, cost in self.commodities[commodity].items()
                })
        return obs_dict

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.
        """
        masks = {}
        # Mobile agents' shop action is masked if they cannot shop with their
        # current location and/or endowment
        for agent in self.world.agents:
            masks[agent.idx] = np.array([
                self.agent_can_shop(agent, commodity)
                for commodity in self.commodities.keys()
            ])

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
                agent.state.update({
                    "shop_" + str(commodity) + "_cost_" + str(resource):
                    float(cost)
                    for resource, cost in self.commodities[commodity].items()
                })

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
