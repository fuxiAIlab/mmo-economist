# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from copy import deepcopy

import numpy as np

from foundation.base.base_component import (
    BaseComponent,
    component_registry,
)


@component_registry.add
class LaunchReadjustment(BaseComponent):
    name = "LaunchReadjustment"
    component_type = None
    required_entities = ["Exp", "Mat", "Token"]
    agent_subclasses = ["BasicPlayer"]

    def __init__(self,
                 *base_component_args,
                 is_biadjustment=True,
                 adjustment_period=500,
                 adjustment_rate_max=0.2,
                 adjustment_rate_min=0.0,
                 adjustment_rate_bin=0.1,
                 **base_component_kwargs):
        super().__init__(*base_component_args, **base_component_kwargs)

        assert adjustment_rate_max >= 0 and adjustment_rate_min >= 0 and adjustment_rate_bin > 0
        assert adjustment_rate_max >= adjustment_rate_min
        assert adjustment_rate_bin <= adjustment_rate_max - adjustment_rate_min

        # TODO: 这个参数可能无用了
        self.adjustment_period = int(adjustment_period)
        assert self.adjustment_period > 0

        self.adjustment_cycle_pos = 1

        self._adjustment_rates = [
            round(adjustment_rate_min + adjustment_rate_bin * x, 2)
            for x in range(
                int((adjustment_rate_max * 10 - adjustment_rate_min * 10) /
                    (adjustment_rate_bin * 10)) + 1)
        ]

        if is_biadjustment:
            self._adjustment_rates = self._adjustment_rates + \
                [-x for x in self._adjustment_rates]

        self._planner_masks = None
        self._adjustment_rates = sorted(set(self._adjustment_rates))
        self._base_launch_adjustment = {k: 1.0 for k in self.required_entities}
        self._curr_launch_adjustment = deepcopy(self._base_launch_adjustment)
        self._last_launch_adjustment = deepcopy(self._base_launch_adjustment)
        self.adjustments = []

    @property
    def adjustment_rates(self):
        return self._adjustment_rates

    @property
    def launch_adjustment(self):
        return self._curr_launch_adjustment

    @property
    def base_launch_adjustment(self):
        return self._base_launch_adjustment

    def start_new_launch_adjustment(self):
        self.adjustment_cycle_pos = 1

    def get_n_actions(self, agent_cls_name):
        """This component is passive: it does not add any actions."""
        if agent_cls_name == "BasicPlanner":
            return [(resource, len(self._adjustment_rates))
                    for resource in self.required_entities]
        return 0

    def get_additional_state_fields(self, agent_cls_name):
        """This component does not add any agent state fields."""
        return {}

    def component_step(self):
        if self.adjustment_cycle_pos == 1:
            self._last_launch_adjustment = deepcopy(
                self._curr_launch_adjustment)
            for resource in self.required_entities:
                planner_action = self.world.planner.get_component_action(
                    self.name, resource)

                if planner_action == 0:
                    self._curr_launch_adjustment[
                        resource] = self._base_launch_adjustment[resource]
                elif planner_action <= len(self._adjustment_rates):
                    self._curr_launch_adjustment[resource] = self._base_launch_adjustment[resource] + \
                        self._adjustment_rates[int(planner_action)-1]
                else:
                    raise ValueError

        self.adjustment_cycle_pos += 1

    def generate_observations(self):
        # is_adjustment_day = float(
        # self.adjustment_cycle_pos >= self.adjustment_period)
        is_first_day = float(self.adjustment_cycle_pos == 1)
        adjustment_phase = self.adjustment_cycle_pos / self.adjustment_period

        obs = dict()

        obs[self.world.planner.idx] = dict(
            # is_adjustment_day=is_adjustment_day,
            is_first_day=is_first_day,
            adjustment_phase=adjustment_phase)

        obs[self.world.planner.idx].update({
            'last_adjustment_' + str(resource):
            self._last_launch_adjustment[resource]
            for resource in self.required_entities
        })

        obs[self.world.planner.idx].update({
            'curr_adjustment_' + str(resource):
            self._curr_launch_adjustment[resource]
            for resource in self.required_entities
        })

        for agent in self.world.agents:

            obs[str(agent.idx)] = dict(
                # is_adjustment_day=is_adjustment_day,
                is_first_day=is_first_day,
                adjustment_phase=adjustment_phase)

            # TODO: planner能观察到player的信息？
            # obs["p" + str(agent.idx)] = dict(

            #     last_income=self._last_income_obs[i],
            #     last_marginal_rate=self.last_marginal_rate[i],
            #     curr_marginal_rate=curr_marginal_rate,
            # )

        return obs

    def generate_masks(self, completions=0):
        if self._planner_masks is None:
            masks = super().generate_masks(completions=completions)
            self._planner_masks = dict(
                adjustment=deepcopy(masks[self.world.planner.idx]),
                zero={
                    k: np.zeros_like(v)
                    for k, v in masks[self.world.planner.idx].items()
                })

        masks = dict()
        if self.adjustment_cycle_pos != 1:
            masks[self.world.planner.idx] = self._planner_masks["zero"]
        else:
            masks[self.world.planner.idx] = self._planner_masks["adjustment"]

        return masks

    def additional_reset_steps(self):
        self.adjustment_cycle_pos = 1
        self.adjustments = []

    def get_dense_log(self):
        return self.adjustments
