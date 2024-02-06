# SPDX-FileCopyrightText: 2024 by NetEase, Inc., All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy

import numpy as np

from foundation.base.base_component import BaseComponent, component_registry


@component_registry.add
class Adjust(BaseComponent):
    """
    Adjust refers to the ability to adjust the rate of different resources.

    Args:
        is_biadjust (bool): Whether to allow reductions in addition to increases.
        adjust_period (int): The number of steps between adjustments.
        adjust_rate_max (float): The maximum adjustment rate.
        adjust_rate_min (float): The minimum adjustment rate.
        adjust_rate_bin (float): The size of the adjustment rate bins.
    """

    name = "Adjust"
    component_type = "P2W"
    required_entities = ["EXP", "MAT", "TOK"]
    agent_subclasses = ["BasicPlayer"]

    def __init__(
        self,
        *base_component_args,
        is_biadjust=True,
        adjust_period=500,
        adjust_rate_max=0.2,
        adjust_rate_min=0.0,
        adjust_rate_bin=0.1,
        **base_component_kwargs,
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        assert adjust_rate_max >= 0 and adjust_rate_min >= 0 and adjust_rate_bin > 0
        assert adjust_rate_max >= adjust_rate_min
        assert adjust_rate_bin <= adjust_rate_max - adjust_rate_min

        self.adjust_period = int(adjust_period)
        assert self.adjust_period > 0

        self.adjust_cycle_pos = 1

        self._adjust_rates = [
            round(adjust_rate_min + adjust_rate_bin * x, 2)
            for x in range(
                int(
                    (adjust_rate_max * 10 - adjust_rate_min * 10)
                    / (adjust_rate_bin * 10)
                )
                + 1
            )
        ]

        if is_biadjust:
            self._adjust_rates = self._adjust_rates + [-x for x in self._adjust_rates]

        self._planner_masks = None
        self._adjust_rates = sorted(set(self._adjust_rates))
        self._base_adjust = {k: 1.0 for k in self.required_entities}
        self._curr_adjust = deepcopy(self._base_adjust)
        self._last_adjust = deepcopy(self._base_adjust)
        self.adjusts = []

    @property
    def adjust_rates(self):
        return self._adjust_rates

    @property
    def last_adjust(self):
        return self._last_adjust

    @property
    def curr_adjust(self):
        return self._curr_adjust

    @property
    def base_adjust(self):
        return self._base_adjust

    def start_new_adjust(self):
        self.adjust_cycle_pos = 1

    def get_n_actions(self, agent_cls_name):
        """This component is passive: it does not add any actions."""
        if agent_cls_name == "BasicPlanner":
            return [
                (resource, len(self._adjust_rates))
                for resource in self.required_entities
            ]
        return 0

    def get_additional_state_fields(self, agent_cls_name):
        """This component does not add any agent state fields."""
        return {}

    def component_step(self):
        if self.adjust_cycle_pos == 1:
            self._last_adjust = deepcopy(self._curr_adjust)
            for resource in self.required_entities:
                planner_action = self.world.planner.get_component_action(
                    self.name, resource
                )

                if planner_action == 0:
                    self._curr_adjust[resource] = self._base_adjust[resource]
                elif planner_action <= len(self._adjust_rates):
                    self._curr_adjust[resource] = (
                        self._base_adjust[resource]
                        + self._adjust_rates[int(planner_action) - 1]
                    )
                else:
                    raise ValueError

        self.adjust_cycle_pos += 1

    def generate_observations(self):
        is_first_day = float(self.adjust_cycle_pos == 1)
        adjust_phase = self.adjust_cycle_pos / self.adjust_period

        obs = dict()

        obs[self.world.planner.idx] = dict(
            is_first_day=is_first_day, adjust_phase=adjust_phase
        )

        obs[self.world.planner.idx].update(
            {
                "last_adjust_" + str(resource): self._last_adjust[resource]
                for resource in self.required_entities
            }
        )

        obs[self.world.planner.idx].update(
            {
                "curr_adjust_" + str(resource): self._curr_adjust[resource]
                for resource in self.required_entities
            }
        )

        for agent in self.world.agents:
            obs[str(agent.idx)] = dict(
                is_first_day=is_first_day, adjust_phase=adjust_phase
            )

        return obs

    def generate_masks(self, completions=0):
        if self._planner_masks is None:
            masks = super().generate_masks(completions=completions)
            self._planner_masks = dict(
                adjust=deepcopy(masks[self.world.planner.idx]),
                zero={
                    k: np.zeros_like(v)
                    for k, v in masks[self.world.planner.idx].items()
                },
            )

        masks = dict()
        if self.adjust_cycle_pos != 1:
            masks[self.world.planner.idx] = self._planner_masks["zero"]
        else:
            masks[self.world.planner.idx] = self._planner_masks["adjust"]

        return masks

    def additional_reset_steps(self):
        self.adjust_cycle_pos = 1
        self.adjusts = []

    def get_dense_log(self):
        """
        Log adjusts.

        Returns:
            adjusts (list): A list of adjust events. Each entry corresponds to a single
                timestep and contains a description of any adjusts that occurred on
                that timestep.

        """
        return self.adjusts
