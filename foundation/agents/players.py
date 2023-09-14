# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from foundation.base.base_agent import BaseAgent, agent_registry


@agent_registry.add
class BasicPlayer(BaseAgent):
    """
    A basic mobile agent represents an individual actor in the economic simulation.

    "Mobile" refers to agents of this type being able to move around in the 2D world.
    """

    name = "BasicPlayer"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._monetary_cost_sensitivity = 0.5
        self._nonmonetary_cost_sensitivity = 0.5
    
    def set_cost_sensitivity(self, monetary_cost_sensitivity, nonmonetary_cost_sensitivity):
        self._monetary_cost_sensitivity = monetary_cost_sensitivity
        self._nonmonetary_cost_sensitivity = nonmonetary_cost_sensitivity

    @property
    def monetary_cost_sensitivity(self):
        return self._monetary_cost_sensitivity
    
    @property
    def nonmonetary_cost_sensitivity(self):
        return self._nonmonetary_cost_sensitivity