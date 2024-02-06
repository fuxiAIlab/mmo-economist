# SPDX-FileCopyrightText: 2024 by NetEase, Inc., All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from foundation.base.base_agent import BaseAgent, agent_registry


@agent_registry.add
class BasicPlayer(BaseAgent):
    """
    A basic (mobile) player agent represents an individual player in the MMO economic simulation.

    "Mobile" refers to agents of this type being able to move around in the 2D world.
    "Mobile" is optional in some cases.
    """

    name = "BasicPlayer"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._monetary_cost_sensitivity = 0.5
        self._nonmonetary_cost_sensitivity = 0.5

    def set_cost_sensitivity(
        self, monetary_cost_sensitivity, nonmonetary_cost_sensitivity
    ):
        """
        Set the sensitivity of the agent to monetary and non-monetary costs.
        """
        self._monetary_cost_sensitivity = monetary_cost_sensitivity
        self._nonmonetary_cost_sensitivity = nonmonetary_cost_sensitivity

    @property
    def monetary_cost_sensitivity(self):
        """
        The sensitivity of the agent to monetary costs. A value of 0.0 means the agent
        is insensitive to monetary costs, while a value of 1.0 means the agent is
        maximally sensitive to monetary costs.
        """
        return self._monetary_cost_sensitivity

    @property
    def nonmonetary_cost_sensitivity(self):
        """
        The sensitivity of the agent to non-monetary costs. A value of 0.0 means the agent
        is insensitive to non-monetary costs, while a value of 1.0 means the agent is
        maximally sensitive to non-monetary costs.
        """
        return self._nonmonetary_cost_sensitivity
