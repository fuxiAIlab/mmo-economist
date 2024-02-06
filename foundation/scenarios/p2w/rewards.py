# SPDX-FileCopyrightText: 2024 by NetEase, Inc., All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np


def get_gini(endowments):
    """Returns the normalized Gini index describing the distribution of endowments.

    https://en.wikipedia.org/wiki/Gini_coefficient

    Args:
        endowments (ndarray): The array of endowments for each of the agents in the
            simulated economy.

    Returns:
        Normalized Gini index for the distribution of endowments (float). A value of 1
            indicates everything belongs to 1 agent (perfect inequality), whereas a
            value of 0 indicates all agents have equal endowments (perfect equality).

    Note:
        Uses a slightly different method depending on the number of agents. For fewer
        agents (<30), uses an exact but slow method. Switches to using a much faster
        method for more agents, where both methods produce approximately equivalent
        results.
    """
    n_agents = len(endowments)

    if n_agents < 30:  # Slower. Accurate for all n.
        diff_ij = np.abs(
            endowments.reshape((n_agents, 1)) - endowments.reshape((1, n_agents))
        )
        diff = np.sum(diff_ij)
        norm = 2 * n_agents * endowments.sum(axis=0)
        unscaled_gini = diff / (norm + 1e-10)
        gini = unscaled_gini / ((n_agents - 1) / n_agents)
        return gini

    # Much faster. Slightly overestimated for low n.
    s_endows = np.sort(endowments)
    return 1 - (2 / (n_agents + 1)) * np.sum(
        np.cumsum(s_endows) / (np.sum(s_endows) + 1e-10)
    )


def get_equality(endowments):
    """Returns the complement of the normalized Gini index (equality = 1 - Gini).

    Args:
        endowments (ndarray): The array of endowments for each of the agents in the
            simulated economy.

    Returns:
        Normalized equality index for the distribution of endowments (float). A value
            of 0 indicates everything belongs to 1 agent (perfect inequality),
            whereas a value of 1 indicates all agents have equal endowments (perfect
            equality).
    """
    return 1 - get_gini(endowments)


def get_profitability(endowments):
    """Returns the total endowments inside the simulated economy.

    Args:
        endowments (ndarray): The array of endowments for each of the
            agents in the simulated economy.

    Returns:
        Total endowment (float).
    """
    return np.sum(endowments)


def isoelastic_utility_for_player(
    income,
    monetary_cost,
    nonmonetary_cost,
    isoelastic_eta,
    labor_coefficient,
    income_exchange_rate=0.2,
    monetary_cost_sensitivity=0.5,
    nonmonetary_cost_sensitivity=0.5,
):
    assert np.all(income >= 0)
    assert 0 <= isoelastic_eta <= 1.0

    # Utility from income
    if isoelastic_eta == 1.0:  # dangerous
        util_income = np.log(np.max(1, income * income_exchange_rate))
    else:  # isoelastic_eta >= 0
        util_income = ((income * income_exchange_rate) ** (1 - isoelastic_eta) - 1) / (
            1 - isoelastic_eta
        )

    # Disutility from currency and labor
    util_cost = (
        monetary_cost_sensitivity * monetary_cost
        + nonmonetary_cost_sensitivity * labor_coefficient * nonmonetary_cost
    )

    # Net utility
    util = util_income - util_cost

    return util


def utility_for_planner(monetary_incomes, nonmonetary_incomes, equality_weight):
    n_agents = len(monetary_incomes)
    profitability = get_profitability(monetary_incomes) / n_agents
    equality = equality_weight * get_equality(nonmonetary_incomes) + (
        1 - equality_weight
    )
    return equality * profitability


def utility2_for_planner(monetary_incomes, nonmonetary_incomes, equality_weight):
    n_agents = len(monetary_incomes)
    # 人均付费 ARPU
    profitability = get_profitability(monetary_incomes) / n_agents
    # 人均公平性加权战力（人均战力*公平性）
    equality = (
        equality_weight * get_equality(nonmonetary_incomes) + (1 - equality_weight)
    ) * (np.sum(nonmonetary_incomes) / n_agents)
    return equality * profitability


def utility_normalized_for_planner(
    monetary_incomes, exp_monetary_incomes, nonmonetary_incomes, equality_weight
):
    n_agents = len(monetary_incomes)
    profitability = (
        get_profitability(monetary_incomes) / n_agents / exp_monetary_incomes
    )
    equality = equality_weight * get_equality(nonmonetary_incomes) + (
        1 - equality_weight
    )
    return equality * profitability


def utility2_normalized_for_planner(
    monetary_incomes,
    exp_monetary_incomes,
    nonmonetary_incomes,
    exp_nonmonetary_incomes,
    equality_weight,
):
    n_agents = len(monetary_incomes)
    # 人均付费 ARPU / 期望人均付费 exp ARPU
    profitability = (
        get_profitability(monetary_incomes) / n_agents / exp_monetary_incomes
    )
    # 人均公平性加权战力（人均战力*公平性）/ 期望人均战力
    equality = (
        (equality_weight * get_equality(nonmonetary_incomes) + (1 - equality_weight))
        * (np.sum(nonmonetary_incomes) / n_agents)
        / exp_nonmonetary_incomes
    )
    return equality * profitability
