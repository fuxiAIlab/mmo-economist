# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from foundation.scenarios.utils import social_metrics


def isoelastic_utility_for_player(income,
                                  monetary_cost,
                                  nonmonetary_cost,
                                  isoelastic_eta,
                                  labor_coefficient,
                                  income_exchange_rate=0.2,
                                  monetary_cost_sensitivity=0.5,
                                  nonmonetary_cost_sensitivity=0.5):
    assert np.all(income >= 0)
    assert 0 <= isoelastic_eta <= 1.0

    # Utility from income
    if isoelastic_eta == 1.0:  # dangerous
        util_income = np.log(np.max(1, income * income_exchange_rate))
    else:  # isoelastic_eta >= 0
        util_income = ((income*income_exchange_rate) ** (1 - isoelastic_eta) - 1) / \
            (1 - isoelastic_eta)

    # Disutility from currency and labor
    util_cost = monetary_cost_sensitivity * monetary_cost + \
        nonmonetary_cost_sensitivity * labor_coefficient * nonmonetary_cost

    # Net utility
    util = util_income - util_cost

    return util


def utility_for_planner(monetary_incomes, nonmonetary_incomes,
                        equality_weight):
    n_agents = len(monetary_incomes)
    profitability = social_metrics.get_profitability(
        monetary_incomes) / n_agents
    equality = equality_weight * social_metrics.get_equality(
        nonmonetary_incomes) + (1 - equality_weight)
    return equality * profitability


def utility2_for_planner(monetary_incomes, nonmonetary_incomes,
                         equality_weight):
    n_agents = len(monetary_incomes)
    # 人均付费 ARPU
    profitability = social_metrics.get_profitability(
        monetary_incomes) / n_agents
    # 人均公平性加权战力（人均战力*公平性）
    equality = (
        equality_weight * social_metrics.get_equality(nonmonetary_incomes) +
        (1 - equality_weight)) * (np.sum(nonmonetary_incomes) / n_agents)
    return equality * profitability


def utility_normalized_for_planner(monetary_incomes, exp_monetary_incomes,
                                   nonmonetary_incomes, equality_weight):
    n_agents = len(monetary_incomes)
    profitability = social_metrics.get_profitability(
        monetary_incomes) / n_agents / exp_monetary_incomes
    equality = equality_weight * social_metrics.get_equality(
        nonmonetary_incomes) + (1 - equality_weight)
    return equality * profitability


def utility2_normalized_for_planner(monetary_incomes, exp_monetary_incomes,
                                    nonmonetary_incomes,
                                    exp_nonmonetary_incomes, equality_weight):
    n_agents = len(monetary_incomes)
    # 人均付费 ARPU / 期望人均付费 exp ARPU
    profitability = social_metrics.get_profitability(
        monetary_incomes) / n_agents / exp_monetary_incomes
    # 人均公平性加权战力（人均战力*公平性）/ 期望人均战力
    equality = (
        equality_weight * social_metrics.get_equality(nonmonetary_incomes) +
        (1 - equality_weight)) * (np.sum(nonmonetary_incomes) /
                                  n_agents) / exp_nonmonetary_incomes
    return equality * profitability
