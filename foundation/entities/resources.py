# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from foundation.base.registrar import Registry


class Resource:
    """Base class for Resource entity classes.

    Resource classes describe entities that can be a part of an agent's inventory.

    Resources can also be a part of the world as collectible entities: for each
    Resource class with Resource.collectible=True, a complementary
    ResourceSourceBlock Landmark class will be created in landmarks.py. For each
    collectible resource in the environment, the world map will include a resource
    source block channel (representing landmarks where collectible resources are
    generated) and a resource channel (representing locations where collectible
    resources have generated).
    """

    name = None
    color = None  # array of RGB values [0 - 1]
    collectible = None  # Is this something that exists in the world?
    tradable = None  # Is this something that exists in the market?
    # (versus something that can only be owned)

    def __init__(self):
        assert self.name is not None
        assert self.color is not None
        assert self.collectible is not None
        assert self.tradable is not None


resource_registry = Registry(Resource)


@resource_registry.add
class Exp(Resource):
    """Exp resource. green. collectible. untradable."""
    name = "Exp"
    color = np.array([0, 255, 0]) / 255.0
    collectible = True
    tradable = False


@resource_registry.add
class Mat(Resource):
    """Mat resource. red. collectible. tradable"""
    name = "Mat"
    color = np.array([255, 0, 0]) / 255.0
    collectible = True
    tradable = True


@resource_registry.add
class Token(Resource):
    """Token resource. yellow. collectible. untradable"""
    name = "Token"
    color = np.array([255, 255, 0]) / 255.0
    collectible = True
    tradable = False


@resource_registry.add
class Currency(Resource):
    """Exp resource. blue. uncollectible. untradable"""
    name = "Currency"
    color = np.array([0, 0, 255]) / 255.0
    collectible = False
    tradable = False
