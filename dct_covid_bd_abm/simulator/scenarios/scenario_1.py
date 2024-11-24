

#  dct_mct_analysis
#  Copyright (c) 2021 FAU - RKI
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

from i2mb.engine.relocator import Relocator
from i2mb.utils.visualization.world_view import draw_world
from i2mb.worlds import CompositeWorld
from i2mb.worlds import Home
from i2mb.worlds import Scenario
from i2mb.worlds.graveyard import Graveyard
from i2mb.worlds.office import Office
from i2mb.worlds.world_base import BlankSpace


class HomesAndOffices(Scenario):
    def __init__(self, population, homes=10, offices=3, home_percentages=None, **kwargs):
        super().__init__(population, **kwargs)

        self.social_network = {}
        if home_percentages is None:
            home_percentages = [1, .9, .5, .2]

        if home_percentages[0] != 1:
            home_percentages = [1] + home_percentages

        self.homes = [Home(dims=(6, 8), always_on=False, gain=10) for h in range(homes)]
        self.home_percentages = home_percentages
        self.offices = [CompositeWorld((10, 10)) for o in range(offices)]
        self.graveyard = Graveyard(dims=(20, 15), lots=len(population))
        self._build_world()
        self.relocator = Relocator(self.population, self.world)

    def _build_world(self):
        w, h = self.arrange_grid([[self.offices],
                                  [BlankSpace((20, 10)) for o in range(len(self.offices))],
                                  [[self.graveyard, BlankSpace((20, 10))], self.homes, [BlankSpace((20, 10))]]], 10)
        self._world = CompositeWorld(population=self.population, regions=self.offices + self.homes + [self.graveyard])
        self._world.dims[0] += self.right
        self._world.dims[1] += self.top

    def draw_world(self, ax=None, padding=5):
        if ax is None:
            fig, ax = plt.subplots()

        scat = draw_world(self.world, ax, padding=padding)
        ax.text(0.3, 0.65, 'Offices',
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                fontsize=12)

        ax.text(0.6, 0.55, 'Homes',
                verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes,
                fontsize=12)

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ww = (x1 - x0) - (2 * padding)
        wh = (y1 - y0) - (2 * padding)
        lx = (padding + 0.01 * ww) / (x1-x0)
        ly = (padding + 0.31 * wh) / (y1 - y0)
        ax.text(lx, ly, 'Deceased',
                verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes,
                fontsize=10)
        return ax, scat

    def dynamic_regions(self):
        return self.homes

    def assign_homes(self):
        n = len(self.population)
        if not hasattr(self.population, "home"):
            self.population.add_property("home", np.empty((n,), dtype=object))

        max_slice = len(self.homes)
        start = 0
        for slice_ in self.home_percentages:
            end = int(max_slice * slice_)
            if start + end > n:
                end = n - start

            slice_ = slice(start, start + end, None)
            self.population.home[slice_] = self.homes[:end]
            start += end
            if start >= n:
                break

        if start < n:
            self.population.home[start:] = np.random.choice(self.homes[::-1], n-start)

        self._world.containment_region[:] = self.population.home

    def assign_offices(self):
        n = len(self.population)
        if not hasattr(self.population, "office"):
            self.population.add_property("office", np.empty((n,), dtype=object))

        self.population.office[:] = np.random.choice(self.offices, (n,))

    def create_social_network(self):
        for office in self.offices:
            idx = self.population.home == office
            self.social_network.update({i: Office for i in combinations(self.population.index[idx], 2)})

        # This overrides existing networks giving preference for the home network.
        for house in self.homes:
            idx = self.population.home == house
            self.social_network.update({i: Home for i in combinations(self.population.index[idx], 2)})

    def move_home(self):
        for home_id, home in enumerate(self.homes):
            self.relocator.move_agents((self.population.home == home), home)
