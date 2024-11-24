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

#  dct_mct_analysis
#  Copyright (C) 2021  FAU - RKI
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from i2mb.utils.visualization.world_view import draw_world
from i2mb.worlds import PartyRoom
from i2mb.worlds import Scenario


class Party(Scenario):
    def __init__(self, population, num_tables=9, duration=15, **kwargs):
        super().__init__(population, **kwargs)
        self.duration = duration
        self.num_tables = num_tables

    def _build_world(self):
        dim = np.sqrt(2.45**2 * self.num_tables)
        self._world = PartyRoom(num_tables=self.num_tables, duration=self.duration,
                                dims=(dim, dim), population=self.population)

    def draw_world(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        scat = draw_world(self.world, ax, padding=0)
        artists = [scat]
        for pos in self._world.tables:
            c = Circle(pos, self._world.table_radius, fill=True, linewidth=1.2, color='k', alpha=0.4)
            ax.add_patch(c)
            artists.append(c)

        # scat2 = ax.scatter(*self._world.tables.T, color="k")
        return ax, artists

    def dynamic_regions(self):
        return [self._world]
