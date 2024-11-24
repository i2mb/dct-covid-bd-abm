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
import numpy as np


def distance_exposure(distances):
    return np.exp(-distances)


def exposure_function(t, region_exposed_contacts, region_contacts, distances, population_index,
                      infectiousness_level):
    dist_mat = ((region_contacts == population_index[:, None, None]).astype(float) *
                distance_exposure(distances).reshape(1, -1, 1))

    dist_mat = dist_mat.sum(axis=2)
    mat = (region_contacts == population_index[:, None, None]).astype(float).sum(axis=2)
    inf_agents = infectiousness_level[population_index].ravel()
    inf_mat = mat * inf_agents.reshape(-1, 1)
    delta = inf_mat[(inf_agents > 0), :].dot((mat * dist_mat).T).sum(axis=0) * (inf_agents == 0)
    return delta
