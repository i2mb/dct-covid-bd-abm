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
import networkx as nx
import numpy as np
from itertools import combinations

from i2mb.engine.relocator import Relocator
from i2mb.worlds import CompositeWorld
from i2mb.worlds import Hospital
from i2mb.worlds import Scenario, Restaurant, Bar, BusMBCitaroK, Apartment
from i2mb.worlds._composite_worlds.car import Car
from i2mb.worlds.graveyard import Graveyard
from i2mb.worlds.office import Office
from i2mb.worlds.world_base import BlankSpace, ravel_grids


class Bus(BusMBCitaroK):
    """Dummy class to have nicer printout names"""
    pass


class ComplexWorld(Scenario):
    def __init__(self, population, homes=10, offices=3, home_percentages=None, buses=3, **kwargs):
        from i2mb.worlds import Home
        super().__init__(population, **kwargs)

        self.social_network = {}
        if home_percentages is None:
            # https://www.statista.com/statistics/464187/households-by-size-germany/ 2019
            home_percentages = [0.42298889, 0.33201629, 0.11930518, 0.09114125, 0.03454839]

        if buses <= 3:
            buses = 3
        bar_config = kwargs.get("bar_config", {})
        restaurant_config = kwargs.get("restaurant_config", {})
        print(f"Using {buses} buses, {homes} homes, {offices} offices.")
        self.homes = [Home(dims=(6, 8), always_on=False, gain=10) for h in range(homes)]
        self.home_percentages = home_percentages
        self.offices = [Office(dims=(10, 10)) for o in range(offices)]
        self.buses = [Bus(orientation="horizontal") for i in range(buses)]
        self.hospital = Hospital(dims=(25, 10), beds=len(population))
        self.restaurants = Restaurant(dims=(20, 20), **restaurant_config)
        self.bar = Bar(bar_shape="U", **bar_config)
        self.graveyard = Graveyard(dims=(20, 15), lots=len(population))
        self._build_world()
        self.relocator = Relocator(self.population, self._world)

        self.fnf_network = nx.Graph()

    def venues(self):
        return [self.restaurants, self.bar]

    def _build_world(self):
        service_row_padding = 6
        services_row = [[BlankSpace((1, 1)), *self.offices],  # column 1
                        [BlankSpace((service_row_padding, 2)), self.restaurants,  # column 3
                         BlankSpace((service_row_padding, 2)), self.bar,
                         BlankSpace((20, 2)), ],
                        [
                         self.hospital,
                         BlankSpace((2, 2)), self.graveyard],
                        ]
        homes_row = [BlankSpace((1, 2)), self.homes, BlankSpace((5, 2))]
        cars_row = [BlankSpace((1, 1)), self.cars, BlankSpace((2, 1)), self.buses, BlankSpace((1, 1))]
        grid = [services_row,
                [BlankSpace((20, 5))],
                cars_row,
                [BlankSpace((40, 5))],
                homes_row,
                [BlankSpace((40, 5))]]

        w, h = self.arrange_grid(grid, 10)
        w, h = self.arrange_column([self.cars], 25, offset=self.cars[0].origin)
        self.left = 2
        self.compact_grid(grid)
        self._world = CompositeWorld(population=self.population, regions=list(ravel_grids(grid)))
        # Add some padding
        self._world.dims += 0, 3

    def draw_world(self, ax=None, padding=5, bbox=False, draw_population=True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        ax.set_axis_off()
        ax.set_aspect("equal", adjustable="box", anchor="NW")
        self.world.draw_world(ax, bbox=bbox, **kwargs)

        ax.set_xlim(self.world.origin[0] - padding, self.world.dims[0] + padding)
        ax.set_ylim(self.world.origin[1] - padding, self.world.dims[1] + padding)

        text_offset = [-3, 5]
        ax.text(*(self.hospital.origin - text_offset), 'Hospital',
                verticalalignment='bottom', horizontalalignment='left',
                # transform=ax.transAxes,
                fontsize=10)

        ax.text(*(self.offices[0].origin - text_offset), 'Offices/Schools',
                verticalalignment='bottom', horizontalalignment='left',
                # transform=ax.transAxes,
                fontsize=10)

        text_origin = self.restaurants.origin + [self.restaurants.dims[0]/2, -text_offset[1]]
        ax.text(*text_origin, 'Restaurant',
                verticalalignment='bottom', horizontalalignment='center',
                # transform=ax.transAxes,
                fontsize=10)

        text_origin = self.bar.origin + [self.bar.dims[0]/2, -text_offset[1]]
        ax.text(*text_origin, 'Bar',
                verticalalignment='bottom', horizontalalignment='center',
                # transform=ax.transAxes,
                fontsize=10)

        text_origin = self.graveyard.origin + [self.graveyard.dims[0]/2, -text_offset[1]]
        ax.text(*text_origin, 'Cemetery',
                verticalalignment='bottom', horizontalalignment='center',
                # transform=ax.transAxes,
                fontsize=10)

        width_bus_block = ((self.buses[-1].origin[0] + self.buses[-1].dims[0]) - self.buses[0].origin[0])
        text_origin = self.buses[0].origin + [width_bus_block/2, -text_offset[1]]
        ax.text(*text_origin, 'Buses',
                verticalalignment='bottom', horizontalalignment='center',
                # transform=ax.transAxes,
                fontsize=10)

        text_origin = np.array([self.homes[0].origin[0], self.homes[-1].origin[1]]) - text_offset
        ax.text(*text_origin, 'Homes',
                verticalalignment='bottom', horizontalalignment='left',
                # transform=ax.transAxes,
                fontsize=10)

        text_origin = np.array([self.cars[0].origin[0], self.cars[-1].origin[1]]) - text_offset
        ax.text(*text_origin, 'Cars',
                verticalalignment='bottom', horizontalalignment='left',
                # transform=ax.transAxes,
                fontsize=10)

        if draw_population:
            scat = ax.scatter(*self.relocator.get_absolute_positions().T, s=16, animated=True)
        else:
            scat = ax.scatter([], [], s=16)

        return ax, scat

    def dynamic_regions(self):
        return self.homes + [self.hospital]

    def assign_homes(self):
        n = len(self.population)
        if not hasattr(self.population, "home"):
            self.population.add_property("home", np.empty((n,), dtype=object))

        if not hasattr(self.population, "car"):
            self.population.add_property("car", np.empty((n,), dtype=object))

        slice_ = 0
        agent_slice = 0
        homes = []
        cars = []
        for hh_size, p in enumerate(self.home_percentages, 1):
            num_homes = int(len(self.homes) * p)
            if hh_size == len(self.home_percentages) - 1:
                homes = self.homes[slice_:]
                cars = self.cars[slice_:]

            else:
                homes = self.homes[slice_:slice_ + num_homes]
                cars = self.cars[slice_:slice_ + num_homes]

            slice_ += len(homes)
            for i in range(hh_size):
                if agent_slice + len(homes) > n:
                    self.population.home[agent_slice:] = homes[:len(self.population.home[agent_slice:])]
                    self.population.car[agent_slice:] = cars[:len(self.population.car[agent_slice:])]
                    agent_slice = n
                    break

                self.population.home[agent_slice:agent_slice+len(homes)] = homes
                self.population.car[agent_slice:agent_slice + len(homes)] = cars
                agent_slice += len(homes)

            if agent_slice >= n:
                break

        if agent_slice < n:
            if len(homes) == 0 or (n - agent_slice) / len(homes) + hh_size > 10:
                self.population.home[agent_slice:] = np.random.choice(self.homes, n - agent_slice)
                self.population.car[agent_slice:] = np.random.choice(self.cars, n - agent_slice)
            else:
                self.population.home[agent_slice:] = np.random.choice(homes, n - agent_slice)
                self.population.car[agent_slice:] = np.random.choice(cars, n - agent_slice)

        self._world.containment_region[:] = self.population.home

    def create_social_network(self):
        n = len(self.population)

        for office in self.offices:
            idx = self.population.office == office
            self.social_network.update({i: Office for i in combinations(self.population.index[idx], 2)})

        # This overrides existing networks giving preference for the home network.
        for house in self.homes:
            idx = self.population.home == house
            self.social_network.update({i: type(house) for i in combinations(self.population.index[idx], 2)})

    def assign_offices(self):
        n = len(self.population)
        if not hasattr(self.population, "office"):
            self.population.add_property("office", np.empty((n,), dtype=object))

        self.population.office[:] = np.random.choice(self.offices, (n,))

    def move_home(self):
        for home_id, home in enumerate(self.homes):
            home.move_home(self.population[self.population.home == home])
            self.relocator.move_agents((self.population.home == home), home)

    def create_fnf_network(self):
        for office in self.offices:
            q = self.population.office == office
            office_idx = self.population.index[q]
            all_connections = np.array(list(combinations(office_idx, 2)))

            friends_idx = np.random.choice(range(len(all_connections)), int(len(office_idx) * 0.4))
            friend_connections = all_connections[friends_idx, :]
            edge_props = {"type": "friend", "recall": 0.80, "dropout": 0.10, "recall_probability": 0}
            self.fnf_network.add_edges_from([(n1, n2, edge_props) for n1, n2 in friend_connections])

            acquaintances = np.array(list(set([tuple(r) for r in all_connections]) - set([tuple(r) for r in
                                                                                          friend_connections])))
            acquaintances_idx = np.random.choice(range(len(acquaintances)), int(len(acquaintances) * 0.5))
            acquaintances_connections = acquaintances[acquaintances_idx, :]
            edge_props = {"type": "acquaintance", "recall": 0.40, "dropout": 0.50, "recall_probability": 0}
            self.fnf_network.add_edges_from([(n1, n2, edge_props) for n1, n2 in acquaintances_connections])

        for home in self.homes:
            q = self.population.home == home
            family_idx = self.population.index[q]
            edge_props = {"type": "family", "recall": 0.99, "dropout": 0.01, "recall_probability": 0}

            self.fnf_network.add_nodes_from(family_idx)
            self.fnf_network.add_edges_from([(n1, n2, edge_props) for n1, n2 in combinations(family_idx, 2)])


class ApartmentsScenario(ComplexWorld):
    class Home(Apartment):
        """Dummy class to have nicer printout names"""
        pass

    def __init__(self, population, homes=10, offices=3, home_percentages=None, buses=3, **kwargs):
        super(ComplexWorld, self).__init__(population, **kwargs)

        self.social_network = {}
        if home_percentages is None:
            # https://www.statista.com/statistics/464187/households-by-size-germany/ 2019
            home_percentages = [0.42298889, 0.33201629, 0.11930518, 0.09114125, 0.03454839]

        if buses <= 3:
            buses = 3

        print(f"Using {buses} buses, {homes} homes, {offices} offices.")

        self.homes = [ApartmentsScenario.Home(num_residents=6, rotation=0) for h in range(homes)]
        self.home_percentages = home_percentages
        self.offices = [Office((10, 10)) for o in range(offices)]
        self.buses = [Bus(orientation="horizontal") for i in range(buses)]
        self.cars = [Car(rotation=90) for i in range(len(self.homes))]
        self.hospital = Hospital(dims=(25, 10), beds=len(population))
        bar_config = kwargs.get("bar_config", {})
        restaurant_config = kwargs.get("restaurant_config", {})
        self.restaurants = Restaurant(dims=(20, 20), **restaurant_config)
        self.bar = Bar(bar_shape="U", **bar_config)
        self.graveyard = Graveyard(dims=(20, 15), lots=len(population))
        self._build_world()
        self.relocator = Relocator(self.population, self._world)

        self.fnf_network = nx.Graph()

    def move_home(self):
        for home_id, home in enumerate(self.homes):
            home.move_home(self.population[self.population.home == home])
            self.relocator.move_agents((self.population.home == home), home)

