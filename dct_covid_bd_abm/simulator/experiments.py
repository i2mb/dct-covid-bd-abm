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

import numpy as np

from i2mb.engine.experiment import Experiment
from i2mb.pathogen import SymptomLevels, UserStates
from i2mb.utils import global_time
from dct_covid_bd_abm.simulator.covid19_engine import COVID19ContactTracing
from dct_covid_bd_abm.simulator.measurements.aggregated import get_contracted_per_location, get_assigned_locations, get_pathogen_metrics, \
    get_intervention_agent_stats, get_pathogen_agent_stats, get_visit_counter
from dct_covid_bd_abm.simulator.measurements.time_series import update_time_spent_per_location, update_state_history_table, \
    measure_time_series_stats
from dct_covid_bd_abm.simulator.scenarios.scenario_complex_world import ApartmentsScenario


class DCTvMCTExperiment(Experiment):
    def __init__(self, run_id, config):
        super().__init__(run_id, config)
        self.relocator = None
        if self.skip_run():
            print(f"Run {self.run_id} from '{self.config_name}.{self.name}' skipped. Files exist.")
            return

        self.scenario = self.build_scenario()
        self.config["sim_engine"]["base_file_name"] = self.get_filename()
        self.sim_engine = None
        self.sim_engine = COVID19ContactTracing(self.scenario, **config["sim_engine"])

        # Runtime variables
        self.days_after_wave_end = config.get("days_after_wave_end", 2)
        self.wave_ended = 0
        self.wave_started = False
        self.day = -1
        self.user_states_array = np.array(list(UserStates)).reshape(1, -1)
        self.sim_steps = config.get("sim_steps", None)

        # Locations of interest
        self.locations = config.get("locations", ["home", "office", "bar", "restaurant", "bus"])

        # Time_series containers
        self.agent_history["time_spent_by_location"] = {k: np.zeros_like(self.population.index, dtype=int) for
                                                        k in self.locations}
        self.agent_history["states"] = np.ones((config["population_size"], len(list(UserStates)))) * np.inf

    def process_stop_criteria(self, frame):
        if self.day != global_time.days(frame):
            self.day = global_time.days(frame)

        no_one_in_isolation = True
        if self.sim_engine.ct_intervention is not None:
            no_one_in_isolation = self.sim_engine.ct_intervention.isolated.sum() == 0

        wave_done = False
        if self.sim_engine.pathogen is not None:
            wave_done = self.sim_engine.pathogen.wave_done

        if self.wave_started and wave_done and no_one_in_isolation:
            self.wave_started = False
            self.wave_ended = self.day

        min_frames = frame > 50
        delayed_stop = (self.day - self.wave_ended) >= self.days_after_wave_end

        walked_enough = False
        if self.sim_steps is not None:
            walked_enough = frame > self.sim_steps

        return min_frames and ((no_one_in_isolation and wave_done and delayed_stop) or walked_enough)

    def collect_time_series_data(self, frame):
        update_time_spent_per_location(self, frame)
        update_state_history_table(self, frame)
        if self.generate_time_series:
            measure_time_series_stats(self, frame)

    def process_trigger_events(self, frame):
        insert_pathogen = self.config.get("insert_pathogen", 1)
        if frame == insert_pathogen and self.sim_engine.pathogen is not None:
            symptom_level = self.config.get("SymptomLevels.mild", SymptomLevels.mild)
            skip_incubation = self.config.get("skip_incubation", True)
            self.sim_engine.pathogen.introduce_pathogen(self.config["num_patient0"], frame,
                                                        symptoms_level=symptom_level,
                                                        skip_incubation=skip_incubation)
            self.wave_started = True

    def collect_aggregated_data(self):
        if self.sim_engine.pathogen is not None:
            self.agent_history["location_of_infection"] = get_contracted_per_location(self)
            self.agent_history.update(get_pathogen_metrics(self))

        self.agent_history.update(get_assigned_locations(self))

        if self.sim_engine.ct_intervention is not None:
            self.agent_history["q_history"] = self.sim_engine.ct_intervention.q_history,

        self.final_agent_stats.update(get_intervention_agent_stats(self))

        if self.sim_engine.pathogen is not None:
            self.final_agent_stats.update(get_pathogen_agent_stats(self))

        # self.agent_history["activity_duration"] = get_activity_times(self)
        self.agent_history["visit_counter"] = get_visit_counter(self)
        return

    def build_scenario(self):
        scenario_cfg = self.config.get("scenario", {})
        scenario_class = scenario_cfg.get("class", ApartmentsScenario)
        scenario_parameters = scenario_cfg.get("kwargs", {})
        scenario = scenario_class(self.population, **scenario_parameters)
        self.relocator = scenario.relocator
        scenario.assign_homes()
        scenario.assign_offices()
        scenario.create_social_network()
        scenario.move_home()
        scenario.create_fnf_network()
        return scenario
