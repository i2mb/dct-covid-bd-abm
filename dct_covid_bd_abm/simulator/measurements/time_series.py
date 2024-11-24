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

from typing import TYPE_CHECKING

import numpy as np

from i2mb.measurements.time_series.user_states import *
from i2mb.pathogen import UserStates

if TYPE_CHECKING:
    pass


def measure_time_series_stats(experiment: "DCTvMCTExperiment", frame):
    user_states_array = experiment.user_states_array
    sim_engine = experiment.sim_engine
    values = get_population_state_summary(experiment.population, user_states_array)
    t = {k.name: v for k, v in zip(UserStates, values)}

    if experiment.sim_engine.ct_intervention is not None:
        t["quarantined"] = get_number_of_isolated_agents(experiment.population)
        t["quarantined_cumulative"] = get_total_number_of_isolations(experiment.population)
        t["fp_rate"] = get_false_positive_rate_of_isolated_agents(experiment.population)

        contacted_by = notified_by_contact_type(sim_engine)
        t["self_isolated"], t["fnf_contacted"], t["mct_contacted"], t["dct_contacted"], t["hh_contacted"] = contacted_by

    experiment.time_series_stats.append(t)


def update_state_history_table(experiment: "DCTvMCTExperiment", frame):
    user_states_array = experiment.user_states_array
    population = experiment.population
    mask = (population.state == user_states_array) * frame * 1.0
    mask[mask == 0] = np.inf
    states = experiment.agent_history["states"]
    states[states > mask] = frame


def update_time_spent_per_location(experiment: "DCTvMCTExperiment", frame):
    """Ignores sleeping hours"""
    population = experiment.population
    awake = get_awake_agents(population)
    if not awake.any():
        return

    location_names = get_location_names(population, awake)
    time_spent_per_location = experiment.agent_history["time_spent_by_location"]
    for loc_, loc_array in time_spent_per_location.items():
        loc_array[awake] += location_names == loc_


def notified_by_contact_type(sim_engine):
    self_isolated = sim_engine.get_tested.self_isolated
    fnf_contacted = sim_engine.fnf_tracing.fnf_contacted
    if sim_engine.m_tracing is not None:
        mct_contacted = sim_engine.m_tracing.num_contacted
    else:
        mct_contacted = np.nan

    if sim_engine.c_tracing is not None:
        dct_contacted = sim_engine.c_tracing.dct_contacted
    else:
        dct_contacted = np.nan

    if sim_engine.ct_intervention is not None:
        hh_contacted = sim_engine.ct_intervention.hh_contacted
    else:
        hh_contacted = np.nan

    return self_isolated, fnf_contacted, mct_contacted, dct_contacted, hh_contacted


def currently_isolated_by_contact_type_code(sim_engine):
    """Alternative to isolation_by_contact_type"""
    isolation_codes = np.array(sim_engine.get_isolation_codes())
    if sim_engine.ct_intervention is None:
        return np.zeros_like(isolation_codes)

    return (sim_engine.ct_intervention.isolated_by == isolation_codes).sum(axis=0)
