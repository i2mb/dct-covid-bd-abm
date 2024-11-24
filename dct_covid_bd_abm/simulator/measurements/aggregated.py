
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

from i2mb.activities import ActivityProperties
from i2mb.measurements.time_series.user_states import get_location_contracted
from i2mb.pathogen import SymptomLevels

if TYPE_CHECKING:
    pass


def get_visit_counter(experiment: "DCTvMCTExperiment"):
    visit_counter = experiment.scenario.relocator.visit_counter
    return visit_counter


def get_contracted_per_location(experiment: "DCTvMCTExperiment"):
    locations = experiment.locations
    loc_of_infection = get_location_contracted(experiment.population, locations)
    return loc_of_infection


def get_pathogen_metrics(experiment: "DCTvMCTExperiment"):
    metrics = dict(
        waves=experiment.sim_engine.pathogen.waves,
        infection_map=experiment.sim_engine.pathogen.infection_map,
        contact_map=experiment.sim_engine.pathogen.contact_map,
        location_map=experiment.sim_engine.pathogen.location_contracted)

    return metrics


def get_assigned_locations(experiment: "DCTvMCTExperiment"):
    metrics = dict(
        homes=get_homes(experiment.population),
        offices=get_offices(experiment.population))

    return metrics


def get_pathogen_agent_stats(experiment: "DCTvMCTExperiment"):
    return {"Symptom Level": experiment.sim_engine.pathogen.symptom_levels.ravel()}


def get_intervention_agent_stats(experiment: "DCTvMCTExperiment"):
    agent_stats = {}
    ct_intervention = experiment.sim_engine.ct_intervention
    if ct_intervention is not None:
        agent_stats.update({"Isolation FP": ct_intervention.isolated_fp.ravel(),
                            "Time In Isolation": ct_intervention.time_in_isolation.ravel(),
                            "Num. Isolations": ct_intervention.num_confinements.ravel(),
                            })
    return agent_stats


def get_activity_times(experiment: "DCTvMCTExperiment"):
    accumulated_ix = ActivityProperties.accumulated
    activity_times = experiment.sim_engine.activity_manager.activity_list.activity_values[:, accumulated_ix, :]
    activity_names = [str(t).split(".")[-1][:-2] for t in experiment.sim_engine.activity_manager.activity_types]
    return {n: c for n, c in zip(activity_names, activity_times.T)}


def get_homes(population):
    return np.array([f"{id(h)}" for h in population.home])


def get_offices(population):
    return np.array([f"{id(o)}" for o in population.office])


def get_symptom_levels_summary(population):
    return (population.symptom_level == SymptomLevels).sum(axis=0)
