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

from collections import Counter

import numpy as np
import pandas as pd

from i2mb.pathogen import UserStates
from i2mb.utils import global_time


def compute_incubation_periods(state_history_):
    incubation_period = state_history_["symptom onset"] - state_history_["infected"]

    # Convert period form ticks to days
    incubation_period = incubation_period

    # remove patient 0
    incubation_period = incubation_period[~(incubation_period == -np.inf)]
    incubation_period = incubation_period[~(incubation_period == np.inf)]
    incubation_period = incubation_period[~incubation_period.isnull()]

    return incubation_period


def compute_illness_duration(state_history_):
    illness_duration = np.zeros_like(state_history_["symptom onset"])

    recovered = state_history_["symptom onset"] != np.inf
    illness_duration[recovered] = (state_history_["immune"] - state_history_["symptom onset"])[recovered]

    deceased = state_history_["deceased"] != np.inf
    illness_duration[deceased] = (state_history_["deceased"] - state_history_["symptom onset"])[deceased]

    return illness_duration


def compute_serial_intervals(state_history, infection_map, asymptomatic_list=None):
    serial_intervals = []
    if asymptomatic_list is None:
        asymptomatic_list = set()

    for infector, infected in infection_map.items():
        if infector in asymptomatic_list:
            continue

        for infectee in infected:
            if infectee in asymptomatic_list:
                continue

            if isinstance(state_history, pd.DataFrame):
                serial_interval = (state_history.loc[infectee, "symptom onset"]
                                   - state_history.loc[infector, "symptom onset"])
            else:
                serial_interval = (state_history[infectee, UserStates.infectious]
                                   - state_history[infector, UserStates.infectious])

            if serial_interval < np.inf:
                serial_intervals.append(serial_interval)

    return serial_intervals


def compute_activity_duration(state_history):
    return state_history["activity_duration"]


def compute_generation_intervals(state_history, infection_map):
    generation_intervals = []
    for infector, infected in infection_map.items():
        for infectee in infected:
            if isinstance(state_history, pd.DataFrame):
                generation_interval = (state_history.loc[infectee, "infected"] -
                                       state_history.loc[infector, "infected"])
            else:
                generation_interval = (state_history[infectee, UserStates.infected] -
                                       state_history[infector, UserStates.infected])
            generation_intervals.append(generation_interval)

    return generation_intervals


def multiplier_interval(infection_times, multiplier=2):
    """Compute the multiplier interval Tm. Given the following formula

    N(t) = N_0 * multiplier ** (t / Tm)

    this function solves for Tm. The default is the doubling interval, i.e., teh amount og time it takes an intial
    population of size N_0 to double in time.

    The infection times parameter is a list with all the times agents where infected, and computes the multiplier
    interval such that it fits the data.
     return interval in days
    """
    n = len(infection_times)
    infection_times = infection_times / global_time.time_scalar
    min_time = min(infection_times)
    max_time = max(infection_times)
    n0 = Counter(infection_times)[min_time]
    return max_time * np.log(multiplier) / np.log(n / n0)


def reproduction_number(infection_map, num_days=1):
    """Compute the reproduction number averaged over num days"""
    buckets = {}
    for index_case, infected in infection_map.items():
        infection_times = [t / global_time.time_scalar for inf_times in infected.values() for t in inf_times]
        infection_time_index = [t // num_days for t in infection_times]
        index_buckets = Counter(infection_time_index)
        for bucket, counts in index_buckets.items():
            buckets.setdefault(bucket, []).append(counts)

    r = {k*num_days: np.mean(v) for k, v in buckets.items()}
    return r