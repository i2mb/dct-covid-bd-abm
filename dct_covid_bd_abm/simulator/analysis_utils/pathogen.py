# Pathogen Data Loaders
from collections import Counter
from functools import partial

import numpy as np
import pandas as pd

from dct_covid_bd_abm.simulator.utilities.cache_utilities import cached
from dct_covid_bd_abm.simulator.measurements import post_processing as pp
from dct_covid_bd_abm.simulator.utilities.file_utilities import load_metadata


def load_state_history(path_npz):
    npz_content = np.load(path_npz, allow_pickle=True)
    state_history = npz_content["states"]
    state_history = pd.DataFrame(state_history,
                                 columns=["susceptible", "immune", "deceased", "exposed", "infected", "symptom onset"])

    return state_history


def load_quarantine_history(path_npz):
    npz_content = np.load(path_npz, allow_pickle=True)
    try:
        q_history = npz_content["q_history"][0]
    except KeyError as e:
        print(f"No q_history {path_npz}")
        q_history = {}

    return q_history


def load_waves(path_npz):
    npz_content = np.load(path_npz, allow_pickle=True)
    wave_duration = npz_content["waves"]
    return wave_duration


def load_infection_location(path_npz):
    npz_content = np.load(path_npz, allow_pickle=True)
    return npz_content["location_map"]


def load_infection_map(path_npz, unit="samples"):
    npz_content = np.load(path_npz, allow_pickle=True)
    if unit == "samples":
        return npz_content["infection_map"].item()

    # Convert time from samples to days
    if unit == "days":
        infection_map = npz_content["infection_map"].item()
        time_factor_ticks_day = load_metadata(path_npz, "time_factor_ticks_day")
        return {key: {i_key: [i / time_factor_ticks_day for i in i_value]
                           for i_key, i_value in values.items()}
                     for key, values in infection_map.items()}


def load_state_time_series(npz_file):
    hdf_file = npz_file.replace(".npz", "_results.hdf")
    stats = pd.read_hdf(hdf_file)
    return stats


def load_isolation_stats(path_npz):
    hdf_file = path_npz.replace(".npz", ".hdf")
    stats = pd.read_hdf(hdf_file)
    try:
        return stats.loc[:, ["Isolation FP", "Num. Isolations", "Time In Isolation"]]
    except KeyError:
        stats.loc[:, ["Isolation FP", "Num. Isolations", "Time In Isolation"]] = 0
        return stats.loc[:, ["Isolation FP", "Num. Isolations", "Time In Isolation"]]


def get_serial_interval_map(infection_map, state_history, time_factor=1):
    """Converts the infection time from global time to serial interval, i.e., the interval of time between the
    infected and infector symptom onsets."""
    si_infection_map = {key: {i_key: [(state_history["symptom onset"][i_key] -
                                       state_history["symptom onset"][key])/time_factor]
                              for i_key, i_value in values.items()}
                        for key, values in infection_map.items()}

    return si_infection_map


def get_serial_interval_per_infector(infection_map, state_history):
    rel_infection_map = get_serial_interval_map(infection_map, state_history)
    si_intervals = {key: np.mean([iv[0] for iv in values.values()]) for key, values in
                    rel_infection_map.items()}
    return pd.DataFrame(rel_infection_map, index=["SI"]).T.sort_index()


def get_serial_intervals(npz_file, compute_only_on_symptomatic_patients=False):
    state_history = load_state_history(npz_file)
    infection_map = load_infection_map(npz_file)

    # try:
    #     contact_history = load_contact_history(npz_file)
    #     fix_infection_map(infection_map, state_history, contact_history)
    # except FileNotFoundError:
    #     pass

    time_scalar = load_metadata(npz_file, "time_factor_ticks_day")
    if compute_only_on_symptomatic_patients:
        symptoms_level = get_symptom_levels(npz_file)
        asymptomatics = symptoms_level == 0
        asymptomatic_list = asymptomatics.index[asymptomatics.values.ravel()]
        return [si / time_scalar for si in pp.compute_serial_intervals(state_history, infection_map, asymptomatic_list)]

    return [si / time_scalar for si in pp.compute_serial_intervals(state_history, infection_map)]


def get_generation_interval_map(infection_map, state_history, time_factor=1):
    """Converts the infection time from global time to generation interval, i.e., the interval of time between the
    infected and infector infection times"""

    relative_infection_map = {key: {i_key: [(i_value[0] - state_history.infected[key])/time_factor]
                                    for i_key, i_value in values.items()}

                              for key, values in infection_map.items()}

    return relative_infection_map


def get_generation_interval_per_infector(infection_map, state_history):
    rel_infection_map = get_generation_interval_map(infection_map, state_history)
    generation_intervals = {key: np.mean([iv[0] for iv in values.values()]) for key, values in
                            rel_infection_map.items()}
    return pd.DataFrame(generation_intervals, index=["GI"]).T.sort_index()


get_serial_intervals_symptomatic_only = partial(get_serial_intervals, compute_only_on_symptomatic_patients=True)


def get_generation_intervals(npz_file, location_filter=None, negate_filter=False):
    state_history = load_state_history(npz_file)
    infection_map = load_infection_map(npz_file)

    # try:
    #     contact_history = load_contact_history(npz_file)
    #     fix_infection_map(infection_map, state_history, contact_history)
    # except FileNotFoundError:
    #     pass

    time_scalar = load_metadata(npz_file, "time_factor_ticks_day")

    if location_filter is not None:
        if negate_filter:
            location_map = load_infection_location(npz_file) != location_filter
        else:
            location_map = load_infection_location(npz_file) == location_filter

        infection_map = dict(filter(lambda x: x[1], [(k, list(np.arange(len(location_map))[list(i)][location_map[list(i)].ravel()])) for k, i in infection_map.items()]))

    return [gi / time_scalar for gi in pp.compute_generation_intervals(state_history, infection_map)]


def get_external_contact_generation_intervals(npz_file):
    return get_generation_intervals(npz_file, location_filter="home", negate_filter=True)


def get_incubation_periods(npz_file):
    state_history = load_state_history(npz_file)
    time_scalar = load_metadata(npz_file, "time_factor_ticks_day")
    ip = pp.compute_incubation_periods(state_history) / time_scalar
    ip = ip[~ip.isnull()]

    return ip


def get_illness_durations(npz_file, full_list=False):
    state_history = load_state_history(npz_file)
    time_scalar = load_metadata(npz_file, "time_factor_ticks_day")
    illness_durations = pp.compute_illness_duration(state_history) / time_scalar
    if full_list:
        return illness_durations
    else:
        return illness_durations[illness_durations > 0]


def get_symptom_levels(npz_file):
    hdf_file = npz_file.replace(".npz", ".hdf")
    symptoms_level = pd.read_hdf(hdf_file)
    return symptoms_level["Symptom Level"]


def compute_infected_per_day(state_history):
    num_days = int(np.nanmax(state_history)) + 1
    inf_per_day = np.zeros(num_days)
    recovered = state_history.loc[:, ["deceased", "immune"]].sum(axis=1)
    for a_id, agent in state_history.iterrows():
        if np.isnan(agent["infected"]):
            continue

        day_infection = int(agent["infected"])
        day_recovery = int(recovered[a_id])
        inf_per_day[day_infection:day_recovery] += 1

    return inf_per_day


def get_reproduction_number(npz_file, generation_interval=None, num_days_averaged=0):
    if generation_interval is None:
        generation_interval = int(np.array(get_generation_intervals(npz_file)).mean().round())
    else:
        generation_interval = int(round(generation_interval))

    time_scalar = load_metadata(npz_file, "time_factor_ticks_day")
    state_history = load_state_history(npz_file)
    state_history /= time_scalar
    state_history[state_history == np.inf] = np.nan
    inf_per_day = compute_infected_per_day(state_history)

    r = np.zeros(len(inf_per_day) + generation_interval)
    ratio = inf_per_day[generation_interval:] / inf_per_day[:-generation_interval]
    r[generation_interval:-generation_interval] = ratio
    if num_days_averaged > 0:
        r = np.convolve(r, np.ones(num_days_averaged) / num_days_averaged)

    return r


def get_wave_duration(npz_file):
    time_scalar = load_metadata(npz_file, "time_factor_ticks_day")
    wave_duration = load_waves(npz_file) / time_scalar
    return np.diff(wave_duration).ravel()


@cached
def get_affected_population(npz_file):
    state_history = load_state_history(npz_file)
    return (state_history.loc[:, ["immune", "deceased"]] != np.inf).sum()


def _get_days_in_isolation(npz_file):
    time_scalar = load_metadata(npz_file, "time_factor_ticks_day")
    days_in_isolation = load_isolation_stats(npz_file)["Time In Isolation"] / time_scalar
    return days_in_isolation

@cached
def get_days_in_isolation(npz_file, full_list=False):
    config = load_metadata(npz_file, "sim_engine")
    time_scalar = load_metadata(npz_file, "time_factor_ticks_day")
    test_delays = config["tester"]["duration"] + config["get_tested"]["delay"]
    test_delays = test_delays / time_scalar
    isolation = get_illness_durations(npz_file, full_list=True)
    isolation[isolation > 0] -= test_delays
    if full_list:
        return isolation
    else:
        return isolation[isolation > 0]

@cached
def get_days_in_quarantine(npz_file):
    days_in_quarantine = _get_days_in_isolation(npz_file)
    days_in_isolation = get_days_in_isolation(npz_file, full_list=True)
    days_in_quarantine -= days_in_isolation
    days_in_quarantine[days_in_quarantine < 0] = 0

    return days_in_quarantine[days_in_quarantine > 0]


def get_days_in_isolation_normalized_by_quarantines(npz_file):
    time_scalar = load_metadata(npz_file, "time_factor_ticks_day")
    isolating_stats = load_isolation_stats(npz_file)
    days_in_isolation = isolating_stats["Time In Isolation"]
    quarantines = isolating_stats["Num. Isolations"]
    days_in_isolation /= time_scalar
    days_in_isolation /= quarantines
    days_in_isolation[quarantines == 0] = 0
    return days_in_isolation

@cached
def get_isolation_false_discovery_rate(npz_file):
    isolating_stats = load_isolation_stats(npz_file)
    quarantines = isolating_stats["Num. Isolations"].sum()
    false_positives = get_isolation_false_positives(npz_file)
    if quarantines == 0:
        return np.array([0])

    return false_positives / quarantines


@cached
def get_isolation_false_positives(npz_file):
    isolating_stats = load_isolation_stats(npz_file)
    false_positive = isolating_stats["Isolation FP"]
    return false_positive.values.sum(keepdims=True)


@cached
def get_isolation_false_negatives(npz_file, full_list=False):
    days_quarantine = load_metadata(npz_file, "sim_engine")["get_tested"]["quarantine_duration"]
    status = load_state_history(npz_file)
    q_history = load_quarantine_history(npz_file)
    population_size = len(status)
    false_negatives = np.zeros(population_size)
    for uid in np.arange(population_size)[status["infected"] < np.inf]:
        sickness_period = (status.loc[uid, "infected"], (status.loc[uid, "immune"]
                                                     if status.loc[uid, "immune"] < status.loc[uid, "deceased"]
                                                     else status.loc[uid, "deceased"]))
        if uid in q_history:
            isolation_periods = np.array([r[1:] for r in q_history[uid]])
            starts = isolation_periods[:, 0]
            ends = isolation_periods[:, 1]

            if (ends == None).any():
                ends[ends == None] = starts[ends == None] + days_quarantine

            isolation_period = ((starts >= sickness_period[0]) & (starts <= sickness_period[1]) |
                                (ends >= sickness_period[0]) & (ends <= sickness_period[0]))
            false_negatives[uid] += ~isolation_period.any()

        else:
            false_negatives[uid] += 1

    if full_list:
        return false_negatives

    return false_negatives[status["infected"] < np.inf].sum(keepdims=True)


@cached
def get_isolation_false_negative_rate(npz_file):
    status = load_state_history(npz_file)
    return get_isolation_false_negatives(npz_file) / (status["infected"] < np.inf).sum()


def get_isolation_f_metric(npz_file, beta=1):
    beta_squared = beta ** 2
    tp = get_affected_population(npz_file).values.sum(keepdims=True)
    fp = get_isolation_false_positives(npz_file)
    fn = get_isolation_false_negatives(npz_file)
    return (1 + beta_squared) * tp / ((1 + beta_squared) * tp + beta_squared * fn + fp)

def get_isolation_fowlkes_mallows_index(npz_file):
    ppv = 1 - get_isolation_false_discovery_rate(npz_file)
    tpr = 1 - get_isolation_false_negative_rate(npz_file)
    return np.sqrt(ppv * tpr)


def get_isolation_ppv(npz_file):
    return 1 - get_isolation_false_discovery_rate(npz_file)


def get_isolation_sensitivity(npz_file):
    return 1 - get_isolation_false_negative_rate(npz_file)


def get_state_time_series(npz_file):
    time_scalar = load_metadata(npz_file, "time_factor_ticks_day")
    time_series = load_state_time_series(npz_file)
    time_series.index = time_series.index / time_scalar
    return time_series


def value_per_100_000(value, population_size):
    """
    Converts a population value to a per 100.000 individual representation
    Parameters
    ----------
    value
    original variable
    population_size
    Size of the population where the parameter was measured.

    Returns
    -------
    ´value´ in per 100.000 individuals
    """
    return value * 100_000 / population_size


def get_infection_incidence(npz_file, window_size=7):
    time_scalar = load_metadata(npz_file, "time_factor_ticks_day")
    infected = load_state_history(npz_file)["symptom onset"]
    population_size = len(infected)
    infected = infected[~(infected == np.inf)]
    data = Counter((infected // time_scalar).astype(int))
    max_data = max(data) + window_size * 4
    data = {"Daily Incidence": data}
    data = pd.DataFrame(data, index=np.arange(max_data))
    data[data.isnull()] = 0
    return data.rolling(window_size).mean()


def get_observable_infection_incidence(npz_file, window_size=7):
    time_scalar = load_metadata(npz_file, "time_factor_ticks_day")
    infected = load_state_history(npz_file)["symptom onset"]
    symptomatic = (get_symptom_levels(npz_file) > 0)
    data = Counter((infected[symptomatic.ravel()] // time_scalar).astype(int))
    max_data = max(data) + window_size * 4
    data = {"Daily Incidence": data, }
    data = pd.DataFrame(data, index=np.arange(max_data))
    return value_per_100_000(data.rolling(window_size).mean(), len(infected))


def get_hospitalization_incidence(npz_file, window_size=7):
    time_scalar = load_metadata(npz_file, "time_factor_ticks_day")
    infected = load_state_history(npz_file)["symptom onset"]
    symptom_levels = get_symptom_levels(npz_file)
    icu_patients = symptom_levels == 4
    hospitalised_patients = symptom_levels >= 3
    data_icu = Counter((infected[icu_patients] // time_scalar).astype(int))
    data_icu[0] = 0
    data_hospitalised = Counter((infected[hospitalised_patients] // time_scalar).astype(int))
    data_hospitalised[0] = 0
    max_data = max(max(data_icu), max(data_hospitalised)) + 4 * window_size
    data = {"ICU": data_icu, "Hospitalised": data_hospitalised}
    data = pd.DataFrame(data, index=np.arange(max_data))
    data = data.fillna(0)
    return data.rolling(window_size).mean()


