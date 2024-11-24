import numpy as np

from dct_covid_bd_abm.simulator.analysis_utils import get_illness_durations
from dct_covid_bd_abm.simulator.analysis_utils.pathogen import get_wave_duration, get_days_in_quarantine, get_days_in_isolation, load_state_history, \
    get_isolation_false_negatives
from dct_covid_bd_abm.simulator.utilities.cache_utilities import cached
from dct_covid_bd_abm.simulator.utilities.file_utilities import load_metadata


@cached
def get_total_samples(npz_file):
    days = get_wave_duration(npz_file).round(0)
    population_size = load_metadata(npz_file, 'population_size')
    return days * population_size

@cached
def get_false_positives(npz_file):
    quarantined = get_days_in_quarantine(npz_file)
    return quarantined.values.sum(keepdims=True).round(0)

@cached
def get_true_positives(npz_file):
    isolated = get_days_in_isolation(npz_file, full_list=True)
    return isolated.sum(keepdims=True).round(0)

@cached
def get_false_negatives(npz_file):
    false_negatives = get_isolation_false_negatives(npz_file, full_list=True)
    sickness_period =  get_illness_durations(npz_file, full_list=True)
    # sickness_period.sum() - get_true_positives(npz_file)

    return (sickness_period[false_negatives > 0].sum(keepdims=True)).round(0)

@cached
def get_true_negatives(npz_file):
    tp = get_true_positives(npz_file)
    fp = get_false_positives(npz_file)
    fn = get_false_negatives(npz_file)
    tn = get_total_samples(npz_file) - tp - fp - fn
    return tn

@cached
def get_false_discovery_rate(npz_file):
    tp = get_true_positives(npz_file)
    fp = get_false_positives(npz_file)
    return fp / (tp + fp)

@cached
def get_false_negative_rate(npz_file):
    tp = get_true_positives(npz_file)
    fn = get_false_negatives(npz_file)
    return fn / (tp + fn)

@cached
def get_fowlkes_mallows_index(npz_file):
    ppv = 1 - get_false_discovery_rate(npz_file)
    tpr = 1 - get_false_negative_rate(npz_file)
    return np.sqrt(ppv * tpr)



