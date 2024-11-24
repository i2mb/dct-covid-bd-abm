from functools import partial

import numpy as np

from dct_covid_bd_abm.simulator.analysis_utils.pathogen import get_isolation_f_metric, \
    get_isolation_fowlkes_mallows_index
from dct_covid_bd_abm.simulator.analysis_utils.pathogen import get_serial_intervals, get_generation_intervals, \
    get_incubation_periods, get_illness_durations, get_days_in_isolation, get_days_in_quarantine, get_wave_duration, \
    get_affected_population, get_infection_incidence, get_hospitalization_incidence, \
    get_external_contact_generation_intervals, get_isolation_false_discovery_rate, get_isolation_false_negative_rate, \
    get_isolation_f_metric, get_isolation_sensitivity, get_isolation_ppv
from dct_covid_bd_abm.simulator.analysis_utils.system_performance import get_false_negative_rate, \
    get_false_discovery_rate, get_fowlkes_mallows_index

display_variables = {"Serial Interval": get_serial_intervals,
                     "Generation Interval": get_generation_intervals,
                     "Incubation Period": get_incubation_periods,
                     "Illness Duration": get_illness_durations,
                     "Days in Isolation": get_days_in_isolation,
                     "Days in Quarantine": get_days_in_quarantine,
                     "Wave Duration": get_wave_duration,
                     "Total infected": get_affected_population,
                     "7-day Incidence": get_infection_incidence,
                     "7-day Hosp. Incidence": get_hospitalization_incidence,
                     "External contacts GI": get_external_contact_generation_intervals,
                     "EVT False discovery rate": get_isolation_false_discovery_rate,
                     "EVT False negative rate": get_isolation_false_negative_rate,
                     "EVT Fowlkes mallows index": get_isolation_fowlkes_mallows_index,
                     # "EVT F1": partial(get_isolation_f_metric, beta=1),
                     # "EVT F0.5": partial(get_isolation_f_metric, beta=0.5),
                     # "EVT Fsqrt(2)": partial(get_isolation_f_metric, beta=np.sqrt(2)),
                     # "EVT F2": partial(get_isolation_f_metric, beta=2),
                     # "EVT Sensitivity": get_isolation_sensitivity,
                     # "EVT PPV": get_isolation_ppv,
                     "False negative rate": get_false_negative_rate,
                     "False discovery rate": get_false_discovery_rate,
                     "Fowlkes Mallows index": get_fowlkes_mallows_index
                     }
