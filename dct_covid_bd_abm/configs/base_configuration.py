#  dct-covid-bd-abm
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

from functools import partial

import numpy as np
from scipy.stats import lognorm, truncnorm

from i2mb.pathogen import SymptomLevels
from i2mb.pathogen.dynamic_infection import RegionVirusDynamicExposureBaseOnViralLoad

# Global configuration
# Import and configure global time before importing any library module-
from i2mb.utils import global_time
from i2mb.worlds.world_base import PublicSpace
from dct_covid_bd_abm.simulator.utilities.math_utilities import estimate_lognormal_parameters
from dct_covid_bd_abm.simulator.pathogen_utils import kissler_model
from dct_covid_bd_abm.simulator.pathogen_utils.exposure_functions import exposure_function
from dct_covid_bd_abm.simulator.pathogen_utils.recovery_functions import recovery_function
from dct_covid_bd_abm.simulator.scenarios.scenario_complex_world import Office, ApartmentsScenario

if __name__ != "builtins":
    """If the base configuration is imported as a module, then we create config instance. If instead, 
    base configuration is executed as part of the configuration creation. We use the instance created externally. """
    from i2mb.engine.configuration import Configuration

    config = Configuration()

# Set time step to 5 minutes
config.ticks_hour = 60 // 5

# Set default population size to 1000
if config.population_size is None:
    config.population_size = 1000


def delay_test_taking():
    samples = np.random.gamma(6, scale=0.45, size=config.population_size) + 2
    return global_time.make_time(day=samples).astype(int).reshape(-1, 1)


# Core configuration
config.update(dict(
    time_factor_ticks_day=config.time_factor_ticks_day,  # samples per days
    ws=8,  # Starting window size in days for time series plots in videos
    dropout=.2,
    sim_steps=None,
    num_patient0=5,
    data_dir="data",
    videos_dir="videos",
    save_files=True,
    overwrite_files=False,
    save_videos=False,
    end_with_wave=True,
    insert_pathogen=1,
    skip_incubation=False,
    symptoms_level=SymptomLevels.mild,
    generate_time_series=False,

    # Locations of interest where to do measurements
    locations=["home", "office", "bar", "restaurant", "bus", "hospital"]
))

# Simulation Engine configuration
config["sim_engine"] = dict(
    hospitalize=True,
    night_out=True,
    intervene=True,
    test_to_exit=True,
    use_buses=True,
    trace_contacts=True,
    quarantine_households=True,
    use_pathogen=True,
    use_contact_history=False,
    dropout=0,
    sleep=True,
    engine={}
)

# Engine modules configurations
# # Base Motion
config["sim_engine"]["motion"] = dict(
    step_size=0.2,
    gravity=None
)

# # Pathogen
# ## Disease duration distribution
# ## Li, Q., et al. (2020). Early Transmission Dynamics in Wuhan, China, of Novel Coronavirus–Infected Pneumonia. New
# ## England Journal of Medicine.
# ## Reported values: Mean = 5.2 CI(95%) 4.1 to 7.0, N=10, 95th percentile = 12.5
s, mu = estimate_lognormal_parameters(percentile=97.5, percentile_value=11.5 * config.time_factor_ticks_day,
                                      mean=5.2 * config.time_factor_ticks_day)
incubation_duration_distribution = lognorm(s=s, loc=0, scale=np.exp(mu)).rvs

# ## Kampen, et al. (2020). Shedding of infectious virus in hospitalized patients with coronavirus disease-2019 (
# ## COVID-19): duration and key determinants,
s, mu = estimate_lognormal_parameters(percentile=75, percentile_value=21 * config.time_factor_ticks_day,
                                      median=18 * config.time_factor_ticks_day)
illness_duration_distribution = lognorm(s=s, loc=0, scale=np.exp(mu)).rvs

config["sim_engine"]["pathogen"] = dict(
    pathogen_class=RegionVirusDynamicExposureBaseOnViralLoad,
    exposure_function=exposure_function,
    recovery_function=recovery_function,
    infectiousness_function=kissler_model.triangular_viral_load,
    symptom_distribution=[0.4, 0.4, .138, .062],
    death_rate=[0.02, 0.05],
    icu_beds=config.population_size * 0.03,  # TODO move to scenario - Replace by available care
    # Function that changes the death rate based on care availability.
    # death_rate_function=lambda x: False,
    clearance_duration_distribution=kissler_model.clearance_period,
    proliferation_duration_distribution=kissler_model.proliferation_period,
    symptom_onset_estimator=kissler_model.compute_symptom_onset,
    max_viral_load_distribution=kissler_model.maximal_viral_load,
    max_viral_load=kissler_model.log_rna(0),
    min_viral_load=kissler_model.log_rna(40),
)

# # Intervention modules
config["sim_engine"]["ct_intervention"] = dict(
    freeze_isolated=False
)
config["sim_engine"]["get_tested"] = dict(
    delay=1.5 * config.time_factor_ticks_day,
    delay_test=delay_test_taking,
    quarantine_duration=14 * config.time_factor_ticks_day,
    share_test_result=1.
)
config["sim_engine"]["tester"] = dict(
    duration=1 * config.time_factor_ticks_day,
    test_method=None,
    closing_time=None, opening_time=None  # Defaults to 8:00 to 16:30
)

# # Contact tracing modules
config["sim_engine"]["fnf_tracing"] = dict(
    radius=3,
    track_time=14 * config.time_factor_ticks_day
)
config["sim_engine"]["contact_history"] = dict(
    radius=3,
)
config["sim_engine"]["dc_tracing"] = dict(
    radius=3, duration=2,
    track_time=14 * config.time_factor_ticks_day,
    coverage=.25,
    dropout=0.02,
    false_positives=0,
    false_negatives=0
)
config["sim_engine"]["mc_tracing"] = dict(
    radius=3, processing_duration=3 * config.time_factor_ticks_day,
    track_time=14 * config.time_factor_ticks_day,
    recall={PublicSpace: 0.3, ApartmentsScenario.Home: .98, Office: .5},
    queue_length=max([config.population_size // 1000, 1]),
    dropout=0.05
)

# # City services modules
config["sim_engine"]["ambulance"] = dict(
    symptom_level=SymptomLevels.strong
)


# # Behaviours
def select_bus_ride_duration():
    return np.random.choice([global_time.make_time(minutes=15),
                             global_time.make_time(minutes=30),
                             global_time.make_time(minutes=45)])


def select_car_ride_duration():
    return np.random.choice([global_time.make_time(minutes=10),
                             global_time.make_time(minutes=15),
                             global_time.make_time(minutes=20),
                             global_time.make_time(minutes=30),
                             global_time.make_time(minutes=35)])


def noisy_time(low, high):
    def __callable():
        return np.random.randint(low, high)

    return __callable


config["sim_engine"]["daily_schedule"] = dict(
    full_time_office_schedule=[{"repeats": "daily",
                               "start_time": noisy_time(global_time.make_time(hour=7, minutes=15),
                                                        global_time.make_time(hour=8, minutes=30)),
                               "duration": select_car_ride_duration,
                               "event_location": "car",
                               "return_to": "office",
                               "auto_return": True},
                              {"repeats": "daily",
                               "start_time": noisy_time(global_time.make_time(hour=15, minutes=-5),
                                                        global_time.make_time(hour=17, minutes=20)),
                               "duration": select_car_ride_duration,
                               "event_location": "car",
                               "return_to": "home",
                               "auto_return": True
                               }],
    part_time_office_schedule=[{"repeats": "daily",
                               "start_time": noisy_time(global_time.make_time(hour=7, minutes=15),
                                                        global_time.make_time(hour=8, minutes=30)),
                               "duration": select_car_ride_duration,
                               "event_location": "car",
                               "return_to": "office",
                               "auto_return": True},
                              {"repeats": "daily",
                               "start_time": noisy_time(global_time.make_time(hour=9, minutes=-5),
                                                        global_time.make_time(hour=14, minutes=20)),
                               "duration": select_car_ride_duration,
                               "event_location": "car",
                               "return_to": "home",
                               "auto_return": True
                               }],
    commuters_schedules={"groups": [{"go_work": noisy_time(global_time.make_time(hour=6, minutes=45),
                                                           global_time.make_time(hour=7, minutes=10)),
                                     "go_home": noisy_time(global_time.make_time(hour=16, minutes=40),
                                                           global_time.make_time(hour=17, minutes=15)),
                                     "proportion": 0.5},
                                    {"go_work": noisy_time(global_time.make_time(hour=7, minutes=20),
                                                           global_time.make_time(hour=7, minutes=40)),
                                     "go_home": noisy_time(global_time.make_time(hour=17),
                                                           global_time.make_time(hour=17, minutes=30)),
                                     "proportion": 0.5}],
                         "repeats": "daily",
                         # Event location is a function that selects the bus the agent will use, receives as
                         # argument the complete list of buses.
                         "event_location": np.random.choice,
                         "return_to": "office",
                         "duration": select_bus_ride_duration,
                         "auto_return": True},

    # 0.12 is an estimation based on VGN 2020 (Nuremberg public transportation) numbers
    pt_commuters=np.random.choice(list(range(config.population_size)), int(0.12 * config.population_size),
                                  replace=False),

    # 43% https://data.worldbank.org/indicator/SL.TLF.PART.ZS?locations=DE
    part_time_workers=.43,
    goes_to_work=np.ones(config.population_size, dtype=bool),
    decides_to_stay_home=None
)


def arrival():
    # Save a few ms by precomputing the parameters for the distribution call
    myclip_a, myclip_b = 17, 24
    loc = 19
    scale = 4
    a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale

    def _arrival():
        _hour = truncnorm.rvs(a=a, b=b, loc=loc, scale=scale)
        return global_time.make_time(hour=_hour).astype(int)

    return _arrival


def duration():
    # Save a few ms by precomputing the parameters for the distribution call
    myclip_a, myclip_b = 0, 9
    loc = 0
    scale = 1.2
    a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale

    def _duration(size):
        _hour = truncnorm.rvs(a=a, b=b, loc=loc, scale=scale, size=size)
        return global_time.make_time(hour=_hour).astype(int)

    return _duration


config["sim_engine"]["night_out_module"] = dict(
    group_location=["home", "office"],
    arrival=arrival(),
    duration=duration(),
    min_capacity=0.5,
    opening_hours=global_time.make_time(hour=17),
    closing_hours=global_time.make_time(hour=2)
)

config["sim_engine"]["sleep_model"] = dict(
    # Distributions for sleep duration and mid-points
    # Wahl, F., and Amft, O. (2018). Data and Expert Models for Sleep Timing and Chronotype Estimation from
    # Smartphone Context Data and Simulations. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 2, 139:1–139:28.
    sleep_duration=partial(np.random.normal, global_time.make_time(hour=8), global_time.make_time(hour=1)),
    sleep_midpoint=partial(np.random.normal, global_time.make_time(hour=1), global_time.make_time(hour=1))
)

# Scenario confiuration
config["scenario"] = {"kwargs": dict(
    # https://www.statista.com/statistics/464187/households-by-size-germany/ => Average size of a German household 2
    # (2019)
    homes=config.population_size // 2,
    offices=config.population_size // 20,

    # 0.12 is an estimation based on VGN 2020 (Nuremberg public transportation) numbers
    # we want to fill all buses to 80% sitting capacity on average.
    buses=int(np.ceil(config.population_size * 0.12 / (35 * 0.8))),

    # https://www.statista.com/statistics/464187/households-by-size-germany/ 2019
    home_percentages=[0.42298889, 0.33201629, 0.11930518, 0.09114125, 0.03454839],
    name="apartment_homes"
)}
