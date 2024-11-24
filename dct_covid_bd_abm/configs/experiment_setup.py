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

from itertools import product

from dct_covid_bd_abm.simulator.scenarios.scenario_complex_world import ApartmentsScenario

available_experiments = {}
for x in product(["both", "DCT", "MCT"], [True, False], [True, False]):
    e__ = dict((i for i in zip(["trace_contacts", "test_to_exit", "quarantine_households"], x)))
    e__label = f"{e__['trace_contacts'].lower()}_{e__['test_to_exit'] and 'rnt' or 'nrnt'}_" \
               f"{e__['quarantine_households'] and 'chq' or 'nchq'}"

    available_experiments[e__label] = dict(intervene=True, **e__)


available_experiments["no_intervention"] = dict(intervene=False, trace_contacts=None, test_to_exit=False,
                                                quarantine_household=False)
available_experiments["personal_intervention"] = dict(intervene=True, trace_contacts=None, test_to_exit=False,
                                                      quarantine_households=False)

available_scenarios = {"ho": {"class": ApartmentsScenario,
                              "name": "ho",
                              "kwargs": {},
                              "engine_props": dict(hospitalize=True, night_out=False, use_buses=True)},
                       "cw_ohb": {"class": ApartmentsScenario,
                                  "kwargs": {},
                                  "name": "cw_ohb",
                                  "engine_props": dict(hospitalize=True, night_out=True, use_buses=True)}
                       }

scenarios = list(available_scenarios.keys())
experiments = list(available_experiments.keys())

# For a lack of a cleaner way of doing this. The list MUST be called configurations
configurations = []
for e in experiments:
    for s in scenarios:
        config = dict()
        config["experiment_name"] = e
        config["scenario"] = available_scenarios[s]
        config["sim_engine"] = available_experiments[e]
        configurations.append(config)

