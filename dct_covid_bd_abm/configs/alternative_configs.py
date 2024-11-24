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

from dct_covid_bd_abm.simulator.pathogen_utils.infectious_profile_functions import inf_func, lognormal_infectious_function, \
    lognormal_with_delay_infectious_function, triangular_infectious_function, \
    triangular_infectious_function_with_delay, triangular_infectious_function_so_in_mean

alt_configs = [
    dict(
        name="inf_func",
        sim_engine=dict(
            pathogen={"infectiousness_function": inf_func})),
    dict(
        name="lognormal_inf_func",
        sim_engine=dict(
            pathogen={"infectiousness_function": lognormal_infectious_function})),
    dict(
        name="lognormal_inf_func_with_delay",
        sim_engine=dict(
            pathogen={"infectiousness_function": lognormal_with_delay_infectious_function})),
    dict(
        name="triangular_inf_func",
        sim_engine=dict(
            pathogen={"infectiousness_function": triangular_infectious_function})),
    dict(
        name="triangular_inf_func_with_delay",
        sim_engine=dict(
            pathogen={"infectiousness_function": triangular_infectious_function_with_delay})
    ),
    dict(
        name="triangular_infectious_function_so_in_mean",
        sim_engine=dict(
            pathogen={"infectiousness_function": triangular_infectious_function_so_in_mean})
    )
]
