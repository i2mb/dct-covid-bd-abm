
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

from collections import OrderedDict

from i2mb.engine.configuration import Configuration


def import_alternatives(config_file_name, config=None):
    catch_last_value = OrderedDict()
    with open(config_file_name) as cfg_file:
        code = cfg_file.read()
        if config is None:
            config = Configuration(population_size=1000)

        exec(code, {"config": config}, catch_last_value)

    configuration_list = []
    for conf_idx, conf in enumerate(catch_last_value["configurations"]):
        conf["alt_file_name"] = config_file_name
        conf["alt_conf_idx"] = conf_idx
        configuration_list.append(conf)

    return catch_last_value["configurations"]


def apply_alternate_config(exp_config, alt_config):
    for parameter, value in alt_config.items():
        if parameter == "population_size":
            raise RuntimeError("population_size should not be changed in alternative configurations. Rather, use the "
                               "command line parameter --num-agents or create the config class accordingly.")

        if isinstance(value, dict):
            default_value = {}
            parameter_value = exp_config.setdefault(parameter, default_value)
            if parameter_value is default_value:
                parameter_value.update(value)
                continue
            else:
                apply_alternate_config(parameter_value, value)

        else:
            exp_config[parameter] = value
