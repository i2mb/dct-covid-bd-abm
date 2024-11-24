#!/usr/bin/env python3
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

import time
from argparse import ArgumentParser, ArgumentTypeError
from datetime import timedelta
from multiprocessing import Pool, cpu_count, set_start_method
from pprint import pprint

from i2mb.engine.configuration import Configuration
from dct_covid_bd_abm.simulator.utilities.config_utilitites import apply_alternate_config, import_alternatives
from dct_covid_bd_abm.simulator.experiments import DCTvMCTExperiment


def run_experiment(run_id, args, config):
    import gc
    import numpy as np

    gc.enable()
    np.random.seed()
    config = prepare_configuration(args, config)
    start_time = time.time()
    p_start_time = time.process_time_ns()
    experiment = DCTvMCTExperiment(run_id, config)
    if experiment.sim_engine is not None:
        experiment.run_sim_engine()

    end_time = time.time()
    p_end_time = time.process_time_ns()
    run_time = timedelta((end_time - start_time) / (3600 * 24))
    p_run_time = timedelta((p_end_time - p_start_time) / (3600e9 * 24))
    print(f"Runtime for configuration '{experiment.config_name}.{experiment.get_base_name()}' - "
          f"Wall time: {run_time}, Process time: {p_run_time} ")

    gc.collect()
    return None


def restricted_float(x):
    """From chpner@stackoverflow https://stackoverflow.com/a/12117065"""
    try:
        x = float(x)
    except ValueError:
        raise ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


def create_argument_parser(experiments_, scenarios_, config_names_):

    experiment_config = Configuration(config_file="dct_covid_bd_abm/configs/base_configuration.py")
    parser = ArgumentParser(description='Executes different experiments using the COVID-19 engine.')

    # We add config argument again to get the help message
    parser.add_argument("--configs", type=import_alternatives,
                        default="./dct_covid_bd_abm/configs/experiment_setup.py",
                        help="Python file containing either a dict or a list of dicts with configuration "
                             "parameters. If alternative-configs is a dict, it will override the base "
                             "configuration parameters and store results in data_dir. With a list, "
                             "a sub-folder in data_dir is created for each dictionary in the list. "
                             "Sub-folders are created using  the name and experiment_name attributes in the "
                             "dict, e.g., ´data_dir/name/experiment_name´ .")

    parser.add_argument("--start", default=0, type=int, help="Start run number.")
    parser.add_argument("--num-runs", default=250, type=int, help="Number of simulation runs.")
    parser.add_argument("--dry-run", action="store_true", help="Set the experiments up, but don't run them.")
    parser.add_argument("--steps", default=experiment_config["sim_steps"], type=int, dest="sim_steps",
                        help="Number of simulation steps per run, default (-1) until the end of the wave.")
    parser.add_argument("-j", "--num-cpus", type=int, default=cpu_count() * 35 // 40,
                        help="Number of CPU cores to use for parallel processing.")
    parser.add_argument("-e", "--experiment", action="append", dest="experiments",
                        choices=experiments_,
                        help="Specify which experiments to run. You can specify multiple -e options.")
    parser.add_argument("-c", "--config_name", action="append", dest="config_names",
                        choices=config_names_,
                        help="Specify configuration files containing experiment definitions. You can specify multiple -c options.")
    parser.add_argument("-o", "--overwrite_files", action="store_true", help="Overwrite output files")
    parser.add_argument("-S", "--scenario", action="append", dest="scenarios",
                        choices=scenarios_,
                        help="Specify which scenarios to run. Multiple -S options are allowed.")
    parser.add_argument("-d", "--data-dir", type=str, default=experiment_config["data_dir"],
                        help=f"Data directory where to read input files, and store results. Default: "
                             f"{experiment_config['data_dir']}")
    parser.add_argument("--num-agents", type=int, default=experiment_config["population_size"], help="Population size")
    parser.add_argument("--num-patients0", type=int, default=experiment_config["num_patient0"],
                        help="Number if infectious individual to insert to start the epidemic")



    return parser


def load_configurations():
    parser_cfg_file = ArgumentParser(description='Executes different experiments using the COVID-19 engine.',
                                     add_help=False)
    parser_cfg_file.add_argument("--configs", type=import_alternatives,
                                 default="./dct_covid_bd_abm/configs/experiment_setup.py")

    config_args, additional_parameters = parser_cfg_file.parse_known_args()
    return config_args.configs, additional_parameters


def extract_scenarios(configs):
    scenarios = set()
    experiments = set()
    config_names = set()
    for conf in configs:
        scenarios.add(conf.get("scenario", {}).get("name", "default"))
        experiments.add(conf.get("experiment_name", "test"))
        config_names.add(conf.get("name", "test_config"))

    return sorted(experiments), sorted(scenarios), sorted(config_names)


def select_configs(configs, experiments, scenarios, config_names):
    configs_ = []
    for conf in configs:
        s = conf.get("scenario", {}).get("name", "default")
        e = conf.get("experiment_name", "test")
        c_name = conf.get("name", "test_config")
        if s in scenarios and e in experiments and c_name in config_names:
            configs_.append(conf)

    return configs_


def prepare_configuration(args, config_):
    config = Configuration(population_size=args.num_agents, config_file="dct_covid_bd_abm/configs/base_configuration.py")

    # Reload alternative configuration with updated population size
    config_ = import_alternatives(config_["alt_file_name"], config=config)[config_["alt_conf_idx"]]
    scenario_parameters = config_.pop("scenario", {})
    engine_parameters = scenario_parameters.pop("engine_props", {})
    engine_parameters.update(config_.pop("sim_engine", {}))
    config.update(config_)
    if args.sim_steps is not None:
        config["sim_steps"] = args.sim_steps

    apply_alternate_config(config["sim_engine"], engine_parameters)
    apply_alternate_config(config["scenario"], scenario_parameters)
    config["data_dir"] = args.data_dir
    config["num_patient0"] = args.num_patients0
    config["overwrite_files"] = args.overwrite_files
    return config


def process_dry_run(args, configs):
    configs_ = []
    for conf in configs:
        if args.dry_run:
            exp_config = prepare_configuration(args, conf)
            configs_.append(exp_config)
            print(f"Name: '{exp_config.get('name')}', \n Data Directory: '{exp_config.get('data_dir')}'")
            print("Complete Config")
            pprint(exp_config)
            answer = input("Continue? [Y/n]: ")
            if answer.lower() == "y" or answer == "":
                continue
            else:
                exit()

    return configs_


def process_experiments(args, configs):
    process_dry_run(args, configs)
    if args.dry_run:
        exit()

    res = []
    try:
        set_start_method('spawn')
        with Pool(args.num_cpus, maxtasksperchild=1) as pool:
            for run in range(args.start, args.start + args.num_runs):
                for exp_config in configs:
                    res.append(pool.apply_async(run_experiment, (run, args, exp_config)))

            for r in res:
                r.get()

    finally:
        # clean up
        # shutil.rmtree("/tmp/sct")
        pass


def main():
    configs, args_ = load_configurations()
    experiments, scenarios, conf_names = extract_scenarios(configs)
    parser = create_argument_parser(experiments, scenarios, conf_names)
    args = parser.parse_args(args_)
    experiments = load_experiments(args, experiments)
    scenarios = load_scenarios(args, scenarios)
    config_names = load_config_names(args, conf_names)
    configs = select_configs(configs, experiments, scenarios, config_names)
    process_experiments(args, configs)


def load_scenarios(args, scenarios):
    scenarios_ = []
    if args.scenarios:
        scenarios_ = args.scenarios
    else:
        scenarios_.extend(scenarios)
    return scenarios_


def load_experiments(args, experiments):
    experiments_ = []
    if args.experiments:
        experiments_ = args.experiments
    else:
        experiments_.extend(experiments)

    return experiments_


def load_config_names(args, config_names):
    experiments_ = []
    if args.config_names:
        experiments_ = args.config_names
    else:
        experiments_.extend(config_names)

    return experiments_


if __name__ == "__main__":
    main()
