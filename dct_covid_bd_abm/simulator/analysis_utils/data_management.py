import os
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from dct_covid_bd_abm.simulator.activity_utils.extrasensory import load_extrasensory_dataset, map_extrasensory_to_i2mb, \
    convert_to_period_representation

from dct_covid_bd_abm.simulator.analysis_utils import display_variables
from dct_covid_bd_abm.simulator.utilities.file_utilities import load_data_description


def load_experiments(data_dir_):
    if len(glob(f"{data_dir_}/*/*")) == 0:
        raise RuntimeError(f"{data_dir_} does not contain any experiments")

    return [d for d in glob(f"{data_dir_}/*/*")]


def tokenize_experiment(experiment):
    tokens = Path(experiment).parts
    c = tokens[-3]
    e = tokens[-2]
    s = tokens[-1]

    return c, e, s


def get_run_number(file_name):
    return int(file_name[:-4][-4:])


def get_variable_from_experiment_file(experiment_path, variable_file_processor):
    npz_files = glob(f"{experiment_path}/*.npz")
    if not npz_files:
        raise RuntimeError(f"{experiment_path} does not contain any experiment npz files")

    variable_values = []
    for npz_file in npz_files:
        run = get_run_number(npz_file)
        variable_values.append((run, variable_file_processor(npz_file)))

    return variable_values


__cache_dirs__ = {}


def expand_experiment_names(names, remove_words):
    new_names = []
    for c_name in names:
        c_name = expand_experiment_name(c_name, remove_words)

        new_names.append(c_name)

    return new_names


def expand_experiment_name(name, remove_words):
    for r_word in remove_words:
        name = name.replace(r_word, "")

    return name.replace("_", " ").title()


def get_variable_from_experiments(data_dir_, variable_file_processor):
    if (data_dir_, id(variable_file_processor)) in __cache_dirs__:
        return __cache_dirs__[data_dir_, id(variable_file_processor)]

    experiments = load_experiments(data_dir_)
    exp_variable = {}
    for experiment in experiments:
        key = tuple(tokenize_experiment(experiment))
        serial_intervals = get_variable_from_experiment_file(experiment, variable_file_processor)
        for run, s_i in serial_intervals:
            key_ = tuple(list(key) + [run])
            exp_variable[key_] = s_i

    __cache_dirs__[data_dir_, id(variable_file_processor)] = exp_variable
    return exp_variable


def dict2df(dict_, replace_infs=False, fill_value: int|float =0, drop_levels=None, remove_words=None):
    df = parse_dictionary(dict_)
    if replace_infs:
        try:
            df[np.isinf(df)] = fill_value
        except TypeError:
            df = df.astype(float)
            df[np.isinf(df)] = fill_value

    names = expand_experiment_names(df.columns.levels[1], remove_words)
    df.columns = df.columns.set_levels(levels=[names], level=[1])

    if drop_levels is None:
        df.columns = df.columns.droplevel([0, 2])

    return df


def parse_dictionary(dict_):
    dfs = []
    for key, value in dict_.items():
        index = None
        if type(value) is pd.DataFrame:
            columns = pd.MultiIndex.from_product([*[[c] for c in key], value.columns])
            index = value.index
            value = value.values

        elif type(value) is pd.Series:
            columns = pd.MultiIndex.from_tuples([(*key, value.name)])
            index = value.index
            value = value.values
        else:
            columns = pd.MultiIndex.from_tuples([key])

        df = pd.DataFrame(value, columns=columns, index=index)
        dfs.append(df)

    return pd.concat(dfs, axis=1)


def load_pathogen_data(data_dir):
    from dct_covid_bd_abm.simulator.analysis_utils.pathogen import (get_hospitalization_incidence,
                                                                    get_affected_population, get_infection_incidence)
    from dct_covid_bd_abm.simulator.analysis_utils import display_variables

    baseline_variables = []
    variable_getters = display_variables.values()
    for variable_getter in variable_getters:
        data_dict = get_variable_from_experiments(data_dir, variable_getter)
        data_frame = dict2df(data_dict, replace_infs=True, fill_value=np.nan, remove_words=["validation_"])
        data_frame.sort_index(axis=1, inplace=True)

        if variable_getter == get_hospitalization_incidence:
            data_frame = data_frame.loc[:, (slice(None), slice(None), "Hospitalised")]
            data_frame = data_frame.max().unstack(0).fillna(0)

        elif variable_getter == get_infection_incidence:
            data_frame = data_frame.max().unstack(0).fillna(0)

        elif variable_getter == get_affected_population:
            data_frame = data_frame.sum().unstack(0).fillna(0)

        else:
            data_frame = data_frame.stack(1, future_stack=True)
            if data_frame.columns.nlevels > 1:
                data_frame = data_frame.droplevel(1, axis=1)

        baseline_variables.append(data_frame.reset_index(drop=True))

    return baseline_variables


def load_reference_data(refs, stage_data, radar_plot_properties):
    refs_data = None
    if refs is not None:
        refs_data = []
        for rd in refs:
            if rd is None:
                continue

            selector = radar_plot_properties[rd].get("select_dp", slice(None))
            if type(selector) is str:
                selector = [selector]

            refs_data.append(stage_data[rd].loc[selector])
    return refs_data


def load_stage_data(base_directory, radar_plot_properties, break_up_index_into_columns=False, prefix="",
                    only_means=True, overwrite_cache=False):
    stage_data = {}
    variable_names = list(display_variables.keys())
    for test, test_prop in radar_plot_properties.items():
        csv_file = os.path.join(base_directory, f"{prefix}_{test}.csv")
        if not os.path.exists(csv_file) or overwrite_cache:
            __create_stage_data_csv_file__(base_directory, break_up_index_into_columns, csv_file, test_prop,
                                           variable_names)

        data_frame = __read_stage_csv_file__(csv_file)
        if only_means:
            stage_data[test] = data_frame.loc[:, (slice(None), "mean")].droplevel(1, axis=1)
        else:
            stage_data[test] = data_frame

    return stage_data


def load_stage_raw_data(base_directory, radar_plot_properties, break_up_index_into_columns=False, prefix="",
                        overwrite_cache=False):
    stage_data = {}
    variable_names = list(display_variables.keys())
    for test, test_prop in radar_plot_properties.items():
        csv_file = os.path.join(base_directory, f"{prefix}_{test}_raw.csv")

        if not os.path.exists(csv_file) or overwrite_cache:
            __create_stage_data_csv_file__(base_directory, break_up_index_into_columns, csv_file, test_prop,
                                           variable_names, raw=True)

        data_frame = __read_stage_csv_file__(csv_file, raw=True)
        stage_data[test] = data_frame

    return stage_data


def __create_stage_data_csv_file__(base_directory, break_up_index_into_columns, csv_file, test_prop, variable_names,
                                   raw=False):
    def choice_nan(x, *args, **kwargs):
        size = kwargs.get('size', 1)
        if size >= len(x):
            return x

        return np.random.choice(x[~x.isnull()].values, *args, **kwargs)

    data_dir = test_prop["data_dir"]
    if type(data_dir) is str:
        data_dir = [data_dir]

    data_filter = test_prop.get("data_filters", None)
    if data_filter is None:
        data_filter = [data_filter] * len(data_dir)
    elif type(data_filter) is str:
        data_filter = [data_filter] * len(data_dir)

    rename_index = test_prop.get("rename_index", None)
    if rename_index is None:
        rename_index = [rename_index] * len(data_dir)
    elif type(data_filter) is str:
        rename_index = [rename_index] * len(data_dir)

    test_dfs = []
    for data_dir_, data_filter_, rename_index_ in zip(data_dir, data_filter, rename_index):
        data = load_pathogen_data(os.path.join(base_directory, data_dir_))

        if raw:
            data_median = pd.concat([d.apply(choice_nan, axis=0, size=50, replace=False) for d in data], keys=variable_names[:])
        else:
            data_median = load_data_description(data, variable_names, os.path.join(base_directory, data_dir_))

        if rename_index_ is not None:
            rename_index_selector, new_labels = rename_index_
            if raw:
                data_median.columns.values[rename_index_selector] = new_labels
            else:
                data_median.index.values[rename_index_selector] = new_labels

        if break_up_index_into_columns:

            if raw:
                index = pd.MultiIndex.from_tuples(
                    [re.search("((?:No |)*Ict|Both|Dct|Mct|None)* ((?:No |)*Rnt)* ((?:No |)*Qch)* ((?:No |)*Bnr)*",
                               c).groups() for c in data_median.columns],
                    names=["DCT", "RNT", "QCH", "CBR"])
                data_median.columns = index

            else:
                index = pd.MultiIndex.from_tuples(
                    [re.search("((?:No |)*Ict|Both|Dct|Mct|None)* ((?:No |)*Rnt)* ((?:No |)*Qch)* ((?:No |)*Bnr)*",
                               c).groups() for c in data_median.index],
                    names=["DCT", "RNT", "QCH", "CBR"])
                data_median.index = index

        if data_filter_ is not None:
            data_median = data_median.loc[data_filter_] if not raw else data_median.loc[:, data_filter_]

        data_median = (data_median.rename(index={"Mct": "No Dct", "Bnr": "No Cbr", "No Bnr": "Cbr", "Both": "Dct"})
                       if not raw else
                       data_median.rename(columns={"Mct": "No Dct", "Bnr": "No Cbr", "No Bnr": "Cbr", "Both": "Dct"}))

        test_dfs.append(data_median)

    if raw:
        test_data = pd.concat(test_dfs, axis=1)
    else:
        test_data = pd.concat(test_dfs)

    test_data.to_csv(csv_file)
    return


def __read_stage_csv_file__(csv_file, raw=False):
    print(f"reading file {csv_file}")
    if "stage_1" in csv_file:
        if raw:
            data_frame = pd.read_csv(csv_file, index_col=[0, 1], header=[0, 1, 2, 3])

        else:
            data_frame = pd.read_csv(csv_file, index_col=[0], header=[0, 1])

    else:
        if raw:
            data_frame = pd.read_csv(csv_file, index_col=[0, 1], header=[0, 1, 2, 3],
                                     # keep_default_na=False, na_values=[" "]
                                     )
            data_frame = data_frame.rename(columns={"Mct": "No Dct", "Lbr": "Cbr", "No Lnr": "No Cbr", "Both": "Dct"})

        else:
            data_frame = pd.read_csv(csv_file, index_col=[0, 1, 2, 3], header=[0, 1],
                                     # keep_default_na=False, na_values=[" "]
                                     )
            data_frame = data_frame.rename(index={"Mct": "No Dct", "Lbr": "Cbr", "No Lnr": "No Cbr", "Both": "Dct"})

    return data_frame


def load_validation_data(base_directory):
    validation_data = {}
    variables = ["Serial Interval", "Generation Interval", "Incubation Period", "Illness Duration"]
    cache_path = f"{base_directory}/validation_data.hdf"
    if os.path.exists(cache_path):
        print(f"Loading validation data form cache {cache_path}")
        validation_data.update({k: pd.read_hdf(cache_path, key=k.replace(" ", "_")) for k in variables})
        return validation_data

    for var in variables:
        data_dict = get_variable_from_experiments(base_directory, display_variables[var])
        data_df = dict2df(data_dict, replace_infs=True, fill_value=np.nan, remove_words=["validation_"])
        data_df = data_df.stack(1)
        validation_data[var] = data_df
        data_df.to_hdf(cache_path, key=var.replace(" ", "_"))

    return validation_data


def load_activity_validation_data(i2mb_data_directory, es_data_directory):
    i2mb_data = pd.read_feather(f"{i2mb_data_directory}/activity_history.feather")
    i2mb_data = i2mb_data[i2mb_data["activity"] != "ActivityNone"]
    i2mb_activities = set(i2mb_data["activity"])

    data_es = load_extrasensory_dataset(es_data_directory).fillna(0)
    es_activities = data_es.columns

    data_es = map_extrasensory_to_i2mb(data_es)

    # Convert time series to period representation
    data_es_periods = convert_to_period_representation(data_es, i2mb_activities)
    data_es_periods["duration"] = (data_es_periods["duration"] / 5).round() * 5
    data_es_periods = data_es_periods[data_es_periods["duration"] > 0]
    data_es_periods.replace(["EatAtRestaurant", "EatAtBar"], ["EatOut", "EatOut"], inplace=True)
    i2mb_data.replace(["EatAtRestaurant", "EatAtBar"], ["EatOut", "EatOut"], inplace=True)
    data_es_periods.replace(["Rest"], ["OtherHomeAct."], inplace=True)
    i2mb_data.replace(["Rest"], ["OtherHomeAct."], inplace=True)
    i2mb_activities = set(i2mb_data["activity"])

    i2mb_truncate_24 = i2mb_data["duration"] > 60 * 24
    es_periods_truncate_24 = data_es_periods["duration"] > 60 * 24

    i2mb_data.loc[i2mb_truncate_24, "duration"] = 60 * 24
    data_es_periods.loc[es_periods_truncate_24, "duration"] = 60 * 24

    i2mb_data["activity"] = i2mb_data["activity"].astype("category")
    data_es_periods["activity"] = data_es_periods["activity"].astype("category")
    return i2mb_data, data_es_periods


def load_contact_validation_data(i2mb_contact_validation_dir):
    feather_name = f"{i2mb_contact_validation_dir}/contact_data_unique_complete.feather"
    if not os.path.exists(feather_name):
        from dct_covid_bd_abm.simulator.analysis_utils.contacts import load_raw_data_complete
        load_raw_data_complete(i2mb_contact_validation_dir)

    contact_data_unique_complete = (
        pd.read_feather(feather_name)
        .replace("Car", "Bus")
        .set_index(["start", "location", "u_id", "run_id"])
        .fillna(0))
    contact_data_unique_complete.index = contact_data_unique_complete.index.set_levels(
        ["Transport", "Home", "Work", "Others"], level=1)

    return contact_data_unique_complete


def load_grid_data(base_directory):
    data_props = {"grid": {"data_dir": "stage_1_grid"}}
    stage_data = load_stage_data(base_directory, data_props, prefix="stage_1")["grid"]
    index = pd.MultiIndex.from_tuples([tuple(float(idx_) for idx_ in idx.split()[2:]) for idx in stage_data.index])
    index.names = ["Compliance", "Adoption",
                   "Adherence"]  # At this point compliance is actually No Compliance
    stage_data.index = index
    stage_data = stage_data.reset_index()
    stage_data.loc[:, "Compliance"] = 1 - stage_data["Compliance"]
    return stage_data


def compute_daily_activity_durations(data_, other_data=None, other_data_name=None, logged_day=True):
    select_columns = ["id", "day", "activity"]
    if "run" in data_.columns:
        select_columns = ["id", "run", "day", "activity"]

    dfs = [data_.groupby(select_columns, observed=logged_day)["duration"]
           .sum()
           .reset_index()
           .groupby("activity")
           .sample(1000)
           .reset_index(drop=True)
           ]
    keys = ["I2MB"]
    if other_data is not None:
        dfs.append(other_data.groupby(["id", "day", "activity"], observed=logged_day)
                   .sum()["duration"]
                   .reset_index()
                   .reset_index(drop=True))
        keys.append(other_data_name)
    daily_duration = pd.concat(dfs, keys=keys, names=["Source", "Values"], axis=1).stack(0).reset_index(1)
    # duration_ = daily_duration["duration"] != 0
    # daily_duration = daily_duration[duration_]
    return daily_duration
