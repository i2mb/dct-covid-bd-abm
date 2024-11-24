import os

import numpy as np
import pandas as pd
import scipy.stats as st


def parse_meta_data(meta_data_file_name, field):
    array = np.array
    with open(meta_data_file_name) as f:
        content = f.read()

    meta_data = eval(content.replace("<locals>", "")
                     .replace("functools.partial", "")
                     .replace("<", '"')
                     .replace(">", '"')
                     )
    return meta_data[field]


def load_data_description(data, var_names, data_dir=None):
    full_data = pd.concat(data[:], keys=var_names[:], axis=1)
    for var in var_names:
        full_data.loc[:, var].to_csv(f"{data_dir}/{var.lower().replace(' ', '_')}.csv")

    return full_data.describe().T.unstack(0).swaplevel(0, 1, axis=1)


def load_mean_data(data, var_names):
    def compute_intervals(data_):
        return st.t.interval(alpha=0.95, df=len(data_) - 1,
                             loc=data_.mean(),
                             scale=st.sem(data_, nan_policy="omit"))

    data = pd.concat(data[:], keys=var_names[:], axis=1)
    ci = data.apply(compute_intervals)
    results = pd.concat([data.mean(), ci.T], axis=1)
    results.columns = ["mean", "95%CI low", "95%CI high"]

    return results.ffill(axis=1).unstack(0).swaplevel(0, 1, axis=1)


def load_mode_data(data, var_names):
    return pd.concat(data[:], keys=var_names[:], axis=1).mode().iloc[0, :].unstack(0)


def load_metadata(npz_file, variable):
    config_file_dirname = os.path.dirname(npz_file)
    config_file_name = os.path.join(config_file_dirname, "meta_data")
    try:
        parsed_value = parse_meta_data(config_file_name, variable)
    except KeyError as e:
        if variable != "time_factor_ticks_day":
            raise e

        variable = "time_scalar"
        parsed_value = parse_meta_data(config_file_name, variable)

    return parsed_value
