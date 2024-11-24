from glob import glob

import numpy as np
import pandas as pd


def merge_continuous(x):
    finish = x["start"] + x["duration"]
    expand = x["start"].isin(finish)
    if not expand.any():
        return x

    extend_duration = x["duration"][expand]
    pos_idx = (np.arange(len(x.index))[expand.where(expand, False)])
    col_ix = x.columns.get_loc("duration")
    x.iloc[pos_idx - 1, col_ix] += extend_duration.values
    drop_idx = x.index[pos_idx]
    x = x.drop(drop_idx)

    return x


def load_dataframes(data_dir_):
    feather_files = glob(f"{data_dir_}/*_activity_history.feather")
    data_frames = []
    for feather in feather_files:
        df = pd.read_feather(feather)
        df["day"] = df.start // (60//5*24)
        # df["duration"] = df["duration"] * 5
        # df = df.groupby(["activity", "id", "day"]).mean().groupby(["activity", "id"]).mean()
        df = df.groupby(["id", "activity"]).apply(merge_continuous)
        df.reset_index(inplace=True, drop=True)
        data_frames.append(df)

    data = pd.concat(data_frames, keys=range(len(data_frames)), names=["run", "ix"])
    # Convert time fileds from ticks to minutes
    data["duration"] = data["duration"] * 5
    data["start"] = data["start"] * 5

    # convert run into a normal variable
    data = data.reset_index("run")

    data["location"] = data.location.astype("category")

    data = data.reset_index()
    return data


def create_aggregated_data_file(data_dir):
    data = load_dataframes(data_dir)
    data.drop("ix", axis=1).to_feather(f"{data_dir}/activity_history.feather")


def load_visual_demo_data():
    df = pd.read_csv(
        "data/visual_demo/no_intervention/cw_ohb/no_intervention_cw_ohb_i2bm_sim_data_0000_activity_history.csv")
    df["activity"] = df["activity"].astype("category")
    df["location"] = df["location"].astype("category")
    df["day"] = df["start"] // (12 * 24)
    return df
