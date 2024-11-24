import numpy as np
import pandas as pd
from scipy.stats import nbinom

"""
Sourced from Tomori et al. 2021, Supplementary material file 3
"""

TOMORI_1_WEIGHTED_WITH_GROUPS = dict(
    means=[18.9, 2, 3.3, 6.2, 6.9],
    sds=[24.6, 1.9, 4.7, 18.4, 26.3],
    mins=[1, 0, 0, 0, 0],
    maxs=[512, 16, 102, 500, 674]
)

TOMORI_4_NO_WEIGHTED_NO_GROUPS = dict(
    means=[7.9, 2.1, 3.6, 3.4, 3.3],
    sds=[6.3, 1.9, 6.1, 5.4, 5.3],
    mins=[1, 0, 0, 0, 0],
    maxs=[58, 16, 102, 101, 100]
)

# We excluded Educational from the list
TOMORI_LOCATIONS = ["Overall", "Home", "Work", "Transport", "Others"]

studies = ["POLYMOD", "COVIMOD 1", "COVIMOD 2", "COVIMOD 3", "COVIMOD 4"]
metric_keys = ["Mean", "SD", "Min", "Max"]
metric_keys_2 = ["N", "Mean", "SD", "Min", "Max"]

table_1 = {
    "Overall": [18.9, 24.6, 1, 512, 2.0, 1.9, 0, 16, 3.3, 4.7, 0, 102, 6.2, 18.4, 0, 500, 6.9, 26.3, 0, 674],
    "Home": [3.3, 2.3, 0, 26, 1.6, 1.5, 0, 12, 1.8, 1.7, 0, 9, 1.6, 1.5, 0, 13, 1.6, 1.6, 0, 23],
    "Educational": [1.8, 3.8, 0, 42, 0.0, 0.2, 0, 3, 0.2, 0.9, 0, 10, 0.1, 1.0, 0, 17, 0.7, 3.0, 0, 60],
    "Work": [20.3, 29.5, 0, 509, 0.2, 0.9, 0, 11, 1.5, 6.1, 0, 100, 5.6, 20.4, 0, 140, 5.6, 30.9, 0, 491],
    "Transport": [0.6, 1.7, 0, 8, 0.0, 0.2, 0, 3, 0.1, 0.4, 0, 4, 0.0, 0.2, 0, 8, 0.1, 0.5, 0, 12],
    "Others": [3.4, 4.6, 0, 45, 0.4, 1.0, 0, 10, 0.9, 1.8, 0, 33, 1.5, 4.9, 0, 149, 2.1, 14.1, 0, 674]
}


table_2 = {
    "Overall": [1341, 20.2, 28.5, 1, 512, 1560, 2.1, 1.9, 0, 16, 1356, 3.6, 6.1, 0, 102, 1081, 5.9, 20.3, 0, 500, 1890,
                7.1, 27.9, 0, 674],
    "Home": [1341, 2.8, 2.3, 0, 26, 1560, 1.6, 1.5, 0, 12, 1356, 1.6, 1.6, 0, 9, 1081, 1.5, 1.5, 0, 13, 1890, 1.5, 1.7,
             0, 23],
    "Educational": [199, 2.8, 4.7, 0, 42, 310, 0.0, 0.2, 0, 3, 247, 0.2, 1.0, 0, 10, 179, 0.3, 1.9, 0, 17, 385, 1.2,
                    4.3, 0, 60],
    "Work": [715, 18.5, 33.2, 0, 509, 690, 0.4, 1.2, 0, 11, 613, 2.0, 7.2, 0, 100, 476, 4.0, 15.1, 0, 140, 809, 4.6,
             23.0, 0, 491],
    "Transport": [1341, 0.3, 0.8, 0, 8,
                  1560, 0.0, 0.3, 0, 3,
                  1356, 0.1, 0.4, 0, 4,
                  1081, 0.1, 0.4, 0, 8,
                  1890, 0.1, 0.6, 0, 12],
    "Others": [1341, 3.0, 3.8, 0, 45, 1560, 0.4, 1.0, 0, 10, 1356, 1.1, 2.4, 0, 33, 1081, 1.9, 6.7, 0, 149, 1890, 2.8,
               21.2, 0, 674]
}

table_3 = {
    "Overall": [8.4, 6.7, 1, 58, 2.0, 1.9, 0, 16, 3.3, 4.7, 0, 102, 3.3, 4.2, 0, 101, 3.1, 4.7, 0, 100],
    "Home": [3.3, 2.3, 0, 26, 1.6, 1.5, 0, 12, 1.8, 1.7, 0, 9, 1.6, 1.5, 0, 13, 1.6, 1.6, 0, 23],
    "Educational": [1.8, 3.8, 0, 42, 0.0, 0.2, 0, 3, 0.2, 0.9, 0, 10, 0.1, 0.9, 0, 17, 0.4, 1.8, 0, 22],
    "Work": [2.8, 4.3, 0, 56, 0.2, 0.9, 0, 11, 1.5, 6.1, 0, 100, 1.8, 5.0, 0, 97, 1.4, 5.1, 0, 100],
    "Transport": [0.6, 1.7, 0, 8, 0.0, 0.2, 0, 3, 0.1, 0.4, 0, 4, 0.0, 0.2, 0, 8, 0.1, 0.5, 0, 12],
    "Others": [3.4, 4.6, 0, 45, 0.4, 1.0, 0, 10, 0.9, 1.8, 0, 33, 1.0, 2.2, 0, 35, 0.9, 2.1, 0, 38],
}

table_4 = {
    "Overall": [1341, 7.9, 6.3, 1, 58, 1560, 2.1, 1.9, 0, 16, 1356, 3.6, 6.1, 0, 102, 1081, 3.4, 5.4, 0, 101, 1890, 3.3,
                5.3, 0, 100],
    "Home": [1341, 2.8, 2.3, 0, 26, 1560, 1.6, 1.5, 0, 12, 1356, 1.6, 1.6, 0, 9, 1081, 1.5, 1.5, 0, 13, 1890, 1.5, 1.7,
             0, 23],
    "Educational": [199, 2.8, 4.7, 0, 42, 310, 0.0, 0.2, 0, 3, 247, 0.2, 1.0, 0, 10, 179, 0.2, 1.6, 0, 17, 385, 0.7,
                    2.5, 0, 22],
    "Work": [715, 2.4, 5.2, 0, 56, 690, 0.4, 1.2, 0, 11, 613, 2.0, 7.2, 0, 100, 476, 1.8, 6.3, 0, 97, 809, 1.6, 5.7, 0,
             100],
    "Transport": [1341, 0.3, 0.8, 0, 8, 1560, 0.0, 0.3, 0, 3, 1356, 0.1, 0.4, 0, 4, 1081, 0.1, 0.4, 0, 8, 1890, 0.1,
                  0.6, 0, 12],
    "Others": [1341, 3.0, 3.8, 0, 45, 1560, 0.4, 1.0, 0, 10, 1356, 1.1, 2.4, 0, 33, 1081, 1.1, 2.9, 0, 35, 1890, 1.1,
               2.5, 0, 38]
}

tables_with_names = {
    "Unweighted": table_4,
    "Weighted": table_3,
    "Unweighted and With Group Contacts": table_2,
    "Weighted and With Group Contacts": table_1
}


def generate_baseline_tables(baseline_table):
    df = pd.DataFrame(baseline_table)
    index = pd.MultiIndex.from_product([studies, metric_keys])
    if len(df) == 25:
        index = pd.MultiIndex.from_product([studies, metric_keys_2])

    df.index = index
    return df.T


def generate_complete_table():
    tables = []
    for name, table in tables_with_names.items():
        tables.append(generate_baseline_tables(table))

    return (pd.concat(tables, keys=tables_with_names)
                .drop("N", level=1, axis=1)
                .loc[:, (studies, ["Mean", "SD", "Min", "Max"])])


def generate_tomori_total_contacts(descriptors: dict, shape=10000):
    samples = []
    for mean_, sd_, low, high in zip(*[descriptors[k] for k in metric_keys]):
        if np.isnan(mean_):
            ser = pd.Series(dtype=float)
            samples.append(ser)
            continue

        if mean_ > 0:
            p = mean_ / (sd_ ** 2)
            n = mean_ ** 2 / (sd_ ** 2 - mean_)
            samples.append(pd.Series(nbinom.rvs(n, p, size=shape)))

        else:
            ser = pd.Series(np.zeros(shape) + low)
            n = int((sd_ ** 2) * shape / (high ** 2))
            ser[slice(-n, None)] = high
            samples.append(ser)

    return pd.concat(samples, axis=1, keys=studies)


def generate_tomori_total_contacts_all_experiments(shape=10000):
    samples = {}
    for name, table in tables_with_names.items():
        table_df = generate_baseline_tables(table).T.unstack(1).stack(0)
        for index, group in table_df.groupby(level=1):
            if "Group Contacts" in name:
                group.loc[(["COVIMOD 1", "COVIMOD 2"], slice(None)), :] = np.nan

            descriptors = group.loc[(studies, slice(None)), :].to_dict("list")
            samples[name, index] = generate_tomori_total_contacts(descriptors, shape)

    return samples
