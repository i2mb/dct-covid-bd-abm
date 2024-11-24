from collections import defaultdict

import pandas as pd

from dct_covid_bd_abm.configs.plot_params import stage2DataPlotProperties
from dct_covid_bd_abm.simulator.analysis_utils.data_management import load_stage_data


def format_columns(metric):
    format_ = defaultdict(lambda: "{:.1f}")
    for metric_ in ["False negative rate",
                     "False discovery rate",
                     "Fowlkes Mallows index"]:
        format_[metric_] = "{:.1%}"

    return format_[metric]


def compute_values_in_text(base_directory, experiment, metrics):
    stage2_data = load_stage_data(base_directory, stage2DataPlotProperties, True, prefix="stage_2",
                                  only_means=False)

    def key(x):
        return [0 if "No" in i else 1 for i in x]

    def merge_columns(x):
        metric = x.columns[0][0]
        format_str = format_columns(metric)
        new_ = ("[" + x.iloc[:, 0].apply(format_str.format) + ", " + x.iloc[:, 1].apply(format_str.format) + "]")

        return new_

    data = stage2_data[experiment]

    table = (data.loc[:, (metrics,
                          ['25%', '50%', '75%'])].sort_index(sort_remaining=True, key=key))

    aggregated = table.loc[:, (slice(None), ['25%', '75%'])].groupby(level=0, axis=1).apply(merge_columns)
    aggregated.columns = pd.MultiIndex.from_product([aggregated.columns, ["IQR"]])
    main_values = (table.loc[:, (slice(None),['50%'] )]
                   .rename({"50%": "Median", "mean": "Mean", "std": "StD"}, axis=1))
    main_values = main_values.apply({k: format_columns(k[0]).format for k in main_values.columns.to_list()})

    table = pd.concat([main_values, aggregated], axis=1).sort_index(axis=1)
    table = (table.loc[:, (metrics, ["Median", "IQR"])]
             .rename({m: m.capitalize() for m in metrics} ,axis=1)
             .rename({l: l.upper() for cat in table.index.levels for l in cat}))

    with pd.option_context('display.max_rows', 999, 'display.max_columns', 999, 'display.width', 999):
        print(table)

    return table




