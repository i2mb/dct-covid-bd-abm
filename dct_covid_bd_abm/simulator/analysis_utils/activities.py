import numpy as np
import pandas as pd

from dct_covid_bd_abm.simulator.activity_utils.extrasensory import process_participant
from dct_covid_bd_abm.simulator.utilities.file_utilities import load_metadata


def get_location_based_activities(npz_content, time_scalar=1):
    location_named_activity_durations = pd.DataFrame(npz_content["time_spent_by_location"].item()).loc[:,
                                        ["bar", "restaurant", "bus"]]
    if "visit_counter" in npz_content:
        columns = [str(c).split(".")[-1][:-2].lower() for c in npz_content['visit_counter'].item().keys()]
        normalisation = pd.DataFrame(npz_content["visit_counter"].item())
        normalisation.columns = columns
        normalisation = normalisation.loc[:, ["bar", "restaurant", "bus"]]
    else:
        duration = npz_content["waves"][-1][1] / time_scalar  # [days]
        normalisation = duration

    location_named_activity_durations /= normalisation
    location_named_activity_durations.columns = [c.title() for c in location_named_activity_durations.columns]
    return location_named_activity_durations  # [ticks/visit]


def get_activity_duration(npz_file):
    npz_content = np.load(npz_file, allow_pickle=True)

    time_scalar = load_metadata(npz_file, "time_factor_ticks_day")  # [ticks/day]
    minutes_scalar = 1 / (time_scalar / (24 * 60))  # [minutes/tick]

    duration = npz_content["waves"][-1][1] / time_scalar  # [days]

    # [minutes/day]
    activity_duration = pd.DataFrame(npz_content["activity_duration"].item()) * minutes_scalar / duration

    # [minutes/visit], [visit] === [1/day]
    location_named_activity_durations = get_location_based_activities(npz_content, time_scalar) * minutes_scalar
    activity_duration = pd.concat([activity_duration, location_named_activity_durations], axis=1)

    # Return average minutes per day
    return activity_duration


def compute_concurrences_from_timeseries(data, labels):
    co_periods = data.groupby("id").apply(get_co_occurrences_periods, labels)

    co_occurrences_instances = co_periods.groupby(["activity", "act_2"]).count().loc[:,"start"]
    inverse = co_occurrences_instances.copy()
    inverse.index = co_occurrences_instances.index.swaplevel(0,1)
    co_occurrences_instances = pd.concat([co_occurrences_instances, inverse]).sort_index().unstack(0).fillna(0).astype(int)
    return  co_occurrences_instances


def get_co_occurrences_periods(df, select_labels=None):
    co_occurrences = []
    if select_labels is None:
        select_labels = slice(None)

    df = df.loc[:, select_labels]
    for col, g in df.items():
        test_df = df.loc[:, col:].drop(col, axis=1)

        results = process_participant(test_df.mul(g, axis=0))
        results["act_2"] = col
        co_occurrences.append(results)

    return pd.concat(co_occurrences).reset_index(drop=True)


def find_co_occurrences(ix, dataset, matrix, label_column, start_column, duration_column):
    start = dataset.loc[ix, start_column]
    end = dataset.loc[ix, start_column] + dataset.loc[ix, duration_column]
    ends = dataset.loc[:, start_column] + dataset.loc[:, duration_column]
    activity = dataset.loc[ix, label_column]

    overlap = (ends > start) & (dataset.loc[:, start_column] < end)
    overlap.at[ix] = True
    window = dataset.loc[overlap, label_column].cat.remove_unused_categories()
    activities = window.groupby(window).count()
    matrix.loc[activity, activities.index] += activities
    return 0


def compute_co_occurrence_matrix(dataset, label_column=None, start_column=None, duration_column=None):
    if label_column is None:
        label_column = "activity"

    if start_column is None:
        start_column = "start"

    if duration_column is None:
        duration_column = "duration"

    labels = set(dataset.loc[:, label_column])
    co_occurrences_matrix = pd.DataFrame(0, columns=labels, index=labels)

    for ix, row in dataset.iterrows():
        find_co_occurrences(ix, dataset, co_occurrences_matrix, label_column, start_column, duration_column)

    # ugly hack to do it as a side effect
    # dataset.rolling(window=1, closed="left").apply(window_function, args=(dataset, co_occurrences_matrix, label_column, start_column, duration_column))

    self_overlap = pd.Series(co_occurrences_matrix.values.diagonal(), index=co_occurrences_matrix.index).sort_index()
    activity_occurrences = dataset.groupby(label_column).count().iloc[:, 0].sort_index()
    np.fill_diagonal(co_occurrences_matrix.values,
                     (self_overlap - activity_occurrences).loc[co_occurrences_matrix.index])

    return co_occurrences_matrix.fillna(0)