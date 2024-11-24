import pandas as pd

from dct_covid_bd_abm.simulator.analysis_utils.data_management import load_activity_validation_data
from dct_covid_bd_abm.simulator.assets import extrasensory_data_dir


def activity_validation_table(activity_validation_dir):
    data_, other_data = load_activity_validation_data(activity_validation_dir, extrasensory_data_dir)
    # daily_duration = compute_daily_activity_durations(data_, other_data, "Extrasensory")
    # results = (daily_duration.set_index(["Source", "activity"])
    #            .groupby(level=[0, 1])
    #            .describe()
    #            .loc[:, ("duration", slice(None))]
    #            .unstack(0))

    freq_es = (other_data.groupby(["id", "day", "activity"], observed=False)
               .count()
               .groupby("activity", observed=False)["start"]
               .describe()[["mean", "std"]]
               .agg(lambda x: "{:0.1f} ± {:0.1f}".format(*x), axis=1))

    freq_ = (data_.groupby(["run", "id", "day", "activity"], observed=False)
             .count()
             .groupby("activity", observed=False)["start"]
             .describe()[["mean", "std"]]
             .agg(lambda x: "{:0.1f} ± {:0.1f}".format(*x), axis=1))

    freq_es_logged = (other_data.groupby(["id", "day", "activity"], observed=True)
                      .count()
                      .groupby("activity", observed=False)["start"]
                      .describe()[["mean", "std"]]
                      .agg(lambda x: "{:0.1f} ± {:0.1f}".format(*x), axis=1))

    freq_logged = (data_.groupby(["run", "id", "day", "activity"], observed=True)
                   .count()
                   .groupby("activity", observed=False)["start"]
                   .describe()[["mean", "std"]]
                   .agg(lambda x: "{:0.1f} ± {:0.1f}".format(*x), axis=1))

    # results[("instances", "Frequency", "Extrasensory")] = freq_es
    # results[("instances", "Frequency", "I2MB")] = freq_

    results = pd.concat([freq_es_logged, freq_logged, freq_es, freq_], axis=1)
    results.columns = pd.MultiIndex.from_product(
        [["Daily instances per agent [Mean ± SD]"], ["Logged activity", "Total population"], ["ES", "I2MB"]])
    results.sort_index(inplace=True)

    # person_days = daily_duration.groupby(["Source", "activity"]).sum()["duration"].unstack(0)/(60.*24.)
    # person_days.columns = pd.MultiIndex.from_product([["duration"], ["Person Days"], person_days.columns.to_list()])
    # results = results.join(person_days).sort_index(axis=1)

    # instances = daily_duration.groupby(["Source", "activity"]).count()["duration"].unstack(0)
    # instances.columns = pd.MultiIndex.from_product([["duration"], ["Instances"], instances.columns.to_list()])
    # results = results.join(instances).sort_index(axis=1)

    # results.rename({"duration": "Daily Duration", "instances": "Daily Instances"}, inplace=True, level=0, axis=1)
    # results.rename({"Frequency": "Mean ± SD"}, inplace=True, level=1, axis=1)
    # results.rename({"Extrasensory": "ES", "I2MB": "Sim"}, inplace=True, level=2, axis=1)

    with pd.option_context('display.max_rows', 999, 'display.max_columns', 999, 'display.width', 999):
        print(results)

    return results
