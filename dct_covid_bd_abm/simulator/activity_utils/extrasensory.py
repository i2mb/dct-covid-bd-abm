from glob import glob

import pandas as pd


def convert_to_period(x):
    if (x == 0).all():
        return

    start = x.index[0][1]
    end = x.index[-1][1]
    duration = (end - start) + pd.Timedelta(1, unit="minutes")
    return [start, duration, x.name]


def convert_label(label1):
    periods = []
    for ix, group in label1.groupby((label1.diff() != 0).cumsum()):
        period = convert_to_period(group)
        if period is None:
            continue

        periods.append(period)

    return pd.DataFrame(periods, columns=["start", "duration", "activity"]).sort_values("start")


def process_participant(participant):
    labels = []
    for col_name, series in participant.items():
        labels.append(convert_label(series))

    try:
        return pd.concat(labels, ignore_index=True)
    except ValueError:
        return pd.DataFrame([], columns=["start", "duration", "activity"])


def load_extrasensory_dataset(es_data_dir):
    data = []
    feathers = glob(f"{es_data_dir}/*.feather")
    if not feathers:
        raise FileNotFoundError(f"The feather formatted file of the Extrasensory dataset were not found in"
                                f" ´{es_data_dir}´. Make sure to configure the path correctly, and to run "
                                f"csv_to_parquete_converter.py")

    for feather in feathers:
        df = pd.read_feather(feather)
        # df["timestamp"] -=  2 * 3600000  # Correct for time zone data is in UTC
        df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, unit="s", utc=True).tz_convert("US/Pacific")
        # df.index = pd.to_datetime(df.index, unit="s").tz_localize("UTC")
        activities = [col for col in df.columns if "label:" in col]
        df = df.loc[:, activities]
        data.append(df)

    return pd.concat(data, keys=range(len(feathers)), names=["id", "timestamp"])


extrasensory_labels = {
    "location": ['label:LOC_home', 'label:IN_A_CAR', 'label:ON_A_BUS', 'label:LOC_main_workplace', 'label:IN_CLASS',
                 'label:IN_A_MEETING',
                 'label:FIX_restaurant', 'label:AT_A_PARTY', 'label:AT_A_BAR', 'label:LOC_beach', 'label:ELEVATOR',
                 'label:AT_SCHOOL',
                 'label:AT_THE_GYM', ],

    "location_qualifier": ['label:OR_indoors', 'label:OR_outside'],

    "activity_qualifier": ['label:LYING_DOWN', 'label:SITTING', 'label:OR_exercise', 'label:OR_standing',
                           'label:WITH_CO-WORKERS',
                           'label:WITH_FRIENDS'],

    "activity": ['label:FIX_walking', 'label:FIX_running', 'label:BICYCLING', 'label:DRIVE_-_I_M_THE_DRIVER',
                 'label:DRIVE_-_I_M_A_PASSENGER',
                 'label:COOKING', 'label:SHOPPING', 'label:STROLLING', 'label:DRINKING__ALCOHOL_',
                 'label:BATHING_-_SHOWER', 'label:CLEANING',
                 'label:DOING_LAUNDRY', 'label:WASHING_DISHES', 'label:WATCHING_TV', 'label:SURFING_THE_INTERNET',
                 'label:SLEEPING',
                 'label:LAB_WORK', 'label:SINGING', 'label:TALKING', 'label:COMPUTER_WORK', 'label:EATING',
                 'label:TOILET', 'label:GROOMING',
                 'label:DRESSING', 'label:STAIRS_-_GOING_UP', 'label:STAIRS_-_GOING_DOWN'],

    "phone": ['label:PHONE_IN_POCKET', 'label:PHONE_IN_HAND', 'label:PHONE_IN_BAG', 'label:PHONE_ON_TABLE']
}


def convert_to_i2mb_timing(x):
    ref = pd.to_datetime(x.min()["start"].date()).tz_localize("US/Pacific")
    x["day"] = pd.TimedeltaIndex((x["start"] - ref)).days
    x["start"] = (x["day"] * 60*60*24 + pd.TimedeltaIndex((x["start"] - ref)).seconds) / 60
    x["duration"] = pd.TimedeltaIndex(x["duration"]).seconds / 60

    return x


def map_to_resting(data):
    resting = (data.loc[:, ['label:LYING_DOWN',
         'label:SITTING',
         'label:WATCHING_TV']].sum(axis=1) > 0) * data["label:LOC_home"] * ~data["label:SLEEPING"].astype(bool)

    return resting


def map_to_cooking(data):
    cooking = (data.loc[:, ['label:COOKING',
                            'label:WASHING_DISHES']].sum(axis=1) > 0).astype(int)

    return cooking


def map_to_work(data):
    working = (data.loc[:, ['label:LAB_WORK', 'label:IN_CLASS', 'label:IN_A_MEETING',
       'label:LOC_main_workplace',
       'label:COMPUTER_WORK',
       'label:AT_SCHOOL']].sum(axis=1) > 0) * ~data["label:LOC_home"].astype(bool)

    return working


def map_to_coffee_break(data):
    coffee_break = data['label:EATING'].astype(bool) & data['label:LOC_main_workplace'].astype(bool)
    return coffee_break


def map_to_commute_car(data):
    coffee_break = (data.loc[:, ['label:IN_A_CAR',
                                'label:DRIVE_-_I_M_THE_DRIVER',
                                'label:DRIVE_-_I_M_A_PASSENGER']]
                         .sum(axis=1)
                         .astype(bool))
    return coffee_break


def map_extrasensory_to_i2mb(data_es):
    data_es["Rest"] = map_to_resting(data_es)
    data_es["Sleep"] = data_es["label:SLEEPING"]
    data_es["KitchenWork"] = map_to_cooking(data_es)
    data_es["Eat"] = data_es["label:EATING"]
    data_es["Shower"] = data_es["label:BATHING_-_SHOWER"]
    data_es["Grooming"] = data_es["label:GROOMING"]
    # data_es["Sink"] = data_es["label:GROOMING"]
    data_es["Toilet"] = data_es["label:TOILET"]
    data_es["Work"] = map_to_work(data_es)
    data_es['EatAtRestaurant'] = data_es['label:FIX_restaurant']
    data_es['EatAtBar'] = data_es['label:AT_A_BAR']
    data_es['CommuteBus'] = data_es['label:ON_A_BUS']
    data_es['CommuteCar'] = data_es['label:IN_A_CAR']
    data_es['CoffeeBreak'] = map_to_coffee_break(data_es)
    return data_es


def convert_to_period_representation(data_es, i2mb_activities):
    # Convert time series to period representation
    data_es_periods = data_es.loc[:, list(i2mb_activities)].groupby("id").apply(process_participant)
    data_es_periods = data_es_periods.groupby("id").apply(convert_to_i2mb_timing)

    return data_es_periods

