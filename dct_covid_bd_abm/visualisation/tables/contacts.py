import pandas as pd

from dct_covid_bd_abm.simulator.analysis_utils.data_management import load_contact_validation_data
from dct_covid_bd_abm.simulator.contact_utils.tomori_model import generate_complete_table, metric_keys, studies


def contact_validation_table(contact_dataset_dir, mode=None):
    if mode is None:
        mode = "Overall"

    contact_data_unique_complete = load_contact_validation_data(contact_dataset_dir)
    tomori_et_all_contacts = generate_complete_table()
    tomori_contacts = tomori_et_all_contacts.loc[(slice(None), mode), :].droplevel(1).T

    if mode == "Overall":
        i2mb_contacts = (contact_data_unique_complete
                         .groupby(level=[0, 2, 3])
                         .sum(min_count=1)
                         .describe()
                         .loc[["mean", "std", "min", "max"], :])
    else:
        i2mb_contacts = (contact_data_unique_complete
                         .loc[(slice(None), mode, slice(None), slice(None))]
                         .describe()
                         .loc[["mean", "std", "min", "max"], :])

    i2mb_contacts.index = metric_keys
    i2mb_contacts = i2mb_contacts.unstack()
    i2mb_contacts.name = "I2MB"

    contacts_table = (pd.concat([i2mb_contacts, tomori_contacts], axis=1)
                      .swaplevel(0, 1)
                      .sort_index()
                      .loc[(metric_keys, studies), :])

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, "display.float_format",
                           "{:.1f}".format):
        print(contacts_table)

    return contacts_table


def rules_contact_validation_table():
    df = pd.DataFrame(
        [["Experiment", "Running period", "Restrictions"],
         ["POLYMOD", "2005 - 2206", "No restriction"],
         ["CVIMOD1", "30 / 04 to 06 / 05 / 2020", "B & R Lockdown, offices 10 % fixed, 5 % dynamic"],
         ["CVIMOD2", "14 / 05 to 21 / 05 / 2020", "B & R Lockdown, offices 10 % fixed, 20 % dynamic"],
         ["CVIMOD3", "28 / 05 to 04 / 06 / 2020", "B & R 6 H to 22 H, offices 10 % fixed, 40 % dynamic"],
         ["CVIMOD4", "11 / 06 to 22 / 06 / 2020", "B & R 6 H to 22 H, offices 10 % fixed, 60 % dynamic"]])
    df = df.T.set_index(0).T  # set first row as header row.
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, "display.float_format",
                           "{:.1f}".format):
        print(df)

    return df
