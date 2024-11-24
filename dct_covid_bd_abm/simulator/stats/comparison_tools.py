import pandas as pd
from scipy.stats import ttest_ind


def distribution_similarity(combined_data_frame, other_data_name, p_value=0.05):
    def my_ttest(df):
        df = df.set_index("Source", append=True).unstack(1)
        a = df.loc[:, ("duration", other_data_name)]
        b = df.loc[:, ("duration", "I2MB")]
        return ttest_ind(b, a, equal_var=False, nan_policy="omit").pvalue

    activity_ttests = combined_data_frame.groupby(["activity"]).apply(my_ttest)
    return pd.concat([activity_ttests, activity_ttests > p_value], axis=1, keys=["p_value", "Are similar"])
