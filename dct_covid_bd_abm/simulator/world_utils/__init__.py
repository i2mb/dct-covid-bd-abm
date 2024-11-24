import numpy as np
import pandas as pd


def load_homes(run_file):
    run_info = np.load(run_file)
    homes = pd.DataFrame(run_info["homes"], columns=["home_id"])
    homes.index.names = ["u_id"]
    return homes


def load_offices(run_file):
    run_info = np.load(run_file)
    offices = pd.DataFrame(run_info["offices"], columns=["office_id"])
    offices.index.names = ["u_id"]
    return offices


def assign_building_ids(homes, id_field="home_id"):
    home_sizes = homes.join(homes.reset_index().groupby(id_field).count(), on=[id_field])
    home_sizes.columns = [id_field, "size"]
    home_sizes = home_sizes.reset_index().groupby([id_field, "size"]).apply(
        lambda x: pd.Series(np.arange(1, len(x) + 1), index=x.index, name=f"u_{id_field}"))
    home_sizes = home_sizes.reset_index([0, 1])
    home_sizes.index.names = ["u_id"]
    return home_sizes
