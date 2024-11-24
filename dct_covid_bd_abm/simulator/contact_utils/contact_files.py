from pathlib import Path

import pandas as pd

from dct_covid_bd_abm.configs.base_configuration import config


stages = {
    "POLYMOD": Path(config["data_dir"]) / Path("contact_validation/POLYMOD/test/"),
    "COVIMOD 1": Path(config["data_dir"]) / Path("contact_validation/COVIMOD 1/test/"),
    "COVIMOD 2": Path(config["data_dir"]) / Path("contact_validation/COVIMOD 2/test/"),
    "COVIMOD 3": Path(config["data_dir"]) / Path("contact_validation/COVIMOD 3/test/"),
    "COVIMOD 4": Path(config["data_dir"]) / Path("contact_validation/COVIMOD 4/test/"),
    "Baseline": Path(config["data_dir"]) / Path("pub2022/stage_1/validation_baseline/apartment_homes/"),
    "MCT and ICT": Path(config["data_dir"]) / Path("pub2022/stage_1/validation_with_ict_mct/apartment_homes/"),
    "Perfect DCT":
        Path(config["data_dir"]) / Path("pub2022/stage_2_perfect_behaviour/both_no_rnt_no_qch_bnr/apartment_homes/"),
    "Perfect DCT LBR":
        Path(config["data_dir"]) / Path("pub2022/stage_2_perfect_behaviour/both_no_rnt_no_qch_no_bnr/apartment_homes/"),
    "Imperfect DCT": Path(config["data_dir"]) / Path("pub2022/stage_1_grid/grid_search_0.3_0.5_0.7/apartment_homes/")
}


def apply_filter(contact_history):
    pass


def default_aggregates(df, numeric="sum", other="last", skip=None):
    if skip is not None:
        df = df.drop(skip, axis=1)

    return {k: numeric if pd.api.types.is_numeric_dtype(dt) else other for k, dt in df.dtypes.iteritems()}


def load_contact_history(filename_npz):
    filename_npz = str(filename_npz)
    if "_contact_history.feather" in filename_npz:
        return pd.read_feather(filename_npz)

    if ".npz" in filename_npz:
        try:
            filename_npz = filename_npz.replace(".npz", "_contact_history.feather")
            return pd.read_feather(filename_npz)

        except FileExistsError:
            print(f"{filename_npz.replace('.npz', '_contact_history.feather')}"
                  f"does not exist. Run csv_to_parquete_converter.py {Path(filename_npz).parent}")
            filename_npz = filename_npz.replace(".npz", "_contact_history.csv")
            return pd.read_csv(filename_npz)
