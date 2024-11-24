#!/usr/bin/env python3
import os
import sys

import pandas as pd

def csv_to_feather(file_name):
    try:
        df = pd.read_csv(file_name, header=0,
                     dtype={" type": "category",
                            " location": "category",
                            " activity": "category"}, on_bad_lines="warn")
    except pd.errors.ParserError as e:
        print(f"Error parsing file {file_name}")
        raise e

    df.columns = [col.strip() for col in df.columns]
    for col, type_ in df.dtypes.items():
        if type_ == "category":
            df[col] = df[col].cat.rename_categories([s.strip() for s in df[col].cat.categories])

    df.to_feather(file_name.replace(".csv", ".feather"))


def get_from_list_with_default(list_: list, index: int, default=None):
    try:
        return list_[index]
    except IndexError:
        print("Index not found using default: '{default}'")
        return default


if __name__ == "__main__":
    dir_ = get_from_list_with_default(sys.argv, 1, "./data/contact_validation")
    for root, dirs, files in os.walk(dir_):
        for f in files:
            if ".csv" in f:
                if os.path.exists(os.path.join(root, f.replace('.csv', '.feather'))):
                    print(f"File {f.replace('.csv', '.feather')} already exists")
                    continue

                print(f"\t{f.replace('.csv', '.feather')}")
                csv_to_feather(os.path.join(root, f))


