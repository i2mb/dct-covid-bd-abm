
#  dct_mct_analysis
#  Copyright (c) 2021 FAU - RKI
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

import numpy as np
import pandas as pd
from scipy.special.cython_special import erfinv


def estimate_lognormal_parameters(percentile, percentile_value, mean=None, median=None):
    if mean is not None:
        a = -0.5
        b = np.sqrt(2) * erfinv(2 * percentile/100 - 1)
        c = -(np.log(percentile_value) - np.log(mean))
        print("Solutions for s:", (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a),
              (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a))
        s = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a), (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        s = s[1]
        mu = np.log(mean) - s ** 2 / 2
        return s, mu

    if median is not None:
        mu = np.log(median)
        s = (np.log(percentile_value) - mu) / (np.sqrt(2) * erfinv(2 * percentile/100 - 1))
        return s, mu


def compute_normalized_means(baseline_variables):
    """baseline_variables = load_pathogen_data(data_dir)"""
    ranges = np.zeros((5, 2))
    for ix, variable in enumerate(baseline_variables):
        _max = variable.max().max()
        _min = variable.min().min()

        if ix == 4:
            _min = 0
            _max = 1000

        if ix == 3:
            _min = 0
            _max = 120

        ranges[ix] = _min, _max

    normalized_mean_baseline = []
    for ix, variable in enumerate(baseline_variables):
        normalized_mean_baseline.append(((variable - ranges[ix, 0]) / (ranges[ix, 1] - ranges[ix, 0])).mean())
    return pd.concat(normalized_mean_baseline, axis=1)