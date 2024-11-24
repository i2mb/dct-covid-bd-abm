
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

from i2mb.utils import global_time


def recovery_function(t, time_exposed, exposure):
    t = time_exposed
    t = 20 / global_time.time_scalar * (t - 0.55 * global_time.make_time(day=1))

    cdf = 1 - 1 / (1 + np.exp(-t))
    return exposure * cdf
