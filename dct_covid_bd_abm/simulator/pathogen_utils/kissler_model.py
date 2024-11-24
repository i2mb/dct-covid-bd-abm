
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
from numpy import log10
from scipy.stats import truncnorm

from i2mb.utils import global_time

"""> For individuals infected with B.1.1.7, the mean duration of the proliferation phase was 5.3 days (90% credible 
interval [2.7, 7.8]), the mean duration of the clearance phase was 8.0 days [6.1, 9.9], and the mean overall duration 
of infection (proliferation plus clearance) was 13.3 days [10.1, 16.5]. These compare to a mean proliferation phase 
of 2.0 days [0.7, 3.3], a mean clearance phase of 6.2 days [5.1, 7.1], and a mean duration of infection of 8.2 days [ 
6.5, 9.7] for non-B.1.1.7 virus. The peak viral concentration for B.1.1.7 was 19.0 Ct [15.8, 22.0] compared to 20.2 
Ct [19.0, 21.4] for non-B.1.1.7. This converts to 8.5 log10 RNA copies/ml [7.6, 9.4] for B.1.1.7 and 8.2 log10 RNA 
copies/ml [7.8, 8.5] for non-B.1.1.7. Data and code are available online.4 > > N = 65 > > 

-- <cite>Kissler et al. (2021)</cite> 

Kissler, S.M., Fauver, J.R., Mack, C., Tai, C.G., Breban, M.I., Watkins, A.E., Samant, R.M., Anderson, D.J., Ho, 
D.D., Grubaugh, N.D., et al. (2021). Densely sampled viral trajectories suggest longer duration of acute infection 
with B.1.1.7 variant relative to non-B.1.1.7 SARS-CoV-2. medRxiv 2021.02.16.21251535. 


|  Phase  |  B.1.1.7  |  non-B.1.1.7  |
| --- | ------- | ----------- |
| Proliferation | 5.3 days 90% CI [2.7, 7.8] | 2.0 days [0.7, 3.3] |
| Clearance     | 8 days 90% CI [6.1, 9.9] | 6.2 days [5.1, 7.1] |
| Peak Concentration | 8.5 log10 RNA copies/ml [7.6, 9.4] | 8.2 log10 RNA copies/ml [7.8, 8.5] |
"""


def normalize_limit(a, my_mean, my_std):
    return (a - my_mean) / my_std


def normalize_range(a, b, mean, std):
    return normalize_limit(a, mean, std), normalize_limit(b, mean, std)


def log_rna(Ct):
    return (Ct - 40.93733) / (-3.60971) + log10(250)


def maximal_viral_load(size=1):
    N = 65
    s = 4.49 * global_time.time_scalar
    m = 8.2 * global_time.time_scalar

    a, b = log_rna(40), log_rna(0)
    a, b = normalize_range(a, b, m, s)
    return truncnorm.rvs(a=a, b=b, loc=m, scale=s, size=size)


def proliferation_period(size=1):
    N = 65
    s = 3.77 * global_time.time_scalar
    m = 2 * global_time.time_scalar
    a, b = 0.25 * global_time.time_scalar, 14 * global_time.time_scalar
    a, b = normalize_range(a, b, m, s)

    return truncnorm.rvs(a=a, b=b, loc=m, scale=s, size=size)


def clearance_period(size=1):
    N = 65
    s = 2.73 * global_time.time_scalar
    m = 6.17 * global_time.time_scalar
    a, b = 2 * global_time.time_scalar, 30 * global_time.time_scalar
    a, b = normalize_range(a, b, m, s)

    return truncnorm.rvs(a=a, b=b, loc=m, scale=s, size=size)


def compute_symptom_onset(proliferation_duration, clearance_duration, h, area_percentage=0.5):
    """Symptom onset is defined as the point in time relative to the origin of the proliferation period that marks the
     desired area percentage. For COVID-19 that percentage is 50% of the area."""
    x = np.zeros_like(proliferation_duration, dtype=float)

    m_p = h / proliferation_duration
    m_c = -h / clearance_duration

    base = proliferation_duration + clearance_duration
    areas = base * h / 2

    # Mark those configuration where the area percentage requires more that the largest sub-triangle
    direct = area_percentage <= (proliferation_duration / base)
    x[direct] = [np.sqrt(2 * a * area_percentage / m) for m, a in zip(m_p[direct], areas[direct])]

    not_direct = ~direct
    right_x = [np.sqrt((1. - area_percentage) * a * 2 / -m) for m, h_, a in
               zip(m_c[not_direct], h[not_direct], areas[not_direct])]
    x[not_direct] = proliferation_duration[not_direct] + clearance_duration[not_direct] - right_x

    return x


def triangular_viral_load(t, proliferation_duration, clearance_duration, h, *args):
    results = np.zeros_like(proliferation_duration, dtype=float)

    m_p = h / proliferation_duration
    m_c = -h / clearance_duration
    proliferation = t < proliferation_duration
    clearance = (proliferation_duration <= t) & (t < clearance_duration + proliferation_duration)

    results[proliferation] = m_p[proliferation] * t[proliferation]
    results[clearance] = m_c[clearance] * (t[clearance] - proliferation_duration[clearance]) + h[
        clearance]
    results[~(clearance | proliferation)] = 0

    return results / 6
