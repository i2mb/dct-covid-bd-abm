
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
from scipy.special.cython_special import erfinv
from scipy.stats import lognorm

from i2mb.utils import global_time


def inf_func(t, incubation_duration, illness_duration, *args):
    """t  is the elapsed time from the time of infection"""
    # Parameters estimated from CWA which iun term are estimated from Hu et al. 2020b (the correction)
    illness_duration /= global_time.time_scalar
    incubation_duration /= global_time.time_scalar
    loc = -19.1024 + incubation_duration
    scale = 19.00568
    scale += illness_duration - scale
    s = 0.147551
    s *= (illness_duration + incubation_duration) / scale
    t = (t / global_time.time_scalar - loc) / scale
    pdf = 1 / (t * s * np.sqrt(2 * np.pi)) * np.exp(-1 * np.log(t) ** 2 / (2 * s ** 2)) / scale
    max_t = np.exp(-s ** 2)
    max_cdf = 1 / (max_t * s * np.sqrt(2 * np.pi)) * np.exp(-1 * np.log(max_t) ** 2 / (2 * s ** 2)) / scale

    # Normalize
    pdf /= max_cdf

    return pdf


def lognormal_infectious_function(t, incubation_duration, illness_duration, *args):
    """t  is the elapsed time from the time of infection"""

    infectious_duration_pso = illness_duration
    a = 1
    b = np.sqrt(2) * erfinv(2 * 0.95 - 1)
    c = -(np.log(infectious_duration_pso + incubation_duration) - np.log(incubation_duration))
    s = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    mu = np.log(incubation_duration) + s ** 2
    max_t = np.exp(mu - s ** 2)

    infectiousness_i = lognorm.pdf(t, s=s, scale=np.exp(mu))
    norm_factor = lognorm.pdf(max_t, s=s, scale=np.exp(mu))
    return infectiousness_i / norm_factor


def lognormal_with_delay_infectious_function(t, incubation_duration, illness_duration, *args):
    """t  is the elapsed time from the time of infection"""

    infectious_duration_pso = illness_duration
    delay = 1. * global_time.time_scalar
    incubation_duration = incubation_duration - delay
    a = 1
    b = np.sqrt(2) * erfinv(2 * 0.95 - 1)
    c = -(np.log(infectious_duration_pso + incubation_duration) - np.log(incubation_duration))
    s = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    mu = np.log(incubation_duration) + s ** 2
    max_t = np.exp(mu - s ** 2) + delay

    infectiousness_i = lognorm.pdf(t, s=s, loc=delay, scale=np.exp(mu))
    norm_factor = lognorm.pdf(max_t, s=s,  loc=delay, scale=np.exp(mu))
    return infectiousness_i / norm_factor


def triangular_infectious_function(t, incubation_duration, illness_duration, *args):
    """Creates a triangular function which peaks at symptom onset with a value of 1. The function range is [0,
    incubation_duration + illness_duration]

    Parameters
    ----------
    t:  is the elapsed time from the time of infection
    incubation_duration: np.array   Period of time between time of infection and symptom-onset
    illness_duration: period of time between symptom onset and recovery/not being infectious
    """

    incubating = (t < incubation_duration).ravel()
    results = np.zeros_like(incubation_duration, dtype=float)
    m = 1/incubation_duration[incubating]
    results[incubating] = m * t[incubating]

    m = -1 / (illness_duration[~incubating])
    results[~incubating] = m * (t[~incubating] - incubation_duration[~incubating]) + 1
    return results


def triangular_infectious_function_with_delay(t, incubation_duration, illness_duration, *args):
    """Creates a triangular function which peaks at symptom onset with a value of 1. The function range is [delay,
    incubation_duration + illness_duration] delay is fixed at 1 day, e.g., 1 * global_time.time_scalar.

    Parameters
    ----------
    t:  is the elapsed time from the time of infection
    incubation_duration: np.array   Period of time between time of infection and symptom-onset
    illness_duration: period of time between symptom onset and recovery/not being infectious
    """
    delay = 1. * global_time.time_scalar
    incubating = (t < incubation_duration).ravel()
    results = np.zeros_like(incubation_duration, dtype=float)
    m = 1. / (incubation_duration[incubating] - delay)
    results[incubating] = m * (t[incubating] - delay)
    results[results < 0] = 0

    m = -1 / (illness_duration[~incubating])
    results[~incubating] = m * (t[~incubating] - incubation_duration[~incubating]) + 1
    return results


def triangular_infectious_function_so_in_mean(t, incubation_duration, illness_duration, *args):
    """Creates a triangular distribution function with the mean at symptom onset. The function range is [0,
    incubation_duration + illness_duration]

    Parameters
    ----------
    t:  is the elapsed time from the time of infection
    incubation_duration: np.array   Period of time between time of infection and symptom-onset
    illness_duration: period of time between symptom onset and recovery/not being infectious
    """

    h = 2/(illness_duration + incubation_duration)
    x1 = (1-incubation_duration*h)*illness_duration
    x = incubation_duration - x1

    incubating = (x > t).ravel()
    results = np.zeros_like(incubation_duration, dtype=float)
    m = h[incubating] / x[incubating]
    results[incubating] = m * t[incubating]

    m = -h[~incubating] / ((illness_duration[~incubating] + incubation_duration[~incubating]) - x[~incubating])
    results[~incubating] = m * (t[~incubating] - x[~incubating]) + h[~incubating]
    results[results < 0] = 0

    return results / h


def triangular_infectious_function_so_in_mean_with_delay(t, incubation_duration, illness_duration, *args):
    """Creates a triangular distribution function with the mean at symptom onset. The function range is [0,
    incubation_duration + illness_duration]

    Parameters
    ----------
    t:  is the elapsed time from the time of infection
    incubation_duration: np.array   Period of time between time of infection and symptom-onset
    illness_duration: period of time between symptom onset and recovery/not being infectious
    """

    delay = 1. * global_time.time_scalar
    # if delay > incubation_duration:
    #     incubation_duration = delay

    h = 2/(illness_duration + incubation_duration - delay)
    x1 = (1-(incubation_duration - delay)*h)*illness_duration
    x = (incubation_duration - delay) - x1

    results = np.zeros_like(incubation_duration, dtype=float)
    delayed = (t < delay).ravel()
    results[delayed] = 0

    incubating = ((t >= delay) & (t < (x + delay))).ravel()
    m = h[incubating] / x[incubating]
    results[incubating] = m * (t[incubating] - delay)

    clearance = ~(incubating | delayed)
    m = -h[clearance] / ((illness_duration[clearance] + incubation_duration[clearance]) - x[
        clearance])
    results[clearance] = m * (t[clearance] - x[clearance]) + h[clearance]
    results[results < 0] = 0

    return results / h


def step_down_function(t, incubation_duration, illness_duration, *args):
    """Step function to simulate presymptomatic transmission

    Parameters
    ----------
    t:  is the elapsed time from the time of infection
    incubation_duration: np.array   Period of time between time of infection and symptom-onset
    illness_duration: period of time between symptom onset and recovery/not being infectious
    """

    return (t <= incubation_duration) * 1.


def step_up_function(t, incubation_duration, illness_duration, *args):
    """Creates a step function where viral load production goes to one after symptom onset

    Parameters
    ----------
    t:  is the elapsed time from the time of infection
    incubation_duration: np.array   Period of time between time of infection and symptom-onset
    illness_duration: period of time between symptom onset and recovery/not being infectious
    """

    return (t > incubation_duration)*1.
