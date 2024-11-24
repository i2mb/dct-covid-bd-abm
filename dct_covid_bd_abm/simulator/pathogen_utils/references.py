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
from numpy import log, sqrt, exp
from scipy.special import erfinv
from scipy.stats import lognorm, gamma, truncnorm

from dct_covid_bd_abm.simulator.assets import assets_dir
from dct_covid_bd_abm.simulator.pathogen_utils.kissler_model import normalize_range


def reference_wrapper(label):
    def wrapper(ref_func):
        def wrapped_ref_func(t=None):
            if t is None:
                t = np.arange(-10, 25, 0.01)

            dist = ref_func()(t)
            return {"t": t, "y": dist, "label": label}

        return wrapped_ref_func
    return wrapper


@reference_wrapper(label="Lauer et al. (2020)")
def incubation_duration_distribution_lauer_et_al():
    # Log norm parameters mean of 5.1
    mu = log(5.1)
    s = (log(11.5) - mu) / (sqrt(2) * erfinv(2 * 0.975 - 1))
    dist = lognorm(s=s, loc=0, scale=exp(mu)).pdf
    return dist


@reference_wrapper(label="Li et al. (2020)")
def incubation_duration_distribution_li_et_al():
    # Log norm parameters mean of 5.2
    # s = solve quadratic equation (log(percentile) - log(mean))
    a = -0.5
    b = sqrt(2) * erfinv(2 * 0.95 - 1)
    c = -(log(12.5) - log(5.2))
    #     print("Solutions for s:", (-b - sqrt(b**2 - 4*a*c)) / (2*a), (-b + sqrt(b**2 - 4*a*c) )/ (2*a))
    s = (-b - sqrt(b ** 2 - 4 * a * c)) / (2 * a), (-b + sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    s = s[1]

    # mu use defintion and log(mean)
    mu = log(5.2) - s ** 2 / 2

    dist = lognorm(s=s, loc=0, scale=exp(mu)).pdf
    return dist


@reference_wrapper(label="Kampen et al. (2020)")
def illness_duration_distribution_kampen_et_al():
    # Log norm parameters mean of 5.2
    # s = solve quadratic equation (log(percentile) - log(mean))
    # We asssume a log normal distribution since they report the median,
    # and the 95th percentile that indicates a skewed distribution.
    # Furthermore, the gamma distribution does not have a close form for
    # estimating the median based on other parametrts.
    mu = log(18)
    s = (log(21) - mu) / (sqrt(2) * erfinv(2 * 0.75 - 1))

    dist = lognorm(s=s, loc=0, scale=exp(mu)).pdf
    return dist


@reference_wrapper(label="Ganyani et al. (2020)")
def generation_interval_ganyani_et_al():
    """
    Ganyani Tapiwa, Kremer Cécile, Chen Dongxuan, Torneri Andrea, Faes Christel,
    Wallinga Jacco, Hens Niel. Estimating the generation interval for coronavirus disease (COVID-19) based on symptom
    onset data, March 2020. Euro Surveill. 2020;25(17):pii=2000257.
    https://doi.org/10.2807/1560-7917.ES.2020.25.17.2000257

    f(x; Θ 1) ≡ Γ(α 1, β 1) and k(δ; Θ 2) ≡ Γ(α_2, β_2). The parameter vector Θ 2 is fixed to (α 2 = 3.45; β 2 = 0.66),
    corresponding to an incubation period with a mean of 5.2 days and a standard deviation (SD) of 2.8 days [6].

    Returns
    -------
    Distribution of generation intervals as estimated by Ganyani et al. 2020
    """

    dist = gamma(a=3.45, scale=1/0.66).pdf
    return dist


@reference_wrapper(label="He et al. (2020)")
def serial_interval_he_et_al():
    """
    He, X., Lau, E.H.Y., Wu, P., Deng, X., Wang, J., Hao, X., Lau, Y.C., Wong, J.Y., Guan, Y., Tan, X.,
    et al. (2020). Temporal dynamics in viral shedding and transmissibility of COVID-19. Nature Medicine 26, 672–675.

    Returns
    -------
    Distribution of serial interval as calculated by He et al. 2020
    """

    csv_file = f"{assets_dir}/Fig1c_data_he_et_al_2020.csv"
    si_he_et_al = pd.read_csv(csv_file, index_col=[0], parse_dates=[1, 2, 3], dayfirst=True)
    bt = si_he_et_al.min().min()
    si_he_et_al -= bt
    x = si_he_et_al.loc[:, ["x.lb", "x.ub"]]
    si_he_et_al["x"] = x.diff(axis=1)["x.ub"] / 2 + si_he_et_al["x.lb"]
    si_he_et_al["si"] = si_he_et_al["y"] - si_he_et_al["x"]
    si_he_et_al = si_he_et_al["si"].apply(lambda x: x.total_seconds() / (3600 * 24))
    alfa, loc1, scale = gamma.fit(si_he_et_al)
    distribution_si_he_et_al = gamma(alfa, loc1, scale).pdf
    return distribution_si_he_et_al


@reference_wrapper(label="Kissler et al. (2020)")
def clearance_period_kissler_et_al():
    """

    Returns
    -------

    """
    N = 65
    s = 2.73
    m = 6.17
    a, b = 2,  30
    a, b = normalize_range(a, b, m, s)

    return truncnorm(a=a, b=b, loc=m, scale=s).pdf
