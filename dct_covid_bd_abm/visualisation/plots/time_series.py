import numpy as np
from matplotlib import pyplot as plt

from dashboard.plots import draw_time_series
from i2mb.utils import global_time
from dct_covid_bd_abm.simulator.analysis_utils.data_management import get_variable_from_experiments, dict2df
from dct_covid_bd_abm.simulator.analysis_utils.pathogen import get_infection_incidence, get_hospitalization_incidence


def draw_time_series_plot(data_dir, variable_getter, data_frame_processor=None, refs=None,
                          ax=None, baseline_dir=None, draw_refs=False, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax_xlabel = kwargs.pop("axlabel", None)
    kwargs["ax_xlabel"] = ax_xlabel

    if baseline_dir is not None:
        baseline_dict = get_variable_from_experiments(baseline_dir, variable_getter)
        data_frame = dict2df(baseline_dict, replace_infs=True, fill_value=np.nan, remove_words=["validation_"])
        data_frame.sort_index(axis=1, inplace=True)
        data_frame.fillna(0, inplace=True)
        if data_frame_processor is not None:
            data_frame = data_frame_processor(data_frame)

        draw_time_series(data_frame, ax=ax, **kwargs)

    data_dict = get_variable_from_experiments(data_dir, variable_getter)
    data_frame = dict2df(data_dict, replace_infs=True, fill_value=np.nan, remove_words=["validation_"])
    data_frame.sort_index(axis=1, inplace=True)
    data_frame.fillna(0, inplace=True)
    if data_frame_processor is not None:
        data_frame = data_frame_processor(data_frame)

    draw_time_series(data_frame, ax=ax, color="gray", alpha=0.5, lw=1, **kwargs)

    if draw_refs:
        draw_time_series([], ax=ax, reference_lines=refs, **kwargs)

    return ax


def infection_incidence_time_series_plot(data_dir, ax=None, baseline_dir=None, draw_refs=False, **kwargs):
    ax = draw_time_series_plot(data_dir, get_infection_incidence, ax=ax, baseline_dir=baseline_dir,
                               axlabel="7 day incidence [Days]",
                               ax_ylabel="Cases per 1000 individuals",
                               draw_refs=draw_refs, **kwargs)
    return ax


def hospitalisation_incidence_time_series_plot(data_dir, ax=None, baseline_dir=None, draw_refs=False, **kwargs):
    ax = draw_time_series_plot(data_dir, get_hospitalization_incidence, ax=ax, baseline_dir=baseline_dir,
                               axlabel="7 Day Hospitalisation Incidence [Days]",
                               ax_ylabel="Cases per 1000 individuals",
                               draw_refs=draw_refs, **kwargs)
    return ax


def daily_contact_metrics(data, xlims=None, gs: plt.GridSpec = None, legend=True, x_label=None, legend_kw=None):
    if legend_kw is None:
        legend_kw = {}

    grouper = data.unstack([0, 1]).T.groupby(level=[0, 1], sort=False)
    mean = grouper.mean().T
    std_ = grouper.std().T
    upper = mean + 2 * std_
    lower = mean - 2 * std_

    # upper = grouper.max()
    # lower = grouper.min()

    lower[lower < 0] = 0

    if gs is None:
        gs = plt.figure(figsize=(7, 4)).add_gridspec(3, 1)

    fig = gs.figure

    axs = gs.subplots(sharex=True)
    ax: plt.Axes
    for metric, ax in zip(data.columns.levels[1], axs):
        cycle = iter(plt.rcParams["axes.prop_cycle"])
        c = next(cycle)["color"]

        mean.loc[:, ("complete", metric)].plot(color=c, ax=ax, label="All contact types (mean)")
        ax.fill_between(upper.index, upper.loc[:, ("complete", metric)], lower.loc[:, ("complete", metric)], color=c,
                        alpha=.20, label="95% of the data")

        for type_metric in ["family", "friend", "acquaintance", "random"]:
            mean.loc[:, (type_metric, metric)].plot(lw=0.8, ax=ax, color=next(cycle)["color"],
                                                    label=f"{type_metric.title()} (mean)")

        ax.set_ylabel(metric)
        if x_label is not None:
            ax.set_xlabel(x_label)

        if xlims is not None:
            ax.set_xlim(xlims)

        if legend:
            legend_kw.setdefault("loc", "outer upper right")
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, **legend_kw)
            # ax.legend(**legend_kw)
            legend = False

    return axs


def draw_recovery_function(ax, generator):
    x = np.arange(global_time.make_time(day=1.5), step=global_time.make_time(day=1) * 0.01)
    y = 1 - generator(None, x, np.ones(len(x)))

    # Change x to days
    x = np.arange(1.5, step=0.01)
    ax: plt.Axes
    ax.plot(x, y)
    ax.set_xlabel("Time post-exposure [d]")
    ax.set_ylabel("$r_i$")
    return ax


def draw_viral_load(ax, generator):
    x = np.arange(global_time.make_time(day=-14), global_time.make_time(day=14), step=global_time.make_time(day=1) * 0.1)
    y = generator(x)
    x = np.arange(-14, 14, step=0.1)

    lc = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    sample_lines = ax.plot(x, y, lw=0.5, color=lc, alpha=0.5, label="Samples")

    lc = plt.rcParams['axes.prop_cycle'].by_key()['color'][3]
    mean = ax.plot(x, y.mean(axis=1), color=lc, label="Mean")
    ax.legend([sample_lines[0], mean[0]], [f"{y.shape[1]} Profiles", "Mean"])
    ax.set_xlabel("Time since symptom onset [d]")
    ax.set_ylabel("$il_i$")

    return ax


def draw_exposure_function(ax, generator):

    x = np.arange(10, step=0.01)
    y = generator(x)
    ax.plot(x, y)
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("$ex_{ij}$")
    return ax
