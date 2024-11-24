from pathlib import Path

import igraph as ig
from itertools import product

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sb
import upsetplot
from matplotlib import pyplot as plt, rcParams
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pandas.errors import IntCastingNaNError

import dct_covid_bd_abm.configs.evaluation_config as evaluation_config


from dashboard.plot_utils import create_radar_plot_grid
from dashboard.plots import fill_distribution_per_experiment_axis, parse_selected_mask, parse_color, \
    parse_line_weight, draw_advanced_axes_plot, plot_grid_joint
from dashboard.text_utils import text_wrapper

from dct_covid_bd_abm.visualisation.plots.heatmaps import coverage_vs_compliance, coverage_vs_result_sharing, \
    result_sharing_vs_compliance, coverage_vs_compliance_hm, coverage_vs_result_sharing_hm, \
    result_sharing_vs_compliance_hm
from dct_covid_bd_abm.visualisation.plots.networks import layout_wrapped
from dct_covid_bd_abm.visualisation.plots.parallel_coordinates import baseline_and_selected_config, dct_effect, dct_and_rnt_effect, \
    remaining_traces, parse_label_only_active

from dct_covid_bd_abm.configs.plot_params import defaultRadarKwargs, stage1RadarPlotProperties, stage2DataPlotProperties, \
    stage2PanelProperties, validationPanelProperties, get_variable_units

from dct_covid_bd_abm.visualisation.plots.selection_process_plot import create_process_flow
from dct_covid_bd_abm.visualisation.plots.time_series import draw_exposure_function, draw_recovery_function, draw_viral_load, \
    daily_contact_metrics
from dct_covid_bd_abm.simulator.analysis_utils.contacts import load_infection_map_enriched_with_contact_info, load_contact_data, \
    load_daily_contact_metrics_in_time_relative_to_infection, load_daily_contact_metrics
from dct_covid_bd_abm.simulator.analysis_utils.data_management import load_reference_data, load_stage_data, load_validation_data, \
    load_activity_validation_data, load_contact_validation_data, load_grid_data, load_stage_raw_data
from dct_covid_bd_abm.visualisation.plots.distribution_plots import daily_activity_duration_distributions, contact_validation_boxplot
from dct_covid_bd_abm.simulator.assets import extrasensory_data_dir
from dct_covid_bd_abm.simulator.contact_utils.contact_files import stages, load_contact_history


@matplotlib.rc_context({
    "font.size": 10
})
def stage_1_spider_plot(base_directory):
    stage_1_data = load_stage_data(base_directory, stage1RadarPlotProperties, prefix="stage_1")

    fig = plt.figure(figsize=(8.5, 10))
    radars = []
    radars_r_lims = []
    rect_iter = create_radar_plot_grid(3, 2, top_padding=0.05, bottom_padding=0.10,
                                       left_padding=0.075, right_padding=0.075,
                                       v_space=0.15, h_space=0.15)
    for rect, test in zip(rect_iter,
                          ["perfect_behaviour", "process_flow", "dropout", "coverage", "rs_boundary", "dct_intro"]):

        if test == "process_flow":
            # Share Results 100%, Coverage 90%, dropout_MCT 5%,
            ax = fig.add_axes(rect, label="ProcessFlow")
            ax.set_position((0.5, 0.69, 0.48, 0.28))
            create_process_flow(ax)
            continue

        if test not in stage1RadarPlotProperties:
            continue

        data = stage_1_data[test].fillna(0.)
        refs = stage1RadarPlotProperties[test].get("refs", None)
        refs_data = load_reference_data(refs, stage_1_data, stage1RadarPlotProperties)

        radar_kwargs = select_radar_kwargs(stage1RadarPlotProperties[test])
        refs_radar_kwargs = load_reference_radar_kwargs(refs, stage1RadarPlotProperties)
        print(data.T)
        radar = draw_spider_plot(data, fig, references=refs_data, grid_spec=rect,
                                 advanced_ax_kwargs=radar_kwargs,
                                 refs_kwargs=refs_radar_kwargs
                                 )

        radars_r_lims.append(radar.get_r_lims()[:, 1])
        radars.append(radar)

    max_r_lim = np.vstack(radars_r_lims).max(axis=0)
    for radar in radars:
        radar.set_r_lims(max_r_lim)

    create_legend(radars, stage_1_data, stage1RadarPlotProperties,
                  ["perfect_behaviour", "dropout", "coverage", "rs_boundary", "dct_intro"])

    # # Move titles to the left (https://stackoverflow.com/questions/24787041/multiple-titles-in-legend-in-matplotlib)
    # for item, label in zip(radar.ax.legend_.legendHandles, radar.ax.legend_.texts):
    #     if label._text in [v["title"] for v in stage1RadarPlotProperties.values() if "title" in v]:
    #         width = item.get_window_extent(fig.canvas.get_renderer()).width
    #         label.set_ha('left')
    #         label.set_position((-2 * width, 0))
    #         label.set(fontweight="bold")

    return fig


def stage_2_parallel_plot(base_directory, reverse_order=False):
    return __stage_2_plotting("pc", base_directory, reverse_order=reverse_order)


def get_handles_and_labels(dataset):
    handles = []
    labels = []

    # First plot Best behaviour and best NPI combination
    props_ = stage2PanelProperties["perfect_behaviour"]
    selected = props_["selected"]
    labels.extend(parse_label_only_active(selected))
    data = dataset["perfect_behaviour"]
    cm = iter(parse_color(data, props_))
    handles.extend([Line2D([], [], lw=2.5, color=next(cm)) for s in selected])

    # Best NPI combination
    props_ = stage2PanelProperties["best_performance"].copy()
    data = dataset["parameter_search"].loc[props_["discrimination"][0], :]
    cm = iter(parse_color(data, props_.copy()))
    lw = props_.get("lw", None)
    labels.extend(parse_label_only_active(data.index))
    handles.extend([Line2D([], [], lw=2.5, color=next(cm))])

    # Selection plots
    props_ = stage2PanelProperties["dct_to_no_dct"].copy()
    cm = props_["cm"]

    labels.extend(["Selected NPI configuration region",
                   "Rejected NPI configuration region",
                   ])
    handles.extend([plt.Rectangle([0, 0], 1, 1, facecolor=cm_, alpha=0.3) for cm_ in cm])

    return handles, labels


@matplotlib.rc_context({
    "xtick.labelsize": "small",
    "xtick.major.pad": "1",
    "ytick.labelsize": "small",
    "axes.labelsize": "small",
    "axes.labelpad": 0,
    "axes.titlesize": "small",
    "legend.fontsize": "small",
    "legend.title_fontsize": "small",
    "figure.subplot.left": 0.015,
    "figure.subplot.bottom": 0.015,
    "figure.subplot.right": 0.985,
    "figure.subplot.top": 0.975,
    "figure.constrained_layout.hspace": 0.01,
    "figure.constrained_layout.wspace": 0.01,
})
def stage_2_parallel_areas(base_directory):
    variable_order = [
        'Generation Interval',
        'Serial Interval',
        '7-day Hosp. Incidence',
        '7-day Incidence',
        # 'Incubation Period',
        # 'Illness Duration',
        # 'Days in Isolation',
        'Total infected',
        'Wave Duration',
        'Days in Quarantine',

    ]
    stage2_data = load_stage_data(base_directory, stage2DataPlotProperties, True, prefix="stage_2")
    artists = [
        baseline_and_selected_config,
        # legend_perfect_behaviour,
        dct_effect,
        dct_and_rnt_effect,
        remaining_traces,
        # legend_remaining_traces
    ]

    # Apply correct units:
    for df in stage2_data.values():
        df["7-day Incidence"] *= 100
        df["7-day Hosp. Incidence"] *= 100
        df["Total infected"] *= 100 / 1000

    fig = plt.figure(figsize=(6.5, 6.5))
    num_rows = len(artists)
    hratios = np.ones(num_rows) * 0.6
    # hratios[1] = 0.01

    # Space for the legends
    legend_pos = [num_rows - 1]
    # hratios[legend_pos] = 0.1
    rect_iter = fig.add_gridspec(num_rows, 1, wspace=0., hspace=0.4, height_ratios=hratios,
                                 left=0.01, right=.8, top=.97, bottom=.2)

    axes = []
    sharedx = 3
    i = 0
    for gs, artist in zip(rect_iter, artists):
        axes.append(artist(stage2_data, variable_order, fig, gs,
                           advanced_ax_kwargs={"sharedx": sharedx != i, "rot": 45}))
        i += 1

    maximums = np.vstack([d_.loc[:, variable_order].max(axis=0) for d_ in stage2_data.values()]).max(axis=0) * 1.15
    labels = "abcde"
    for ax_idx, advanced_ax in enumerate(axes):
        advanced_ax.set_r_lims(maximums)
        title = advanced_ax.ax.get_title()
        advanced_ax.ax.title.set_x(0.02)
        advanced_ax.set_title(f"{labels[ax_idx]}. {title}")

    # handles, labels = get_handles_and_labels(stage2_data)
    # fig.legend(handles, labels,
    #            bbox_to_anchor=rect_iter[len(artists)].get_position(fig),
    #            loc="lower center",
    #            ncol=3)

    rect_iter.tight_layout(fig, pad=0.2, h_pad=0, w_pad=0)
    return fig


def stage_2_spider_plot(base_directory, reverse_order=False):
    return __stage_2_plotting("spider", base_directory, reverse_order=reverse_order)


@matplotlib.rc_context({
    "xtick.labelsize": "small",
    "xtick.major.pad": "1",
    "ytick.labelsize": "small",
    "axes.labelsize": "small",
    "axes.labelpad": 0,
    "axes.titlesize": "small",
    "legend.fontsize": "small",
    "figure.subplot.left": 0.015,
    "figure.subplot.bottom": 0.015,
    "figure.subplot.right": 0.985,
    "figure.subplot.top": 0.975,
    "figure.constrained_layout.hspace": 0.01,
    "figure.constrained_layout.wspace": 0.01,
})
def __stage_2_plotting(kind, base_directory, reverse_order=False):
    stage_2_data = load_stage_data(base_directory, stage2DataPlotProperties, True, prefix="stage_2")
    variable_order = [
        'Generation Interval',
        'Serial Interval',
        'Wave Duration',
        # 'Incubation Period',
        # 'Illness Duration',
        # 'Days in Isolation',
        'Total infected',
        '7-day Incidence',
        '7-day Hosp. Incidence',
        'Days in Quarantine',
    ]

    maximums = np.vstack([d_.loc[:, variable_order].max(axis=0) for d_ in stage_2_data.values()]).max(axis=0) * 1.15
    radars = []
    test_data = {}
    order = ["perfect_behaviour",
             "best_performance",
             "qch_to_no_qch", "lbr_to_no_lbr",
             "rnt_to_no_rnt", "dct_to_no_dct"
             ]

    if reverse_order:
        order = ["perfect_behaviour",
                 "best_performance",
                 "dct_to_no_dct",
                 "rnt_to_no_rnt",
                 "lbr_to_no_lbr",
                 "qch_to_no_qch",
                 ]

    fig = plt.figure(figsize=(6.5, 10))
    if kind == "pc":
        rect_iter = fig.add_gridspec(len(order), 1, wspace=0., hspace=0.2, left=0.05, right=.9, top=.95, bottom=.1)

    elif kind == "spider":
        rect_iter = create_radar_plot_grid(3, 2, top_padding=0.05, bottom_padding=0.10,
                                           left_padding=0.075, right_padding=0.075,
                                           v_space=0.15, h_space=0.15)

    for rect, test in zip(rect_iter, order):
        if test == "perfect_behaviour":
            data = stage_2_data[test].copy()

        else:
            data = stage_2_data["parameter_search"].copy()

        data = data.rename(index={"Mct": "No Dct", "Bnr": "No Cbr", "No Bnr": "Cbr", "Both": "Dct"})

        data_filters = stage2PanelProperties[test].get("discrimination", None)
        radar_plot_properties = stage2PanelProperties[test].copy()
        if data_filters is not None:
            color_maps = radar_plot_properties.pop("cm", None)
        else:
            data_filters = [slice(None)]
            color_maps = [stage2PanelProperties[test].get("cm")]
        p_coordinates = None
        for selection, cm_ in zip(data_filters, color_maps):
            radar_plot_properties["cm"] = cm_
            refs = radar_plot_properties.get("refs", None)
            refs_data = None
            if refs is not None:
                refs_data = load_reference_data(refs, stage_2_data, stage2DataPlotProperties)
                refs_data = [df.copy().rename(index={"Mct": "No Dct", "Bnr": "No Cbr",
                                                     "No Bnr": "Cbr", "Both": "Dct"})
                             for df in refs_data]

            radar_kwargs = select_radar_kwargs(radar_plot_properties)
            refs_radar_kwargs = load_reference_radar_kwargs(refs, stage2PanelProperties)
            if p_coordinates is not None:
                refs_data = None

            radar_kwargs["sharedx"] = True
            if test == order[-1]:
                radar_kwargs["sharedx"] = False
                radar_kwargs["rot"] = 45

            p_coordinates = draw_advanced_axes_plot(kind, data.loc[selection], fig, references=refs_data,
                                                    grid_spec=rect,
                                                    advanced_ax_kwargs=radar_kwargs,
                                                    refs_kwargs=refs_radar_kwargs,
                                                    advanced_ax=p_coordinates,
                                                    maximums=maximums,
                                                    variable_names=variable_order
                                                    )
        radars.append(p_coordinates)
        test_data[test] = data

    if isinstance(rect_iter, GridSpec):
        rect_iter.tight_layout(fig)

    create_legend(radars, test_data, stage2PanelProperties, order, kind=kind, n_cols=2)

    return fig


@matplotlib.rc_context({
    "xtick.labelsize": "small",
    "xtick.major.pad": "1",
    "ytick.labelsize": "small",
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.labelsize": "small",
    "axes.labelpad": 0,
    "axes.titlesize": "small",
    "figure.subplot.left": 0.015,
    "figure.subplot.bottom": 0.015,
    "figure.subplot.right": 0.985,
    "figure.subplot.top": 0.975,
    "figure.constrained_layout.hspace": 0.01,
    "figure.constrained_layout.wspace": 0.01,
})
def validation_panel(base_directory, activity_validation_dir, contact_validation_dir):
    fig = plt.figure(figsize=(6.5, 8.5), layout="constrained")
    gs0 = fig.add_gridspec(2, 1, height_ratios=[4, 1])

    gs1 = gs0[0].subgridspec(5, 2, height_ratios=[0.1, 1, 1, 1, 1])
    gs2 = gs0[1].subgridspec(2, 1, height_ratios=[0.1, 1])

    for gs, title in zip([gs1[0, 0], gs1[0, 1], gs2[0]],
                         ["A) Activities of daily living", "B) Epidemiological variables", "C) Daily Contacts"]):
        ax = fig.add_subplot(gs)
        ax.text(*[0, 0], title, fontweight="bold", fontsize="small", va="center", ha="left",
                # transform=ax.transAxes,
                # rotation=90
                )
        ax.set_axis_off()
    activity_ax = fig.add_subplot(gs1[1:5, 0])
    contact_ax = fig.add_subplot(gs2[1:])

    # Virus model calibration
    ax1 = fig.add_subplot(gs1[1, 1])
    ax2 = fig.add_subplot(gs1[2, 1])
    ax3 = fig.add_subplot(gs1[3, 1])
    ax4 = fig.add_subplot(gs1[4, 1])
    distro_axs = [ax1, ax2, ax3, ax4]
    viral_model_calibration(base_directory, distro_axs=distro_axs)

    # Activity calibration
    activity_calibration(activity_validation_dir, activity_ax)

    # # contact_ax = ax_dict["contacts"]  # type: plt.Axes
    contact_data_unique_complete = load_contact_validation_data(contact_validation_dir)
    # contact_ax.set_axisbelow(True)
    contact_validation_boxplot(contact_data_unique_complete, ax=contact_ax)
    contact_ax.set_yticks([1, 2, 5, 50, 500])
    return fig


def viral_model_calibration(base_directory, distro_axs=None, legend_kwg=None):
    if distro_axs is None:
        fig = plt.figure(figsize=(6.5, 3.5), layout="constrained")
        gs = GridSpec(3, 2, figure=fig, height_ratios=[0.3, 1, 1])
        distro_axs = [fig.add_subplot(gs[pid]) for pid in product([1, 2], [0, 1])]

    else:
        fig = distro_axs[0].figure()

    plot_label = iter("bcdeaf".upper())
    validation_data = load_validation_data(base_directory)
    ax: plt.Axes
    for var, ax, num in zip(validation_data, distro_axs, plot_label):
        refs = validationPanelProperties[var].get("refs", None)
        ax_label = validationPanelProperties[var].get("rename", var)
        fill_distribution_per_experiment_axis(validation_data[var], ax, axlabel=f"{ax_label} [d]", refs=refs,
                                              legend=False)
        ax.tick_params(pad=0)
        ax.set_xlim(0, 15 * 1.15)
        ax.set_xticks(np.arange(0, 16, step=5))
        ax.set_ylim(0, 0.22)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-1:], labels[-1:], fontsize="small", loc=1)
    #

    # Combined legend:
    default_legend_kwg = dict(fontsize="small", loc="lower left",
                              bbox_to_anchor=(0, 1.025, 1, 1.2), borderaxespad=0.0,
                              mode="expand", ncol=2)
    default_legend_kwg.update(legend_kwg)
    ax_label = distro_axs[0]
    b_legend = fig.legend(handles, ["ICT (simulated)", "MCT (simulated)", "MCT and ICT (simulated)", "Reference"],
                          **default_legend_kwg)
    # ax_label.add_artist(a_legend)
    distro_axs[0].set_xlim(-5, 22)

    return fig


def activity_calibration(activity_validation_dir, activity_ax=None, horizontal=False, logged_day=True):
    if activity_ax is None:
        fig_size = (3.25, 4) if not horizontal else (6.5, 3.25)
        fig = plt.figure(figsize=fig_size, layout="constrained")
        activity_ax = fig.add_subplot()

    else:
        fig = activity_ax.figure()

    i2mb_activity_data, es_activity_data = load_activity_validation_data(activity_validation_dir,
                                                                         extrasensory_data_dir)

    xscale, yscale = (None, "symlog") if horizontal else ("symlog", None)
    daily_activity_duration_distributions(i2mb_activity_data, es_activity_data, "Extrasensory",
                                          logged_day=logged_day,
                                          ax=activity_ax, orient="h" if not horizontal else "v",
                                          yscale=yscale, xscale=xscale,
                                          palette=['#fc8d62', '#4878d0'],
                                          linewidth=1.)

    sb.move_legend(activity_ax, "lower left", title_fontsize="small", fontsize="small", ncol=2,
                   bbox_to_anchor=(0, 1.005, 1, 1.2),
                   borderaxespad=0.0,
                   mode="expand")

    hours_ticks = [5, 15, 60, 4 * 60, 16 * 60]
    hours_ticks_minor = [5, 7.5, 15, 30, 45, 60,
                         2 * 60, 3 * 60, 4 * 60,
                         8 * 60, 12 * 60, 16 * 60]

    if not horizontal:
        activity_ax.set_xticks(hours_ticks)
        activity_ax.set_xticks(hours_ticks_minor, minor=True)
        activity_ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        activity_ax.set_xlim(5, 22 * 60)

        activity_ax.set_yticks(activity_ax.get_yticks(), activity_ax.get_yticklabels(), rotation=45)

    else:
        activity_ax.set_yticks(hours_ticks)
        activity_ax.set_yticks(hours_ticks_minor, minor=True)
        activity_ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        activity_ax.set_ylim(5, 22 * 60)

        activity_ax.set_xticks(activity_ax.get_xticks(), activity_ax.get_xticklabels(), rotation=45, ha="right")

    fig.text(0.03, 0.33, "Log scale", fontsize="x-small", ha="left", va="center")

    return fig


def create_legend(radars, stage_data, radar_plot_properties, test_order, kind="spider", n_cols=4):
    # create legend based on the last radar plot
    for radar, test in zip(radars, test_order):
        data = stage_data[test]
        # radar_kwargs = select_radar_kwargs(radar_plot_properties[test])
        radar_kwargs = apply_default_properties(radar_plot_properties[test])

        legend_lines = []
        legend_labels = []

        n_cols_ = n_cols
        selected = radar_kwargs.pop("selected", None)
        selected_mask = parse_selected_mask(data, selected)
        if "discrimination" not in radar_kwargs:
            # Add lines
            color = parse_color(data, radar_kwargs)
            lw = parse_line_weight(data, radar_kwargs, selected_mask)
            for pos, idx in enumerate(data.index):
                if type(idx) is str:
                    if test == "perfect_behaviour":
                        legend_labels = [
                            "$\\mathrm{\\overline{DCT}}$, $\\mathrm{\\overline{MCT}}$, $\\mathrm{\\overline{ICT}}$",
                            "$\\mathrm{\\overline{DCT}}$, $\\mathrm{\\overline{MCT}}$, ICT",
                            "$\\mathrm{\\overline{DCT}}$, MCT, ICT",
                            "DCT, MCT, ICT"]
                        n_cols_ = 2

                    legend_labels.append(idx.split()[-1])
                    legend_lines.append(plt.Line2D([], [], color=color[pos], lw=lw[pos]))

                else:
                    new_label = ["No" in label
                                 and f"$\\mathrm{{\\overline{{"
                                     f"{label.replace('No', '').replace('Both', 'DCT').upper()}}}}}$"
                                 or label.replace('No ', '').replace('Both', 'DCT').upper()
                                 for label in idx]

                    if len(new_label) == 4:
                        l_ = new_label[-1]
                        new_label[-1] = new_label[-2]
                        new_label[-2] = l_

                    new_label = ", ".join(new_label)
                    legend_labels.append(new_label)
                    legend_lines.append(plt.Line2D([], [], color=color[pos], lw=lw[pos]))

        else:
            discriminants = radar_kwargs.pop("discrimination")

            discriminants_pruned = []
            for discriminants_ in discriminants:
                if type(discriminants_) is tuple:
                    discriminants_pruned.append([type(d) is str and [d] or d for d in discriminants_
                                                 if type(d) is not slice])
                    continue

                if type(discriminants_) is list:
                    discriminants_pruned.append([[c] for d in discriminants_ for c in d])
                    continue

            for discriminant_pruned, cm_ in zip(discriminants_pruned, radar_kwargs.get("cm")):
                for pos, idx in enumerate(product(*discriminant_pruned)):
                    new_label = ["No" in label
                                 and f"$\\mathrm{{\\overline{{{label.replace('No', '').upper()}}}}}$"
                                 or label.replace('No ', '').upper()
                                 for label in idx]

                    if len(new_label) == 4:
                        l_ = new_label[-1]
                        new_label[-1] = new_label[-2]
                        new_label[-2] = l_

                    new_label = ", ".join(new_label)
                    legend_labels.append(new_label)

                    if len(idx) < 4:
                        legend_lines.append(plt.Rectangle([0, 0], 1, 1, color=cm_(len(discriminants_pruned) / (pos + 1))
                                                          , alpha=0.3))
                    else:
                        legend_lines.append(plt.Line2D([], [], color=cm_(len(discriminants_pruned) / (pos + 1))))

        if n_cols == 4:
            if len(legend_labels) % 4 == 0:
                n_cols_ = 4

            elif len(legend_labels) % 3 == 0:
                n_cols_ = 3

        if test == "dct_intro":
            legend_labels = [10, 30, 50]

        if kind == "spider":
            radar.ax.legend(legend_lines, legend_labels,
                            bbox_to_anchor=(0, -0.5, 1, 0.3),
                            ncol=n_cols_,
                            loc="upper center",
                            )

        else:
            fig_transform = radar.fig.transFigure
            ax_transform = radar.ax.transAxes
            bbox = fig_transform.inverted().transform(ax_transform.transform([[0.5, -1.], [1, 0]]))
            l = radar.fig.legend(legend_lines, legend_labels,
                                 # bbox_to_anchor=bbox.ravel(),
                                 ncol=3,
                                 loc="lower left",
                                 )

            l.set(clip_on=False)


def load_reference_radar_kwargs(refs, radar_plot_properties):
    refs_kwargs = None
    if refs is not None:
        refs_kwargs = []
        for rd in refs:
            if rd is None:
                continue

            radar_kwargs_ = select_radar_kwargs(radar_plot_properties[rd].copy())
            refs_kwargs.append(radar_kwargs_)

    return refs_kwargs


def select_radar_kwargs(radar_plot_properties):
    radar_kwargs_ = apply_default_properties(radar_plot_properties)

    # Remove keys not used by radar plot, or draw_spider_plot
    for key in ["data_dir", "refs", "data_filters", "discrimination"]:
        radar_kwargs_.pop(key, None)

    return radar_kwargs_


def apply_default_properties(radar_plot_properties):
    radar_kwargs_ = defaultRadarKwargs.copy()
    radar_kwargs_.update(radar_plot_properties)
    return radar_kwargs_


def grid_search_panel(base_dirname):
    variables = [
        "Days in Quarantine",
        "Wave Duration",
        "Total infected",
        "7-day Incidence",
        "7-day Hosp. Incidence"
    ]
    artists = [
        (coverage_vs_compliance, "Shared test results"),
        (coverage_vs_result_sharing, "Compliance"),
        (result_sharing_vs_compliance, "Adoption"),
    ]
    fig = plt.figure(figsize=(6.5, 8.5), layout="constrained")
    gs0 = fig.add_gridspec(len(variables), 3)
    share_x = True
    stage_data = load_grid_data(base_dirname)
    for row, variable in enumerate(variables):
        if row == len(variables) - 1:
            share_x = False

        v_max = stage_data.loc[:, variable].astype(int).max()
        v_min = stage_data.loc[:, variable].astype(int).min()
        step = (v_max - v_min) / 5

        levels = np.arange(v_min + step, v_max, step)
        print(levels)
        cb = False
        for col, (artist, parameter) in enumerate(artists):
            if col == len(artists) - 1:
                cb = True

            if col == 1:
                share_y = True
            else:
                share_y = False

            ax = fig.add_subplot(gs0[row, col])
            if row == 0:
                ax.set_title(f"v: {parameter}", fontsize="medium")

            artist(stage_data, variable, ax, color_bar=cb, plot_props=dict(sharex=share_x, sharey=share_y,
                                                                           v_max=v_max, v_min=v_min
                                                                           ))

            if row == 0:
                ax.legend()


@matplotlib.rc_context({
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "axes.labelsize": "small",
    "axes.titlesize": "medium",
    "xtick.major.pad": "1",
    "ytick.major.pad": "1",
    "axes.labelpad": "2",
    # "text.usetex": False,
    # "font.family": "Helvetica",

    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Lucida Grande", "Verdana",
                        "Geneva", "Lucid", "Avant Garde", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "font.serif": ["DejaVu Sans", "Arial", "Helvetica", "Lucida Grande", "Verdana",
                   "Geneva", "Lucid", "Avant Garde", "sans-serif"],
    # "font.sans-serif": "Helvetica",
    "mathtext.sf": "DejaVu Sans",
    "legend.fontsize": "small",
    # 'text.latex.preamble': r'\usepackage{cmbright}',
    "pgf.texsystem": "xelatex",
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{sfmath}",

    # "font.size": 10,
    "figure.constrained_layout.h_pad": 0.01,  # 4167  # Padding around axes objects. Float representing
    "figure.constrained_layout.w_pad": 0.01,  # 4167  # inches. Default is 3/72 inches (3 points)
    "figure.constrained_layout.hspace": 0.01,  # 2     # Space between subplot groups. Float representing
    "figure.constrained_layout.wspace": 0.01,  # 2
})
def grid_search_panel_hm(base_dirname, original_search=False, interpolate=False, complete=False):
    variables = {
        'Days in Quarantine': "[d]",
        'Total infected': "[%]",
        '7-day Incidence': "[Cases/100KI]",
        '7-day Hosp. Incidence': "[Cases/100KI]",
    }
    if complete:
        variables.update({
            'Generation Interval': "[d]",
            'Serial Interval': "[d]",
            'Wave Duration': "[d]",
            'Incubation Period': "[d]",
            'Illness Duration': "[d]",
            'Days in Isolation': "[d]"
        })

    artists = [
        (result_sharing_vs_compliance_hm, "Adoption", 0.5),
        (coverage_vs_compliance_hm, "Adherence", 0.7),
        (coverage_vs_result_sharing_hm, "Compliance", 0.7),

    ]
    par_values = dict([t[1:] for t in artists])

    fig = plt.figure(figsize=(6.5, 6.5),
                     layout="constrained",
                     )
    hratios = np.ones(len(variables) + 1)
    hratios[-1] = 0.08
    gs0 = fig.add_gridspec(len(variables)+1, len(artists) + 2, bottom=0, left=0,
                           height_ratios=hratios,
                           width_ratios=[0.09, 1, 1, 1, 0.09])
    share_x = True
    stage_data = load_grid_data(base_dirname)

    original_grid = np.array(list(product([.1, .2, .3, .4],
                                          [.4, .5, .6, .7],
                                          [.6, .7, .8, .9])))
    original_sub_space = original_grid[
        (original_grid[:, 0] == evaluation_config.fixed_do) |
        (original_grid[:, 1] == evaluation_config.fixed_coverage) |
        (original_grid[:, 2] == evaluation_config.fixed_rs)]

    sub_space = evaluation_config.sub_space.copy()

    if original_search:
        sub_space = original_sub_space
        sub_space[:, 0] = 1 - sub_space[:, 0]
    # else:
    #     sub_space = np.vstack([sub_space, original_sub_space])
    #     sub_space = np.unique(sub_space, axis=0)

    sub_space[:, 0] = 1 - sub_space[:, 0]
    stage_data = stage_data.set_index(["Compliance", "Adoption", "Adherence"]).loc[sub_space.tolist(), :]
    stage_data = stage_data.reset_index()
    colorbars = []
    for row, variable in enumerate(variables):
        if row == len(variables) - 1:
            share_x = False

        # Convert to Incidence per 100000
        if "7-day" in variable:
            stage_data.loc[:, variable] *= 100

        if "Affected" in variable:
            stage_data.loc[:, variable] *= 100 / 1000  # Population percentage

        try:
            # v_max = int(np.ceil(np.nanmax(stage_data.loc[:, variable])))
            # v_min = int(np.floor(np.nanmin(stage_data.loc[:, variable])))
            v_max = np.nanmax(stage_data.loc[:, variable])
            v_min = np.nanmin(stage_data.loc[:, variable])

        except IntCastingNaNError as e:
            print(variable, stage_data.loc[:, variable])
            raise e

        cb_ax = fig.add_subplot(gs0[row, len(artists) + 1])
        colorbars.append(cb_ax)

        # Row title:
        title_ax = fig.add_subplot(gs0[row, 0])
        title_ax.axis('off')
        # label_bbox = title_ax.yaxis.label.get_tightbbox().transformed(ax.transAxes.inverted())
        xy = list(title_ax.yaxis.label.get_position())
        # xy[0] -= label_bbox.height
        title_ax.text(*xy, variable.capitalize(), fontweight="bold", fontsize="small", va="center", ha="left",
                      # transform=ax.transAxes,
                      rotation=90)

        for col, (artist, parameter, par_value) in enumerate(artists, start=1):
            cb = col == len(artists) - 1
            share_y = col == 3
            ax = fig.add_subplot(gs0[row, col])
            if row == 0:
                ax.set_title(f"{parameter.title()}: {par_value}")

            add_var_label = col == 0

            artist(stage_data, variable, par_value, ax, mark_values=par_values, color_bar=cb, color_bar_axes=cb_ax,
                   plot_props=dict(sharex=share_x, sharey=share_y,
                                   v_max=v_max, v_min=v_min,
                                   interpolate=interpolate,
                                   draw_min_max_vector=False,
                                   color_bar_label=get_variable_units(variable),
                                   add_variable_label=False))
            ylims = ax.get_ylim()
            ax.set_ylim(ylims[0], 1)
            xlims = ax.get_xlim()
            ax.set_xlim(xlims[0], 1)

    fig.align_ylabels(colorbars)

    # Legend
    cmap = plt.cm.get_cmap(matplotlib.rcParams["image.cmap"])
    handles = [
        (plt.Rectangle([0, 0], 1, 1, facecolor=cmap(0.1)), plt.Line2D([], [], color="w")),
        plt.Line2D([], [], color="gray", marker="x", ls="None"),
        (plt.Rectangle([0, 0], 1, 1, facecolor=cmap(0.9)), plt.Line2D([], [], color="purple")),
        plt.Line2D([], [], color="red", marker="x", ls="None"),
    ]
    labels = ["Bottom 10% of the range", "Simulated point (Avg. over 50 runs)", "Top 10% of the range",
              "Selected parameters"]
    fig.legend(handles, labels,
               # bbox_to_anchor=gs0[len(variables), 1].get_position(fig),
               loc="outside lower center",
               ncol=2)

    return fig


@matplotlib.rc_context({
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "axes.labelsize": "small",
    "axes.titlesize": "medium",
    "xtick.major.pad": "1",
    "ytick.major.pad": "1",
    "axes.labelpad": "2",
    # "text.usetex": False,
    # "font.family": "Helvetica",

    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Lucida Grande", "Verdana",
                        "Geneva", "Lucid", "Avant Garde", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "font.serif": ["DejaVu Sans", "Arial", "Helvetica", "Lucida Grande", "Verdana",
                   "Geneva", "Lucid", "Avant Garde", "sans-serif"],
    # "font.sans-serif": "Helvetica",
    "mathtext.sf": "DejaVu Sans",
    "legend.fontsize": "small",
    # 'text.latex.preamble': r'\usepackage{cmbright}',
    "pgf.texsystem": "xelatex",
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{sfmath}",

    # "font.size": 10,
    "figure.constrained_layout.h_pad": 0.05,  # 4167  # Padding around axes objects. Float representing
    "figure.constrained_layout.w_pad": 0.05,  # 4167  # inches. Default is 3/72 inches (3 points)
    "figure.constrained_layout.hspace": 0.0,  # 2     # Space between subplot groups. Float representing
    "figure.constrained_layout.wspace": 0.0,  # 2
})
def viral_model_innards():
    from dct_covid_bd_abm.configs.base_configuration import recovery_function, kissler_model
    from dct_covid_bd_abm.simulator.pathogen_utils.exposure_functions import distance_exposure

    def create_generator_n_patients(n):
        clearance_duration = kissler_model.clearance_period(n).reshape(-1, 1)
        proliferation_duration = kissler_model.proliferation_period(n).reshape(-1, 1)
        max_viral_load = kissler_model.maximal_viral_load(n).reshape(-1, 1) / kissler_model.log_rna(0)
        symptom_onset = kissler_model.compute_symptom_onset(proliferation_duration, clearance_duration,
                                                            max_viral_load).reshape(-1, 1)

        def __infectiousness_level(t):
            # t = np.tile(t.ravel(), (n, 1)).T
            results = []
            for time in t:
                results.append(kissler_model.triangular_viral_load(np.array(([time] * n)).reshape(-1, 1) +
                                                                   symptom_onset
                                                                   , proliferation_duration,
                                                                   clearance_duration, max_viral_load).ravel())

            results = np.array(results)
            results[results < 0] = 0
            return results

        return __infectiousness_level

    fig = plt.figure(figsize=(6.5, 2), layout="constrained")
    ax_viral_load, ax_exposure, ax_recovery  = fig.subplots(1, 3)

    ax_distance = draw_exposure_function(ax_exposure, distance_exposure)
    ax_distance.set_title("b. Exposure function", fontweight="bold")

    ax_recovery = draw_recovery_function(ax_recovery, recovery_function)
    ax_recovery.set_title("c. Viral load removal", fontweight="bold")

    ax_viral_load = draw_viral_load(ax_viral_load, create_generator_n_patients(50))
    ax_viral_load.set_title("a. Infectiousness levels", fontweight="bold")

    return fig


@matplotlib.rc_context({
    "axes.grid": True,
    "axes.labelsize": "small",
    "axes.titlesize": "medium",

    "xtick.labelsize": "small",
    "xtick.major.pad": "1",

    "ytick.labelsize": "small",
    "ytick.major.pad": "1",
    "axes.labelpad": "2",
    # "text.usetex": False,
    # "font.family": "Helvetica",

    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Lucida Grande", "Verdana",
                        "Geneva", "Lucid", "Avant Garde", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "font.serif": ["DejaVu Sans", "Arial", "Helvetica", "Lucida Grande", "Verdana",
                   "Geneva", "Lucid", "Avant Garde", "sans-serif"],
    # "font.sans-serif": "Helvetica",
    "mathtext.sf": "DejaVu Sans",
    "legend.fontsize": "small",
    # 'text.latex.preamble': r'\usepackage{cmbright}',
    "pgf.texsystem": "xelatex",
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{sfmath}",

    # "font.size": 10,
    "figure.constrained_layout.h_pad": 0.05,  # 4167  # Padding around axes objects. Float representing
    "figure.constrained_layout.w_pad": 0.05,  # 4167  # inches. Default is 3/72 inches (3 points)
    "figure.constrained_layout.hspace": 0.0,  # 2     # Space between subplot groups. Float representing
    "figure.constrained_layout.wspace": 0.0,  # 2

})
def contact_validation(contact_dataset_dir, mode=None):
    fig, contact_ax = plt.subplots(1, 1, figsize=(6.5, 2), layout="constrained")
    contact_data_unique_complete = load_contact_validation_data(contact_dataset_dir)
    contact_ax.set_axisbelow(True)
    contact_validation_boxplot(contact_data_unique_complete, ax=contact_ax, mode=mode)

    return fig


@matplotlib.rc_context({
    # "axes.grid": True,
    "axes.labelsize": "small",
    "axes.titlesize": "medium",

    "xtick.labelsize": "small",
    "xtick.major.pad": "1",

    "ytick.labelsize": "small",
    "ytick.major.pad": "1",
    "axes.labelpad": "2",
    # "text.usetex": False,
    # "font.family": "Helvetica",

    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Lucida Grande", "Verdana",
                        "Geneva", "Lucid", "Avant Garde", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "font.serif": ["DejaVu Sans", "Arial", "Helvetica", "Lucida Grande", "Verdana",
                   "Geneva", "Lucid", "Avant Garde", "sans-serif"],
    # "font.sans-serif": "Helvetica",
    "mathtext.sf": "DejaVu Sans",
    "legend.fontsize": "small",
    # 'text.latex.preamble': r'\usepackage{cmbright}',
    "pgf.texsystem": "xelatex",
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{sfmath}",

    # "font.size": 10,
    "figure.constrained_layout.h_pad": 0.05,  # 4167  # Padding around axes objects. Float representing
    "figure.constrained_layout.w_pad": 0.05,  # 4167  # inches. Default is 3/72 inches (3 points)
    "figure.constrained_layout.hspace": 0.0,  # 2     # Space between subplot groups. Float representing
    "figure.constrained_layout.wspace": 0.0,  # 2

})
def network_metric_distribution(base_path):
    dataframes = []
    test_stages = ["Baseline", "MCT and ICT", "Imperfect DCT", "Perfect DCT"]
    test_stages_names = ["Baseline",
                         "MCT and ICT",
                         "Realistic behaviour",
                         "Optimal behaviour"]
    type_order = ["Complete", "Family", "Friend", "Acquaintance", "Random"]

    for stage in test_stages:
        print(f"Loading stage {stage}")
        stage_dataset = load_contact_data(stages[stage])
        dataframes.append(stage_dataset)

    contact_data = pd.concat(dataframes, keys=test_stages_names)

    data = (contact_data.rename({t:t.title() for t in contact_data.columns.levels[0]}, axis=1)
            .loc[:, (type_order, slice(None))].groupby(level=0)
            .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
            .unstack(0)
            .melt()
            .set_axis(["Type", "Metric", "stage", "Normalised levels"], axis=1))

    g = sb.FacetGrid(data, col="stage", row="Type", hue="Metric", aspect=1.4, height=6 / len(test_stages),
                     margin_titles=True,
                     # subplot_kws=dict(top=0.99)
                     )
    g.map_dataframe(sb.histplot, x="Normalised levels",
                    bins=15,
                    # bins="doane",
                    stat="density", common_norm=False, common_bins=False, linewidth=0)
    # g.set_axis_labels("Total bill ($)", "Tip ($)")

    for text in g._margin_titles_texts:
        text.remove()
    g._margin_titles_texts = []

    if g.row_names is not None:
        # Draw the row titles on the right edge of the grid
        for i, row_name in enumerate(g.row_names):
            ax = g.axes[i, 0]
            # args.update(dict(row_name=row_name))
            # title = row_template.format(**args)
            text = ax.annotate(
                row_name, xy=(-0.2, .5), xycoords="axes fraction",
                rotation=90, ha="right", va="center", fontweight="bold",
            )
            g._margin_titles_texts.append(text)

    # g.set_titles(col_template="{col_name}", row_template="{row_name}", fontweight="bold")
    g.set()

    g.add_legend(loc="upper right", bbox_to_anchor=(0.96, 0.95))
    g.legend.get_title().set(weight="bold")
    g.tight_layout(rect=[0, 0, 1.0, 1.0])
    # g.figure.tight_layout()

    return g.figure


@matplotlib.rc_context({
    # "axes.grid": True,
    "axes.labelsize": "small",
    "axes.titlesize": "medium",

    "xtick.labelsize": "small",
    "xtick.major.pad": "1",

    "ytick.labelsize": "small",
    "ytick.major.pad": "1",
    "axes.labelpad": "2",
    # "text.usetex": False,
    # "font.family": "Helvetica",

    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Lucida Grande", "Verdana",
                        "Geneva", "Lucid", "Avant Garde", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "font.serif": ["DejaVu Sans", "Arial", "Helvetica", "Lucida Grande", "Verdana",
                   "Geneva", "Lucid", "Avant Garde", "sans-serif"],
    # "font.sans-serif": "Helvetica",
    "mathtext.sf": "DejaVu Sans",
    "legend.fontsize": "small",
    # 'text.latex.preamble': r'\usepackage{cmbright}',
    "pgf.texsystem": "xelatex",
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{sfmath}",

    # "font.size": 10,
    "figure.constrained_layout.h_pad": 0.05,  # 4167  # Padding around axes objects. Float representing
    "figure.constrained_layout.w_pad": 0.05,  # 4167  # inches. Default is 3/72 inches (3 points)
    "figure.constrained_layout.hspace": 0.0,  # 2     # Space between subplot groups. Float representing
    "figure.constrained_layout.wspace": 0.0,  # 2

})
def source_target_degree_strength(var_x, var_y, scalers=None, columns_with_units=None):
    import locale
    locale.setlocale(locale.LC_ALL, '')

    if scalers is None:
        scalers = {}

    if columns_with_units is None:
        columns_with_units = {}

    # Helper vectorized functions
    vget_xlim = np.vectorize(lambda x: x.get_xlim() if x is not None else (np.nan, np.nan))
    vset_xlim = np.vectorize(lambda ax_, lower, upper: ax_.set_xlim(lower, upper))

    dataframes = {}
    test_stages = ["Baseline", "MCT and ICT", "Imperfect DCT", "Perfect DCT"]
    test_stages_names = ["Baseline",
                         "MCT and ICT",
                         "Realistic behaviour",
                         "DCT-optimal behaviour"]
    c1, c2, c3, c4 = [v["color"] for v in plt.rcParams["axes.prop_cycle"][:4]]

    # Load data
    for stage in test_stages:
        stage_dataset = load_infection_map_enriched_with_contact_info(stages[stage])
        for k, v in stage_dataset.items():
            dataframes.setdefault(k, []).append(v)

    edge_dataset = pd.concat(dataframes["edges"], keys=test_stages_names)
    for var in set([var_y]).union(var_x):
        s = scalers.get(var, 1)
        if s == 1:
            continue

        edge_dataset[var] *= s

    # edge_dataset[var_y] /= (60 // 5 * 24)
    # vertex_dataset = pd.concat(dataframes["vertices"], keys=test_stages)


    # Create figure
    fig = plt.figure(figsize=(6.5, 2 * len(test_stages)),
                     layout="constrained"
                     )
    gs = fig.add_gridspec(len(test_stages), 2,
                          right=0.8,
                          width_ratios=[0.01, 1])

    plt_labels = "ABCDEFGHI"
    grouper = edge_dataset.groupby(level=0)

    joint_axes = np.full((len(test_stages), 4), None)
    for plt_ix, test in enumerate(test_stages_names):
        data = grouper.get_group(test)

        # Create title ax
        title_ax = fig.add_subplot(gs[plt_ix, 0])
        title_ax.set_axis_off()
        title_ax.text(0, 0.5, test, ha="center", va="center", fontweight="bold", rotation=90)

        # Create sub_figure
        sub_fig = fig.add_subfigure(gs[plt_ix, 1])
        plot_grid_joint(data,
                        y_vars=[var_y],
                        x_vars=[var_x],
                        fig=sub_fig,
                        hue="type",
                        hue_order=['family', 'friend', 'acquaintance', 'random'],
                        palette=[c1, c2, c4, c3],
                        legend_kwargs={"loc": "outside upper left", "borderaxespad": 0.0, "mode": "expand",
                                       "ncols": 4})

        for ax in sub_fig.get_axes():
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()

            xlabel = columns_with_units.get(xlabel, xlabel)
            ylabel = columns_with_units.get(ylabel, ylabel)

            ax.set_xlabel("\n".join(text_wrapper(xlabel, 2, min_width=20)), wrap=True)
            ax.set_ylabel("\n".join(text_wrapper(ylabel, 2, min_width=20)), wrap=True)

            column = ax.get_subplotspec().get_topmost_subplotspec().colspan.start // 5
            if ax.get_subplotspec().is_last_row():
                if not ax.get_subplotspec().is_first_col() and ax.get_subplotspec().is_last_col():
                    continue

                joint_axes[plt_ix, column] = ax

    # Unify x_limits across columns
    limit_grouper = edge_dataset.loc[:, var_x].groupby(level=0)
    lower_limits, upper_limits = limit_grouper.min(), limit_grouper.max()
    np.apply_along_axis(vset_xlim, 1, joint_axes, lower_limits.min().values, upper_limits.max().values)

    return fig


@matplotlib.rc_context({
    # "axes.grid": True,
    "axes.labelsize": "small",
    "axes.titlesize": "medium",

    "xtick.labelsize": "small",
    "xtick.major.pad": "1",

    "ytick.labelsize": "small",
    "ytick.major.pad": "1",
    "axes.labelpad": "2",
    # "text.usetex": False,
    # "font.family": "Helvetica",

    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Lucida Grande", "Verdana",
                        "Geneva", "Lucid", "Avant Garde", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "font.serif": ["DejaVu Sans", "Arial", "Helvetica", "Lucida Grande", "Verdana",
                   "Geneva", "Lucid", "Avant Garde", "sans-serif"],
    # "font.sans-serif": "Helvetica",
    "mathtext.sf": "DejaVu Sans",
    "legend.fontsize": "small",
    # 'text.latex.preamble': r'\usepackage{cmbright}',
    "pgf.texsystem": "xelatex",
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{sfmath}",

    # "font.size": 10,
    "figure.constrained_layout.h_pad": 0.05,  # 4167  # Padding around axes objects. Float representing
    "figure.constrained_layout.w_pad": 0.05,  # 4167  # inches. Default is 3/72 inches (3 points)
    "figure.constrained_layout.hspace": 0.0,  # 2     # Space between subplot groups. Float representing
    "figure.constrained_layout.wspace": 0.0,  # 2
    "figure.titlesize": "small"

})
def daily_nx_metrics(stage, remove_hospital_contacts=True, stage_name=None):
    units = {"Strength": "Strength [min]",
             "Degree": "Degree [Contact persons]",
             }
    if stage_name is None:
        stage_name = stage

    # Load data
    try:
        daily_results_rti = load_daily_contact_metrics_in_time_relative_to_infection(
            remove_hospital_contacts=remove_hospital_contacts,
            base_path=stages[stage])
    except KeyError:
        daily_results_rti = load_daily_contact_metrics_in_time_relative_to_infection(
            remove_hospital_contacts=remove_hospital_contacts,
            base_path=stages[stage], refresh=True)

    try:
        daily_results = load_daily_contact_metrics(
            remove_hospital_contacts=remove_hospital_contacts,
            base_path=stages[stage])
    except KeyError:
        daily_results = load_daily_contact_metrics(
            remove_hospital_contacts=remove_hospital_contacts,
            base_path=stages[stage], refresh=True)

    # Paint the picture
    fig = plt.figure(figsize=(8, 5), layout="constrained")

    gs = fig.add_gridspec(1, 2)
    left_gs = gs[0].subgridspec(2, 1)
    right_gs = gs[1].subgridspec(2, 1)

    daily_results.loc[:, (slice(None), "Strength")] //= (60 // 5)
    daily_results_rti.loc[:, (slice(None), "Strength")] //= (60 // 5)

    daily_results = daily_results.rename(columns=units)
    daily_results_rti = daily_results_rti.rename(columns=units)


    daily_contact_metrics(daily_results,
                          xlims=(0, 140),
                          gs=left_gs,
                          x_label="Simulation time [d]",
                          legend=False)

    daily_contact_metrics(daily_results_rti,
                          xlims=(-100, 100),
                          gs=right_gs,
                          x_label="Time relative to infection time [d]",
                          legend_kw=dict(ncols=3,
                                         loc="outside lower left",
                                         mode="expand",
                                         # bbox_to_anchor=[0.99, 0.97]
                                         ))

    fig.suptitle(stage_name, fontsize="large")
    return fig


@matplotlib.rc_context({
    # "axes.grid": True,
    "axes.labelsize": "small",
    "axes.titlesize": "medium",

    "xtick.labelsize": "small",
    "xtick.major.pad": "1",

    "ytick.labelsize": "small",
    "ytick.major.pad": "1",
    "axes.labelpad": "2",
    # "text.usetex": False,
    # "font.family": "Helvetica",

    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Lucida Grande", "Verdana",
                        "Geneva", "Lucid", "Avant Garde", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "font.serif": ["DejaVu Sans", "Arial", "Helvetica", "Lucida Grande", "Verdana",
                   "Geneva", "Lucid", "Avant Garde", "sans-serif"],
    # "font.sans-serif": "Helvetica",
    "mathtext.sf": "DejaVu Sans",
    "legend.fontsize": "small",
    # 'text.latex.preamble': r'\usepackage{cmbright}',
    "pgf.texsystem": "xelatex",
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{sfmath}",

    # "font.size": 10,
    "figure.constrained_layout.h_pad": 0.05,  # 4167  # Padding around axes objects. Float representing
    "figure.constrained_layout.w_pad": 0.05,  # 4167  # inches. Default is 3/72 inches (3 points)
    "figure.constrained_layout.hspace": 0.0,  # 2     # Space between subplot groups. Float representing
    "figure.constrained_layout.wspace": 0.0,  # 2

})
def network_shapes_by_contact_type(base_path):
    file_name = sorted(Path(base_path).glob("*.npz"))[10]
    contact_history = load_contact_history(file_name)

    ig.config['plotting.backend'] = 'matplotlib'

    type_grouper = contact_history.groupby("type")
    fig = plt.figure(figsize=(7, 7), layout="constrained")
    gs = fig.add_gridspec(2, 2, hspace=0.05)

    labels = list("abcd")
    for index, ss in zip(["family", "friend", "acquaintance", "random"], gs):
        type_df = type_grouper.get_group(index)
        type_df = type_df.loc[:, ("id_1", "id_2", "duration")].groupby(["id_1", "id_2"]).sum().reset_index()
        type_df["duration"] *= 1/12  # assuming 5 minute simulation time steps
        g = ig.Graph.DataFrame(edges=type_df, directed=False, use_vids=False)
        edge_width = g.get_edge_dataframe().loc[:, "duration"] / g.get_edge_dataframe().loc[:, "duration"].max() * 2.5

        norm = matplotlib.colors.Normalize(vmax=g.get_edge_dataframe().loc[:, "duration"].max(),
                                           vmin=g.get_edge_dataframe().loc[:, "duration"].min(),
                                           # linear_width=100
                                           )

        edge_color = matplotlib.colormaps["viridis_r"](norm(g.get_edge_dataframe().loc[:, "duration"]))

        ax = fig.add_subplot(ss)
        ig.plot(g, target=ax, vertex_size=0.25 if index != "random" else 0.01,
                edge_width=edge_width,
                edge_color=edge_color.tolist(),
                layout=layout_wrapped(g, max_items=10))

        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis_r"), ax=ax,
                     label="Accumulated contact duration [hours]",
                     use_gridspec=True,
                     location="bottom",
                     shrink=0.5)

        ax.set_title(f"{labels.pop(0)}. {index.title()}", fontweight="bold", loc="left", pad=0)

    return fig

@matplotlib.rc_context({
    # "axes.grid": True,
    "axes.labelsize": "small",
    "axes.titlesize": "medium",

    "xtick.labelsize": "small",
    "xtick.major.pad": "1",

    "ytick.labelsize": "small",
    "ytick.major.pad": "1",
    "axes.labelpad": "2",
    # "text.usetex": False,
    # "font.family": "Helvetica",

    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Lucida Grande", "Verdana",
                        "Geneva", "Lucid", "Avant Garde", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "font.serif": ["DejaVu Sans", "Arial", "Helvetica", "Lucida Grande", "Verdana",
                   "Geneva", "Lucid", "Avant Garde", "sans-serif"],
    # "font.sans-serif": "Helvetica",
    "mathtext.sf": "DejaVu Sans",
    "legend.fontsize": "small",
    # 'text.latex.preamble': r'\usepackage{cmbright}',
    "pgf.texsystem": "xelatex",
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{sfmath}",

    # "font.size": 10,

})
def metric_analysis_plot(base_directory, metric, order_by="Fowlkes Mallows index"):
    stage2_data = load_stage_raw_data(base_directory, stage2DataPlotProperties, True, prefix="stage_2")

    fig = plt.figure(figsize=(6.5, 8), layout="constrained")
    gs = fig.add_gridspec(ncols=3, nrows=3, height_ratios=[0.025, 9/16,  1], width_ratios=[.3, .5, 1])
    vp_rb = fig.add_subplot(gs[2, 2])
    vp_pb = fig.add_subplot(gs[1, 2], sharex=vp_rb)
    vs_rb = fig.add_subplot(gs[2, 1])
    vs_pb = fig.add_subplot(gs[1, 1], sharex=vs_rb)
    mp_rb = fig.add_subplot(gs[2, 0])
    mp_pb = fig.add_subplot(gs[1, 0], sharex=mp_rb)

    vp_legend = True

    if isinstance(order_by, str):
        order_by = [order_by]

    fmetric = []
    fmetric.extend(order_by)
    fmetric.extend(metric)

    colors = rcParams["axes.prop_cycle"].by_key()['color'][:2][::-1]

    for stage, [vp_ax, vs_ax, mp_ax] in zip(["perfect_behaviour", "parameter_search"], [[vp_pb, vs_pb, mp_pb], [vp_rb, vs_rb, mp_rb]]):
        data_orig = stage2_data[stage].loc[(fmetric, slice(None)), :]
        data_orig = data_orig.rename(columns=lambda x: not "No" in x).T

        if "Total infected" in data_orig.columns:
            data_orig.loc[:, ("Total infected", slice(None))] /= 1000

        sorted_index = data_orig.T.groupby(level=0).mean().T.sort_values(by=order_by).index
        data = data_orig.loc[sorted_index, (metric, slice(None))].melt(ignore_index=False)
        data_order_by = data_orig.loc[sorted_index, (order_by, slice(None))].melt(ignore_index=False)

        # Plot matrix plot
        upset = upsetplot.UpSet(data, totals_plot_elements=0, intersection_plot_elements=0, orientation="vertical")
        upset._element_size = 25
        upset.intersections = upset.intersections.loc[sorted_index[::-1]]
         #sort_index(inplace=True, ascending=False))
        upset.plot_matrix(ax=mp_ax)
        # mp_ax.xaxis.set_ticks_position("bottom")
        mp_ax.set_xlim(-(4 * 0.15), 4 * 1.15)
        mp_ax.xaxis.set_tick_params(rotation=90)
        mp_ax.set_yticks([i for i in range(len(upset.intersections))])
        mp_ax.set_yticklabels([f"OPT {i:2}" for i in range(len(upset.intersections), 0, -1)])


        data["combi"] = data.index.to_frame().astype(int).apply(lambda x: "".join([str(i) for i in x]), axis=1)
        data_order_by["combi"] = data_order_by.index.to_frame().astype(int).apply(lambda x: "".join([str(i) for i in x]), axis=1)
        sb.boxplot(ax=vp_ax, x="value", y="combi",
            hue="variable_0",
            #        split=True,
                   legend=vp_legend,
                   flierprops={"marker": "."},
                   palette=colors,
            #         dodge=False,
            #        cut=0, linewidth=0, inner="box", density_norm="count",
            # inner_kws=dict(box_width=5, whis_width=2, color="0.4", marker="*"),
                   data=data.reset_index(drop=True))

        sb.boxplot(ax=vs_ax, x="value", y="combi",
                   hue="variable_0",
                   #        split=True,
                   legend=vp_legend,

                   flierprops={"marker": "."},
                   # boxprops=dict(facecolor=rcParams["axes.prop_cycle"].by_key()['color'][4], edgecolor="black", lw=0.7),
                   palette=[rcParams["axes.prop_cycle"].by_key()['color'][i] for i in [4, 5]],
                   # width=0.4,
                   data=data_order_by.reset_index(drop=True))

        vp_ax.yaxis.set_visible(False)
        vs_ax.yaxis.set_visible(False)
        for x in ["top", "left", "right"]:
            vp_ax.spines[x].set_visible(False)
            vs_ax.spines[x].set_visible(False)

        tick_axis = vp_ax.xaxis
        tick_axis.grid(True, which="major", linestyle="-")

        vp_legend = False


    mp_rb.xaxis.set_visible(False)
    vp_pb.xaxis.label.set_visible(False)
    vs_pb.xaxis.label.set_visible(False)
    [l.set_visible(False) for l in vp_pb.xaxis.get_ticklabels()]
    [l.set_visible(False) for l in vs_pb.xaxis.get_ticklabels()]
    # vp_pb.xaxis.grid(True, which="major", linestyle="-")

    patches, handles = vp_pb.get_legend_handles_labels()
    patches_s, handles_s = vs_pb.get_legend_handles_labels()
    vs_pb.get_legend().set_visible(False)
    vp_pb.get_legend().set_visible(False)

    fig.legend(patches_s + patches, handles_s + handles, loc="lower left", bbox_to_anchor=[0.23,0.94, 0.77, 0.99],
                 ncols=2, mode="expand")
    vp_rb.set_xlabel("Rate")
    vs_rb.set_xlabel("Rate")
    mp_rb.set_ylabel("Realistic behaviour", fontweight="bold")
    # mp_rb.set_yticklabels([])
    mp_rb.yaxis.set_visible(True)

    mp_pb.set_ylabel("DCT-optimal behaviour", fontweight="bold")
    # mp_pb.set_yticklabels([])
    mp_pb.yaxis.set_visible(True)


    return fig