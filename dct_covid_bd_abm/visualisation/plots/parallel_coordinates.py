import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from dashboard.legends import table_legend
from dashboard.parallel_coordinates import ParallelCoordinates
from dashboard.plots import parse_color, fill_between_min_max
from dashboard.text_utils import text_fill
from dct_covid_bd_abm.configs.plot_params import stage2PanelProperties, get_variable_units


def baseline_and_selected_config(dataset, variable_names, fig=None, gs=None, advanced_ax_kwargs=None, ):
    if advanced_ax_kwargs is None:
        advanced_ax_kwargs = {}

    advanced_ax = ParallelCoordinates(fig, variable_names, grid_spec=gs, **advanced_ax_kwargs)

    # Plot perfect behaviour
    __plot_perfect_behaviour(advanced_ax, dataset, variable_names)

    # # Plot selected NPIs
    # props_ = stage2PanelProperties["best_performance"].copy()
    # data = dataset["parameter_search"].loc[props_["discrimination"][0], variable_names]
    # cm = parse_color(data, props_.copy())
    # lw = props_.get("lw", None)
    # advanced_ax.plot(data, lw=lw, color=cm)

    advanced_ax.set_title("DCT-optimal behaviour scenarios")
    # advanced_ax.tight_layout(pad=0.2)
    return advanced_ax


def parse_labels_logical_negation(index):
    return [", ".join(["No" in label
                       and f"$\\mathrm{{\\overline{{"
                           f"{label.replace('No', '').replace('Both', 'DCT').upper()}}}}}$"
                       or label.replace('No ', '').replace('Both', 'DCT').upper()
                       for label in idx]) for idx in index]


def parse_label_only_active(index):

    return [", ".join([label.replace('Both', 'DCT').upper()
                       for label in idx if "No" not in label]) for idx in index]


def parse_label_checkmark_matrix(index):
    matrix = [[0 if "No " in label else 1 for label in idx] for idx in index]
    for row, idx in zip(range(len(index)), index):
        label = idx[0]
        matrix[row][0] = 1 if "Dct" == label else 0 if "No " in label else label

    return matrix


def __plot_perfect_behaviour(advanced_ax, dataset, variable_names, only_selected=False):
    data = dataset["perfect_behaviour"].loc[:, variable_names]
    data = data.sort_values(variable_names[2:], ascending=False)
    props_ = stage2PanelProperties["perfect_behaviour"].copy()
    cm = parse_color(data, props_.copy())
    lw = props_.get("lw", None)
    selected = props_["selected"]
    if not only_selected:
        fill_between_props = props_.copy()
        fill_between_props["cm"] = props_.get("fill_between_cm", props_["cm"])
        fill_between_idx = props_.get("fill_between_index", slice(None))
        fill_between_min_max(advanced_ax, fill_between_props, data.loc[fill_between_idx, :], variable_names)
        lws = [2.5 if idx in selected else 0.5 for idx in data.index]
        advanced_ax.plot(data, lw=lws, color=cm)
        # handles = [plt.Line2D([], [], lw=lw, color=c) for idx, c, lw in
        #            zip(data.index, cm, lws)]
        # label_matrix = parse_label_checkmark_matrix(data.index)
        handles = [
            # Line2D([0], [0], color=cm[0], lw=2.5),
            Rectangle((0., 0.), 0, 0, facecolor=props_.get("fill_between_cm", props_["cm"])[0],
                             edgecolor=None, alpha=0.3),
            Text(0,0,"Baselines"),
            Line2D([0], [0], color=cm[0], lw=2.5),
            Line2D([0], [0], color=cm[-1], lw=2.5),
        ]
        label_matrix = [ [1, -1, -1, -1],  [], [0,0,0,0], [1, 0, 0, 0] ]
        legend = table_legend(handles, data.index.names, label_matrix, advanced_ax, check_mark="P", uncheck_mark="_",
                              title="NPI Performance Range", combined_mark="$\pm$",
                              alignment="center")
        legend.set_bbox_to_anchor([0.55, 0, 2.2, 1])

        return advanced_ax

    advanced_ax.plot(data.loc[selected, :], lw=2.5, color=[cm[data.index.get_loc(s)] for s in selected])

    return advanced_ax


def dct_effect(dataset, variable_names, fig=None, gs=None, advanced_ax_kwargs=None, ):
    if advanced_ax_kwargs is None:
        advanced_ax_kwargs = {}

    advanced_ax = ParallelCoordinates(fig, variable_names, grid_spec=gs, **advanced_ax_kwargs)

    # Plot perfect behaviour
    __plot_perfect_behaviour(advanced_ax, dataset, variable_names, only_selected=True)

    # Plot data
    props_ = stage2PanelProperties["dct_to_no_dct"].copy()
    discrimination = props_.pop("discrimination", [])
    cms = props_.pop("cm", [])
    for group, cm in zip(discrimination, cms):
        local_props = props_.copy()
        local_props["color"] = [cm]
        data = dataset["parameter_search"].loc[group, variable_names]
        cm = parse_color(data, local_props.copy())
        fill_between_min_max(advanced_ax, local_props.copy(), data, variable_names)
        advanced_ax.plot(data, lw=0.5, color=cm)

    advanced_ax.set_title("Realistic behaviour scenario - DCT effect")
    handles = [Rectangle((0., 0.), 0, 0, facecolor=c, edgecolor=None, alpha=0.3) for c in cms]

    label_matrix = [[1, -1, -1, -1], [0, -1, -1, -1]]
    # legend = advanced_ax.legend(handles, parse_labels_logical_negation([[d[0], ] for d in discrimination]),
    #                             mode="expand",
    #                             loc="upper left",
    #                             bbox_to_anchor=(0, +0.45),
    #                             title="Operational Regions",
    #                             ncols=1)

    legend = table_legend(handles, data.index.names, label_matrix, advanced_ax, check_mark="P", uncheck_mark="_",
                          combined_mark="$\pm$", title="NPI Performance Ranges")
    legend.set_bbox_to_anchor([0.55, 0, 2.2, 1])
    return advanced_ax


def dct_and_rnt_effect(dataset, variable_names, fig=None, gs=None, advanced_ax_kwargs=None, ):
    if advanced_ax_kwargs is None:
        advanced_ax_kwargs = {}

    advanced_ax = ParallelCoordinates(fig, variable_names, grid_spec=gs, **advanced_ax_kwargs)

    # Plot perfect behaviour
    __plot_perfect_behaviour(advanced_ax, dataset, variable_names, only_selected=True)

    # Plot data
    props_ = stage2PanelProperties["rnt_to_no_rnt"].copy()
    discrimination = props_.pop("discrimination", [])
    cms = props_.pop("cm", [])
    for group, cm in zip(discrimination, cms):
        local_props = props_.copy()
        local_props["color"] = [cm]
        data = dataset["parameter_search"].loc[group, variable_names]
        cm = parse_color(data, local_props.copy())
        advanced_ax.plot(data, lw=0.5, color=cm)
        fill_between_min_max(advanced_ax, local_props, data, variable_names)

    advanced_ax.set_title("Realistic behaviour scenario - RCT effect with DCT")
    handles = [Rectangle((0., 0.), 0, 0, facecolor=c, edgecolor=None, alpha=0.3) for c in cms]
    label_matrix = [[1, 0, -1, -1], [1, 1, -1, -1]]
    # legend = advanced_ax.legend(handles, parse_labels_logical_negation([["DCT", d[1]] for d in discrimination]),
    #                             mode="expand",
    #                             loc="upper left",
    #                             title="Operational Regions",
    #                             ncols=1)
    legend = table_legend(handles, data.index.names, label_matrix, advanced_ax, check_mark="P", uncheck_mark="_",
                          combined_mark="$\pm$", title="NPI Performance Ranges")

    legend.set_bbox_to_anchor([0.55, 0, 2.2, 1])
    # advanced_ax.tight_layout(pad=0.1)
    return advanced_ax


def remaining_traces(dataset, variable_names, fig=None, gs=None, advanced_ax_kwargs=None, ):
    if advanced_ax_kwargs is None:
        advanced_ax_kwargs = {}

    v_names = [text_fill(f"{v.capitalize()} {u}", 2, min_width=15) for v, u in zip(variable_names, get_variable_units(variable_names))]
    advanced_ax = ParallelCoordinates(fig, v_names, grid_spec=gs, **advanced_ax_kwargs)

    # Plot perfect behaviour
    __plot_perfect_behaviour(advanced_ax, dataset, variable_names, only_selected=True)

    # Plot remaining traces
    props_ = stage2PanelProperties["rnt_to_no_rnt"].copy()
    group = props_.pop("discrimination", [slice(None)])[0]
    data = dataset["parameter_search"].loc[group, variable_names]
    # cm = parse_color(data, props_.copy())
    advanced_ax.plot(data, ls="--")

    advanced_ax.set_title("Realistic behaviour scenario - QCH and CBR effects with "
                          "DCT and no RNT")

    prop_cycle = iter(rcParams["axes.prop_cycle"])
    handles = [plt.Line2D([], [], ls="--", c=next(prop_cycle)["color"]) for _ in data.index]
    label_matrix = parse_label_checkmark_matrix(data.index)
    # legend = advanced_ax.legend(handles, parse_labels_logical_negation(data.index),
    #                             mode="expand",
    #                             loc="upper left",
    #                             # bbox_to_anchor=(0, +0.45),
    #                             ncols=1)
    legend = table_legend(handles, data.index.names, label_matrix, advanced_ax, check_mark="P", uncheck_mark="_",
                          combined_mark="$\pm$", title="Combination Performance")
    legend.set_bbox_to_anchor([0.55, 0, 2.2, 1])
    # advanced_ax.tight_layout(pad=0.1)
    return advanced_ax


def legend_perfect_behaviour(stage2_data, variable_names, fig=None, gs=None, advanced_ax_kwargs=None, ):
    handles, labels = get_handles_and_labels_perfect_behaviour(stage2_data)
    return fig.legend(handles, labels,
                      bbox_to_anchor=gs.get_position(fig).translated(0.05, 0.021),
                      loc="upper left",
                      ncol=3)


def legend_remaining_traces(stage2_data, variable_names, fig=None, gs=None, advanced_ax_kwargs=None, ):
    handles, labels = get_handles_and_labels_remaining_traces(stage2_data)
    return fig.legend(handles, labels,
                      bbox_to_anchor=gs.get_position(fig).translated(0.05, 0.0),
                      loc="lower left",
                      ncol=2)


def get_handles_and_labels_perfect_behaviour(dataset):
    handles = []
    labels = []

    props_ = stage2PanelProperties["perfect_behaviour"].copy()
    selected = props_["selected"]
    labels.extend(parse_labels_logical_negation(selected))
    data = dataset["perfect_behaviour"]
    cm = iter(parse_color(data, props_))
    handles.extend([plt.Line2D([], [], lw=2.5, color=next(cm)) for s in selected])
    return handles, labels


def get_handles_and_labels_remaining_traces(dataset):
    handles = []
    labels = []

    # Final Plot
    props_ = stage2PanelProperties["rnt_to_no_rnt"].copy()
    group = props_.pop("discrimination", [slice(None)])[0]
    data = dataset["parameter_search"].loc[group, :]
    labels.extend(parse_labels_logical_negation(data.index))
    cm = iter(plt.rcParams["axes.prop_cycle"])
    handles.extend([plt.Line2D([], [], ls="--", color=next(cm)["color"]) for idx in data.index])
    return handles, labels
