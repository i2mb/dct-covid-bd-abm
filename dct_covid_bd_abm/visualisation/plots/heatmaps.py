from itertools import product

import numpy as np
import pandas as pd
import ternary

from matplotlib import pyplot as plt, rcParams
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import RectBivariateSpline

from dct_covid_bd_abm.simulator.analysis_utils.data_management import load_grid_data


def grid_ternary_heat_map(base_directory):
    """Trying to plot ternary factor. it does not work since not all values of the grid == to K"""

    stage_data = load_grid_data(base_directory)
    data = stage_data.loc[:, ("Compliance", "Adoption", "Adherence")].to_numpy()
    scale = max(data.sum(axis=1))
    data = compute_simplex(data, scale)

    draw_ternary_heat_map(data, scale)

    return


def draw_ternary_heat_map(data, scale):
    figure, tax = ternary.figure(scale=scale)

    # Draw Boundary and Gridlines
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="black", multiple=scale * 0.1)
    tax.gridlines(color="blue", multiple=scale * 0.2, linewidth=1)

    # Set Axis labels and Title
    fontsize = "medium"
    tax.bottom_axis_label("Compliance", fontsize=fontsize, offset=0.14)
    tax.right_axis_label("Adoption", fontsize=fontsize, offset=0.14)
    tax.left_axis_label("Adherence", fontsize=fontsize, offset=0.14)

    tax.scatter(data, marker='s', color='red', label="Red Squares")
    tax.ticks(axis='cbr', multiple=scale * 0.2, linewidth=1, offset=0.025)
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()


def coverage_vs_compliance(stage_data, variable, ax=None, color_bar=True, plot_props=None):
    """Shows a heatmap of the dependency between coverage and compliance"""

    ax = draw_contour_lines(ax, stage_data, variable, "Compliance", "Adoption", "Adherence",
                            color_bar=color_bar, plot_props=plot_props)

    return ax


def coverage_vs_result_sharing(stage_data, variable, ax=None, color_bar=True, plot_props=None):
    """Shows a heatmap of the dependency between coverage and compliance"""

    ax = draw_contour_lines(ax, stage_data, variable, "Adherence", "Adoption", "Compliance",
                            color_bar=color_bar, plot_props=plot_props)

    return ax


def result_sharing_vs_compliance(stage_data, variable, ax=None, color_bar=True, plot_props=None):
    """Shows a heatmap of the dependency between coverage and compliance"""

    ax = draw_contour_lines(ax, stage_data, variable, "Compliance", "Adherence", "Adoption",
                            color_bar=color_bar, plot_props=plot_props)

    return ax


def coverage_vs_compliance_hm(stage_data, variable, parameter_val=None, ax=None, mark_values=None, color_bar=True,
                              color_bar_axes=None, plot_props=None):
    """Shows a heatmap of the dependency between coverage and compliance"""

    ax = draw_heatmaps(ax, stage_data, variable, "Compliance", "Adoption", "Adherence", parameter_val,
                       mark_values=mark_values,
                       color_bar=color_bar, color_bar_axes=color_bar_axes, plot_props=plot_props)

    return ax


def coverage_vs_result_sharing_hm(stage_data, variable, parameter_val, ax=None, mark_values=None, color_bar=True,
                                  color_bar_axes=None, plot_props=None):
    """Shows a heatmap of the dependency between coverage and compliance"""

    ax = draw_heatmaps(ax, stage_data, variable, "Adherence", "Adoption", "Compliance", parameter_val,
                       mark_values=mark_values, color_bar=color_bar,  color_bar_axes=color_bar_axes,
                       plot_props=plot_props)

    return ax


def result_sharing_vs_compliance_hm(stage_data, variable, parameter_val, ax=None, mark_values=None, color_bar=True,
                                    color_bar_axes=None, plot_props=None):
    """Shows a heatmap of the dependency between coverage and compliance"""

    ax = draw_heatmaps(ax, stage_data, variable, "Compliance", "Adherence", "Adoption", parameter_val,
                       mark_values=mark_values,
                       color_bar=color_bar, color_bar_axes=color_bar_axes, plot_props=plot_props)

    return ax


def draw_contour_lines(ax, stage_data, variable, x_var, y_var, grouping, color_bar=True, plot_props=None):
    ax: plt.Axes
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if plot_props is None:
        plot_props = {}

    v_max = plot_props.get("v_max", None)
    v_min = plot_props.get("v_min", None)
    levels = plot_props.get("levels", 3)
    interpolate_ = plot_props.get("interpolate", False)
    for group_ix, group in stage_data.groupby(grouping):
        affected_population = group.set_index([y_var, x_var]).unstack(1).loc[:, variable]
        affected_population_interpolated = interpolate_data(affected_population, x_var, y_var)
        x = affected_population_interpolated.columns
        y = affected_population_interpolated.index
        contours = ax.contour(x, y, affected_population_interpolated,
                              levels=3,
                              vmax=v_max, vmin=v_min, extend="both",
                              linewidths=0.5 + (group_ix - 0.5) / 0.4 * 3)
        # cl = ax.clabel(contours, fmt=lambda x: f'v: {group_ix}')

    if color_bar:
        cb = plt.colorbar(ScalarMappable(norm=Normalize(v_min, v_max)))
        cb.set_label(variable.capitalize(), labelpad=5)

    shared_x = plot_props.get("sharex", False)
    shared_y = plot_props.get("sharey", False)
    if shared_x:
        ax.xaxis.set_ticklabels([])
    else:
        ax.set_xlabel(x_var.capitalize())

    if shared_y:
        ax.yaxis.set_ticklabels([])
    else:
        ax.set_ylabel(y_var.capitalize())

    return ax


def draw_heatmaps(ax, stage_data, variable, x_var, y_var, parameter, parameter_value, mark_values=None,
                  color_bar=True, color_bar_axes=None, plot_props=None):
    ax: plt.Axes
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if plot_props is None:
        plot_props = {}

    color_bar_label = plot_props.pop("color_bar_label", variable)
    add_variable_label = plot_props.pop("add_variable_label", False)
    label_contour_lines = plot_props.pop("label_contour_lines", False)

    v_max = plot_props.get("v_max", None)
    v_min = plot_props.get("v_min", None)
    interpolate_ = plot_props.get("interpolate", False)
    fill_gaps_ = plot_props.get("fill_gaps", False)
    data = (stage_data.set_index(parameter)
                      .loc[parameter_value, :]
                      .set_index([y_var, x_var])
                      .unstack(1)
                      .loc[:, variable])

    if fill_gaps_:
        data = interpolate_data(data, x_var, y_var)
        data.index.name = x_var
        data.columns.name = y_var

    if interpolate_:
        interpolated_data = interpolate_data(data, x_var, y_var, num=30)
    else:
        interpolated_data = data

    # averaged_ = convolve(data.fillna(0).to_numpy(), np.ones((3, 3)) / 9, mode="mirror", cval=0)
    # nans_ = np.where(interpolated_data.isnull())
    # interpolated_data.iloc[nans_[0], nans_[1]] = averaged_[nans_[0], nans_[1]]

    x = interpolated_data.columns
    y = interpolated_data.index
    color_map = ax.pcolor(x, y, interpolated_data, vmax=v_max, vmin=v_min, )
    min_ = min(interpolated_data.values.ravel())
    max_ = max(interpolated_data.values.ravel())
    boundaries_10p = [v_min + (v_max - v_min) * 0.1, v_max - (v_max - v_min) * 0.1]
    contours = ax.contour(x, y, interpolated_data,
                          # levels=[min_ + (max_ - min_) * 0.1, max_ - (max_ - min_) * 0.1],
                          levels=boundaries_10p,
                          vmax=v_max, vmin=v_min,
                          colors=["w", "purple"])

    offset_ = 0
    # Label contour lines
    if label_contour_lines:
        color_ = "w"
        ax_mid_point = np.array([sum(ax.get_xlim())/2, sum(ax.get_ylim())/2])

        ax_midpoint = np.array([sum(ax.get_xlim())/2, sum(ax.get_ylim())/2])

        for ix_, c in enumerate(contours.levels):
            pc = contours.collections[ix_]
            nearest_ = []
            for path in pc.get_paths():
                nearest_pos = np.argmin(np.linalg.norm(path.vertices - ax_mid_point, axis=1))
                nearest_.append(path.vertices[nearest_pos])

            if not nearest_:
                continue

            nearest_pos = np.argmin(np.linalg.norm(ax_mid_point - nearest_, axis=1))
            nearest_ = nearest_[nearest_pos]
            padding_ = (nearest_ > ax_midpoint) * -2 + 1
            padding_ *= (4, 4)
            va_ = "bottom"
            ha_ = "left"
            if padding_[0] < 0:
                ha_ = "right"

            if padding_[1] < 0:
                va_ = "top"

            ax.annotate(f"{c:0.1f}", nearest_, xytext=padding_, textcoords="offset points", color=color_,
                        ha=ha_, va=va_, fontproperties={"weight": "bold"})
            color_ = "purple"

    samples_ = np.array(list(product(data.index, data.columns)))
    ax.scatter(x=samples_[:, 1], y=samples_[:, 0], marker="x", color="gray")

    if mark_values is not None:
        ax.scatter(x=[mark_values[data.columns.name]], y=[mark_values[data.index.name]], marker="x", color="red")

    draw_ascend_line = plot_props.get("draw_ascend_line", False)
    if draw_ascend_line:
        color = 2
        for traces, label in zip(
                (max_descend(interpolated_data.values), max_ascent(interpolated_data.values)),
                ("Fastest descend", "Fastest ascend")):
            color += 1
            for trace in traces:
                path_data = np.vstack(trace)
                ax.plot(x[path_data[:, 1]], y[path_data[:, 0]], c=rcParams["axes.prop_cycle"].by_key()["color"][color],
                        label=label)

    draw_min_max_vector = plot_props.get("draw_min_max_vector", True)
    if draw_min_max_vector:
        max_, min_ = compute_max_min(interpolated_data)
        path_data = np.vstack([max_, min_])
        ax.plot(x[path_data[:, 1]], y[path_data[:, 0]], c=rcParams["axes.prop_cycle"].by_key()["color"][3])

    if color_bar:
        cb = plt.colorbar(color_map, cax=color_bar_axes, ax=ax,  ticks=MaxNLocator(5, integer=True))

        # cb._long_axis().set_major_locator()

        cb.ax.axhline(boundaries_10p[0], lw=2.,  color="w")
        cb.ax.axhline(boundaries_10p[1], lw=2.,  color="purple")

        cb.set_label(color_bar_label, va="top", ha="center")


    shared_x = plot_props.get("sharex", False)
    shared_y = plot_props.get("sharey", False)
    if shared_x:
        ax.xaxis.set_ticklabels([])
    else:
        ax.set_xlabel(x_var.capitalize())

    if shared_y:
        ax.yaxis.set_ticklabels([])
    else:
        label = r"\textbf{" + variable.capitalize() + f"}}\n{y_var.capitalize()}" if add_variable_label else y_var.capitalize()
        ax.set_ylabel(label)

    return ax


def interpolate_data(data, x_var, y_var, num=None):
    melted_ap = data.stack().reset_index()
    data = data.fillna(np.nanmin(data.values))  # Interpolation does not support NAN Values
    f = RectBivariateSpline(data.index, data.columns, data.values)
    step_x = step_y = .1
    if num is not None:
        step_x = (max(melted_ap[x_var]) - min(melted_ap[x_var])) / num
        step_y = (max(melted_ap[y_var]) - min(melted_ap[y_var])) / num

    x = np.arange(min(melted_ap[x_var])-0.05, max(melted_ap[x_var])+step_x/2 + 0.05, step=step_x)
    y = np.arange(min(melted_ap[y_var])-0.05, max(melted_ap[y_var])+step_y/2 + 0.05, step=step_y)

    interpolated_data = pd.DataFrame(f(x, y), columns=x, index=y)
    return interpolated_data


def partition_dataset(dataset, levels):
    def enumerate_group(g):
        count = np.arange(len(g))
        count = pd.Series(count, index=g.index)
        return count

    selector = dataset.groupby(levels).apply(enumerate_group)
    return dataset.groupby(selector.values)


def compute_simplex(data, scale):
    # inspired form here: https://github.com/marcharper/python-ternary#rgba-colors
    # r * w = w - y_C
    # y_c = w - r * w
    # y_c = w(1-r)
    # y = scale * y_c / w
    # simplex = (1 - data) * scale

    missing_factor = 1 - data.sum(axis=1) / scale
    simplex = data * (1 + missing_factor).reshape(-1, 1)
    return simplex


def max_descend(data):
    max_, min_ = compute_max_min(data)
    return follow_gradient(data, max_, min_, np.nanmin)


def max_ascent(data):
    max_, min_ = compute_max_min(data)
    return follow_gradient(data, min_, max_, np.nanmax)


def compute_max_min(data):
    min_ = data.min().min()
    max_ = data.max().max()
    min_ = np.hstack(np.where(data == min_)).reshape(-1, 2)
    max_ = np.hstack(np.where(data == max_)).reshape(-1, 2)
    return max_, min_


def follow_gradient(data, sources, targets, direction):
    # we will consider multiple starting points
    traces = []
    specs = product(sources, targets)
    for pos_source, pos_target in specs:
        pos = pos_source
        concurrent_trace = [pos.copy()]
        traces.append(concurrent_trace)
        while not (pos == pos_target).all():
            window = get_window_at_pos(data, pos)
            max_val = direction(window)
            max_pos = np.hstack(np.where(window == max_val)) - [1, 1]

            if (max_pos == [0, 0]).all():
                break

            pos += max_pos
            concurrent_trace.append(pos.copy())

    return traces


def get_window_at_pos(data, pos):
    """Get all the points surrounding pos. this returns a 9x9 matrix. nans are used on the borders."""
    pos = np.array(pos)
    top_left_edge = pos - 1
    bottom_right_edge = pos + 1

    window_slice_r = [0, 3]
    window_slice_c = [0, 3]

    if pos[0] == 0:
        top_left_edge[0] = pos[0]
        window_slice_r[0] = 1

    if pos[0] == data.shape[0] - 1:
        bottom_right_edge[0] = pos[0]
        window_slice_r[1] = 2

    if pos[1] == 0:
        top_left_edge[1] = pos[1]
        window_slice_c[0] += 1

    if pos[1] == data.shape[1] - 1:
        bottom_right_edge[1] = pos[1]
        window_slice_c[1] -= 1

    dataa_slice_r = [top_left_edge[0], bottom_right_edge[0]+1]
    dataa_slice_c = [top_left_edge[1], bottom_right_edge[1]+1]

    window = np.full((3, 3), np.nan)
    data_window = data[slice(*dataa_slice_r), slice(*dataa_slice_c)]

    window[slice(*window_slice_r), slice(*window_slice_c)] = data_window

    return window
