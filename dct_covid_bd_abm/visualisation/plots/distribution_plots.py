import matplotlib
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

from dashboard.plots import draw_daily_distribution, paired_violin_plot
from dct_covid_bd_abm.simulator.analysis_utils.activities import get_activity_duration
from dct_covid_bd_abm.simulator.analysis_utils.data_management import get_variable_from_experiments, dict2df, \
    compute_daily_activity_durations
from dct_covid_bd_abm.simulator.analysis_utils.pathogen import get_serial_intervals, get_serial_intervals_symptomatic_only, \
    get_generation_intervals, get_incubation_periods, get_wave_duration, get_affected_population
from dct_covid_bd_abm.simulator.contact_utils.tomori_model import generate_tomori_total_contacts_all_experiments
from dct_covid_bd_abm.simulator.pathogen_utils import references
from dct_covid_bd_abm.simulator.stats.comparison_tools import distribution_similarity


def draw_distribution_plot(data_dir, variable_getter, data_frame_processor=None, refs=None,
                           ax=None, baseline_dir=None, draw_refs=False, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if baseline_dir is not None:
        baseline_dict = get_variable_from_experiments(baseline_dir, variable_getter)
        data_frame = dict2df(baseline_dict, replace_infs=True, fill_value=np.nan, remove_words=["validation_"])
        if data_frame_processor is not None:
            data_frame = data_frame_processor(data_frame)

        for k, data_ in data_frame.items():
            draw_daily_distribution(data_, ax=ax, label=k, hist=False, **kwargs)

    data_dict = get_variable_from_experiments(data_dir, variable_getter)
    data_frame = dict2df(data_dict, replace_infs=True, fill_value=np.nan, remove_words=["validation_"])
    if data_frame_processor is not None:
        data_frame = data_frame_processor(data_frame)

    for k, data_ in data_frame.items():
        draw_daily_distribution(data_, ax=ax, hist=False, color="gray", label=k, kde_kws=dict(alpha=0.5, lw=1, ),
                                **kwargs)

    if draw_refs:
        draw_daily_distribution([], ax=ax, reference_lines=refs, **kwargs)

    return ax


def stack1(df):
    return df.stack(1)


def serial_interval_distribution_plot(data_dir, ax=None, baseline_dir=None, draw_refs=False, **kwargs):
    refs = [references.serial_interval_he_et_al()]
    ax = draw_distribution_plot(data_dir, get_serial_intervals, ax=ax, baseline_dir=baseline_dir,
                                axlabel="Serial Interval [d]",
                                data_frame_processor=stack1,
                                refs=refs, draw_refs=draw_refs, **kwargs)
    return ax


def serial_interval_wo_asymptomatics_distribution_plot(data_dir, ax=None, baseline_dir=None, draw_refs=False, **kwargs):
    # Load data
    refs = [references.serial_interval_he_et_al()]
    ax = draw_distribution_plot(data_dir, get_serial_intervals_symptomatic_only, ax=ax, baseline_dir=baseline_dir,
                                data_frame_processor=stack1,
                                refs=refs, draw_refs=draw_refs, **kwargs)
    return ax


def generation_interval_distribution_plot(data_dir, ax=None, baseline_dir=None, draw_refs=False, **kwargs):
    # Load Data
    refs = [references.generation_interval_ganyani_et_al()]
    ax = draw_distribution_plot(data_dir, get_generation_intervals, ax=ax, baseline_dir=baseline_dir,
                                data_frame_processor=stack1,
                                truncate=[0, None], xlim=[-2, 22],
                                axlabel="Generation Interval [d]",
                                refs=refs, draw_refs=draw_refs, **kwargs)
    return ax


def incubation_period_distribution_plot(data_dir, ax=None, baseline_dir=None, draw_refs=False, **kwargs):
    refs = [references.incubation_duration_distribution_lauer_et_al(),
            references.incubation_duration_distribution_li_et_al()]

    ax = draw_distribution_plot(data_dir, get_incubation_periods, ax=ax, baseline_dir=baseline_dir,
                                data_frame_processor=stack1,
                                axlabel="Incubation Period [d]",
                                refs=refs, draw_refs=draw_refs, **kwargs)
    return ax


def wave_duration_distribution_plot(data_dir, ax=None, baseline_dir=None, draw_refs=False, **kwargs):
    ax = draw_distribution_plot(data_dir, get_wave_duration, ax=ax, baseline_dir=baseline_dir,
                                data_frame_processor=stack1,
                                axlabel="Wave Duration [d]",
                                draw_refs=draw_refs, **kwargs)
    return ax


def affected_population_distribution_plot(data_dir, ax=None, baseline_dir=None, draw_refs=False, **kwargs):
    def df_processor(df):
        return df.stack(1).unstack(0).groupby(level=0, axis=1).sum()

    ax = draw_distribution_plot(data_dir, get_affected_population, ax=ax, baseline_dir=baseline_dir,
                                data_frame_processor=df_processor,
                                axlabel="Total infected",
                                truncate=[0, 1000],
                                draw_refs=draw_refs, **kwargs)
    return ax


def create_activity_plot(data_dir, activity_ax):
    si = get_variable_from_experiments(data_dir, get_activity_duration)
    activity_timing = dict2df(si, remove_words=["validation_"])
    activity_timing = activity_timing.stack(level=1).reset_index(drop=True)
    activity_timing[activity_timing == 0] = np.nan
    activity_timing.columns.names = ["Experiment", "Activity"]
    ad = activity_timing.melt()
    ad["Experiment"] = ad["Experiment"].astype("category")
    ad["Activity"] = ad["Activity"].astype("category")
    sb.violinplot(ax=activity_ax, x="Activity", y="value", data=ad, hue="Experiment")


def activity_instance_duration_distributions(data_, other_data=None, other_data_name=None):
    dfs = [data_
           # .groupby("activity")
           # .sample(500)
           # .reset_index(drop=True)
           ]

    keys = ["I2MB"]

    if other_data is not None:
        dfs.append(other_data.set_index(["activity"], append=True)["duration"]
                   .reset_index()
                   .reset_index(drop=True))
        keys.append(other_data_name)

    duration = pd.concat(dfs, keys=keys, names=["Source", "Values"], axis=1).stack(0).reset_index(1)
    duration_ = duration["duration"] != 0
    duration = duration[duration_]

    activity_ttests = distribution_similarity(duration, other_data_name)
    print("Daily instance duration per person\n", activity_ttests)

    paired_violin_plot(duration.sort_values(["activity", "Source"]), "Instance duration per person")


def daily_activity_duration_distributions(data_, other_data=None, other_data_name=None, ax=None, logged_day=False,
                                          **kwargs):
    daily_duration = compute_daily_activity_durations(data_, other_data, other_data_name, logged_day=logged_day)

    # Truncate total commulative to 24 H
    daily_duration.loc[daily_duration["duration"] > 60 * 24, "duration"] = 60 * 24
    ax = paired_violin_plot(daily_duration.sort_values(["activity", "Source"]),
                            "Daily duration [hh:mm]", ax=ax, **kwargs)

    return ax


def daily_activity_instances_distributions(data_, other_data=None, other_data_name=None):
    select_columns = ["id", "day", "activity"]
    if "run" in data_.columns:
        select_columns = ["id", "run", "day", "activity"]

    dfs = [data_.groupby(select_columns)
           .count()["duration"]
           .reset_index()
           # .groupby("activity")
           # .sample(500)
           # .reset_index(drop=True)
           ]

    keys = ["I2MB"]

    if other_data is not None:
        dfs.append(other_data.groupby(["id", "day", "activity"], observed=True)
                   .count()["duration"]
                   .reset_index()
                   .reset_index(drop=True))
        keys.append(other_data_name)

    daily_instances = pd.concat(dfs, keys=keys, names=["Source", "Values"], axis=0).reset_index(0).reset_index(drop=True)
    duration_ = daily_instances["duration"] != 0
    daily_instances = daily_instances[duration_]

    activtiy_ttests = distribution_similarity(daily_instances, other_data_name)
    print("Daily Instances per Person\n", activtiy_ttests)

    paired_violin_plot(daily_instances.sort_values(["activity", "Source"]),
                       "Daily instances per person")


def contact_validation_boxplot(contact_data_unique_complete, ax=None, mode=None):
    if mode is None:
        mode = "Overall"

    tomori_gen_samples = generate_tomori_total_contacts_all_experiments(10000)
    tomori_gen_samples = (pd.concat(tomori_gen_samples.values(), keys=tomori_gen_samples)
                          .loc[(slice(None), mode), :]
                          .droplevel(1)
                          .unstack(0)
                          .swaplevel(0, 1, axis=1))

    if mode == "Overall":
        mode = slice(None)

    # tomori_gen_samples.index = tomori_gen_samples.index.remove_unused_levels()
    i2mb_overall_samples = (contact_data_unique_complete
                            .loc[(slice(None), mode, slice(None), slice(None)), :]
                            .groupby(level=[0, 2, 3])
                            .sum(min_count=1)
                            .reset_index(drop=True)
                            )

    i2mb_overall_samples.columns = pd.MultiIndex.from_product([["I2MB"], i2mb_overall_samples.columns])

    overall_samples = pd.concat([i2mb_overall_samples, tomori_gen_samples], axis=1)
    overall_samples.columns.names = ["Source", "Study"]
    overall_samples = (overall_samples.melt(value_name="Contact Persons", ignore_index=True)
                       .dropna()
                       .set_index("Source")
                       .loc[["I2MB",
                             "Unweighted",
                             "Weighted",
                             "Unweighted and With Group Contacts",
                             "Weighted and With Group Contacts"], :]
                       .reset_index("Source", drop=False))
    sb.boxplot(x="Study", y="Contact Persons", data=overall_samples,
               hue="Source",
               ax=ax,
               showmeans=True,
               meanprops={"marker": "o",
                          "markerfacecolor": "white",
                          "markeredgecolor": "black",
                          "markersize": "5"},
               # palette=['#4878d0', '#FFBCB2', '#FF8774', '#EC2A0B', '#B61900'],
               # palette=['#4878d0', '#fc8d62', '#fdaf91', '#fec6b1', '#feddd0'],
               # palette=['#4878d0', '#FAFEB1', '#F6FD73', '#DFEA0B', '#ABB400'],
               palette=['#4878d0', '#F6FD73', '#fec6b1', '#ABB400', '#fc8d62'],
               flierprops={"markersize": "3"},
               linewidth=1.,
               )

    ax.set_xticklabels(ax.get_xticklabels())
    ax.set_yscale("symlog")
    ax.set_ylim(0, 600)
    ax.set_yticks([1, 2, 5, 20, 50, 200, 500])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Simulated study conditions")
    ax.set_ylabel("Contact persons")

    handles, legend_labels = ax.get_legend_handles_labels()
    legend_labels = [l.replace("Group Contacts", "GC") for l in legend_labels]
    r = matplotlib.patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none', visible=False)
    legend_labels.insert(1, "")
    handles.insert(1, r)
    legend_labels = [l.replace("and With", "with") for l in legend_labels]
    ax.legend(handles, legend_labels, loc="lower left", title_fontsize="small", fontsize="small",
              ncol=3,
              bbox_to_anchor=(0, 1.025, 1, 1.2),
              borderaxespad=0.0,
              mode="expand")
    ax.get_figure().text(0.01, 0.15, "Log scale", fontsize="x-small", ha="left", va="center")

