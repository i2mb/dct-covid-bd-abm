import os
import shutil
import tempfile
from logging import warning

import matplotlib

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from dct_covid_bd_abm.visualisation.plots.heatmaps import coverage_vs_compliance, coverage_vs_result_sharing, \
    result_sharing_vs_compliance, grid_ternary_heat_map
from dct_covid_bd_abm.visualisation.plots.panels import stage_1_spider_plot, stage_2_spider_plot, validation_panel, grid_search_panel, \
    grid_search_panel_hm, stage_2_parallel_plot, stage_2_parallel_areas, viral_model_innards, contact_validation, \
    network_metric_distribution, source_target_degree_strength, daily_nx_metrics, network_shapes_by_contact_type, \
    activity_calibration, viral_model_calibration, metric_analysis_plot
from dct_covid_bd_abm.visualisation.plots.spatial_figures import draw_scenario
from dct_covid_bd_abm.visualisation.tables.activities import activity_validation_table
from dct_covid_bd_abm.visualisation.tables.contacts import contact_validation_table, rules_contact_validation_table
from dct_covid_bd_abm.visualisation.tables.values_in_text import compute_values_in_text

BASE_DIRECTORY = Path("./figures")
DATA_DIRECTORY = Path("./data/simulations")
POLYMOD_DATA_DIRECTORY = "./data/contact_validation/POLYMOD/test"
CONTACT_VALIDATION_DATA_DIRECTORY = "./data/contact_validation/"
NETWORK_ANALYSIS_DATA_DIRECTORY = Path("./data/simulations")
FORMAT = "png"
TAB_FORMAT = "xlsx"

matplotlib.use('Qt5Agg')


def save_figures(figures):
    for img_name, fig in figures.items():
        if fig is None:
            print(f"{img_name} is externally created")
            continue

        image_path = BASE_DIRECTORY / f"{img_name}.{FORMAT}"
        if os.path.exists(image_path):
            os.remove(image_path)

        print(f"Saving: {image_path}")
        fig.savefig(image_path, format=FORMAT, transparent=True, bbox_inches='tight', pad_inches=0.1)


def save_tables_csv(tables):
    for tab_name, tab in tables.items():
        tab_path = BASE_DIRECTORY / f"{tab_name}.csv"
        tab.to_csv(tab_path)


def save_tables_xlsx(tables):
    if not tables:
        return

    spread_sheet_path = BASE_DIRECTORY / "tables.xlsx"
    temp_spread_sheet_path = Path(tempfile.gettempdir()) / "tables.xlsx"
    try:
        saved_tables = pd.read_excel(spread_sheet_path, sheet_name=None)
        tables = saved_tables | tables
        os.remove(spread_sheet_path)
    except FileNotFoundError:
        pass

    with pd.ExcelWriter(temp_spread_sheet_path) as writer:
        for tab_name, tab in tables.items():
            print(f"Saving: {tab_name}")
            tab.to_excel(writer, sheet_name=tab_name, float_format="%.1f")

    shutil.move(temp_spread_sheet_path, BASE_DIRECTORY, shutil.copyfile)


def save_tables(tables):
    if TAB_FORMAT == "csv":
        save_tables_csv(tables)
        return

    if TAB_FORMAT == "xlsx":
        save_tables_xlsx(tables)
        return

    warning(f"Format {TAB_FORMAT} not supported. Skipping!")


def main_text_images():
    main_figures = (("grid_search_panel", (grid_search_panel_hm, (DATA_DIRECTORY, False, True), {})),
                    ("parallel_areas", (stage_2_parallel_areas, (DATA_DIRECTORY,), {})),
                    ("zeitgeber_timeslots", (lambda x: None, (DATA_DIRECTORY,), {})),
                    ("viral_model_innards", (viral_model_innards, (), {}))
                    )
    main_fig_dict = {}
    for num, (fig_name, fig) in zip(range(2, len(main_figures) + 2), main_figures):
        main_fig_dict[f"{num}_{fig_name}"] = fig

    return main_fig_dict


def supplement_text_images():
    scalers = {
        "infection time": 1 / ((60 / 5) * 24),  # day
        "source strength": 1 / ((60 / 5) * 24),
        "target strength": 1 / ((60 / 5) * 24),
        "source strength pre infection": 1 / ((60 / 5) * 24),
        "target strength pre infection": 1 / ((60 / 5) * 24),
        "source 7-day strength pre infection": 1 / ((60 / 5) * 24),
        "target 7-day strength pre infection": 1 / ((60 / 5) * 24)
    }
    columns_with_units = {
        "infection time": "Infection time [d]",
        "7-day incidence": "7-day incidence [cases]",
        "source degree": "Source degree [CP]",
        "source strength": "Source strength [d]",
        "target degree": "Target degree [CP]",
        "target strength": "Target strength [d]",
        "source degree pre infection": "Source degree pre infection [CP]",
        "source strength pre infection": "Source strength pre infection [d]",
        "target degree pre infection": "Target degree pre infection [CP]",
        "target strength pre infection": "Target strength pre infection [d]",
        "source 7-day degree pre infection": "Source 7-day degree pre infection [CP]",
        "source 7-day strength pre infection": "Source 7-day strength pre infection [d]",
        "target 7-day degree pre infection": "Target 7-day degree pre infection [CP]",
        "target 7-day strength pre infection": "Target 7-day strength pre infection [d]",
    }
    supp_visuals = (

        #
        ("world", (draw_scenario, (), dict(population_size=100,
                                           config_file="./dct_covid_bd_abm/configs/base_configuration.py",
                                           inventory=False))),
        ("world_infrastructure", (draw_scenario, (), dict(population_size=100,
                                                          config_file="./dct_covid_bd_abm/configs/base_configuration.py",
                                                          world=False))),

        # Activity validation
        ("activity_calibration_ld", (activity_calibration, (POLYMOD_DATA_DIRECTORY,),
                                     dict(horizontal=True, logged_day=True))),

        # Contact calibration
        ("tomori_overall", (contact_validation, (CONTACT_VALIDATION_DATA_DIRECTORY, "Overall"), {})),
        ("tomori_home", (contact_validation, (CONTACT_VALIDATION_DATA_DIRECTORY, "Home"), {})),
        ("tomori_work", (contact_validation, (CONTACT_VALIDATION_DATA_DIRECTORY, "Work"), {})),
        ("tomori_transport", (contact_validation, (CONTACT_VALIDATION_DATA_DIRECTORY, "Transport"), {})),
        ("tomori_others", (contact_validation, (CONTACT_VALIDATION_DATA_DIRECTORY, "Others"), {})),

        # Viral model calibration
        ("viral_model_calibration", (viral_model_calibration, (DATA_DIRECTORY / "stage_1",),
                                     dict(legend_kwg=dict(ncol=4, bbox_to_anchor=(0.05, 0.91, 0.9, 0.99))))),

        # # Network dynamics
        ("network_structures", (network_shapes_by_contact_type, (POLYMOD_DATA_DIRECTORY,), {})),
        ("network_metric_distribution", (network_metric_distribution, (NETWORK_ANALYSIS_DATA_DIRECTORY,), {})),

        ("network_infection_time_vs_nx_metrics",
         (source_target_degree_strength,
          (["source degree", "source strength", "target degree", "target strength"], "infection time"),
          dict(scalers=scalers, columns_with_units=columns_with_units))),
        ("network_infection_time_vs_nx_metrics_pre_infection",
         (source_target_degree_strength,
          (), dict(var_x=["source degree pre infection", "source strength pre infection",
                          "target degree pre infection", "target strength pre infection"],
                   var_y="infection time", scalers=scalers, columns_with_units=columns_with_units))),

        ("network_infection_time_vs_nx_metrics_7_day_pre_infection",
         (source_target_degree_strength,
          (), dict(var_x=["source 7-day degree pre infection", "source 7-day strength pre infection",
                          "target 7-day degree pre infection", "target 7-day strength pre infection"],
                   var_y="infection time", scalers=scalers, columns_with_units=columns_with_units))),
        ("network_7_day_incidence_vs_nx_metrics_7_day_pre_infection",
         (source_target_degree_strength,
          (), dict(var_x=["source 7-day degree pre infection", "source 7-day strength pre infection",
                          "target 7-day degree pre infection", "target 7-day strength pre infection"],
                   var_y="7-day incidence", scalers=scalers, columns_with_units=columns_with_units))),


        # ("network_daily_nx_metrics", (daily_nx_metrics, ("Baseline",), dict(remove_hospital_contacts=False))),
        ("network_daily_nx_metrics_wo_hospital", (daily_nx_metrics, ("Baseline",),
                                                  {"stage_name": "Baseline"})),
        ("network_daily_nx_metrics_wo_hospital_ict_mct", (daily_nx_metrics, ("MCT and ICT",), {})),
        ("network_daily_metrics_imperfect_dct", (daily_nx_metrics, ("Imperfect DCT",),
                                                 {"stage_name": "Realistic behaviour"})),
        ("network_daily_metrics_perfect_dct", (daily_nx_metrics, ("Perfect DCT",),
                                               {"stage_name": "DCT-optimal behaviour"})),

        ("fdr_fnr_metric_analysis_ti", (metric_analysis_plot, (DATA_DIRECTORY, ["False negative rate", "False discovery rate"]),
                             {"order_by": ["Total infected", "Fowlkes Mallows index"]})),
        # ("fdr_fnr_metric_analysis_fmi",
         # (metric_analysis_plot, (DATA_DIRECTORY, ["False negative rate", "False discovery rate"]),
         #     {"order_by": "Fowlkes Mallows index"})),

    )


    supp_visuals_dict = {}
    for num, (fig_name, fig) in zip(range(1, len(supp_visuals) + 1), supp_visuals):
        supp_visuals_dict[f"S{num}_{fig_name}"] = fig

    return supp_visuals_dict


def other_plots():
    other_visuals = (
        (stage_1_spider_plot, (DATA_DIRECTORY,), {}),
        (stage_2_spider_plot, (DATA_DIRECTORY,), dict(reverse_order=True)),
        (stage_2_parallel_plot, (DATA_DIRECTORY,), dict(reverse_order=True)),
        (coverage_vs_compliance, (DATA_DIRECTORY, "Total infected"), {}),
        (coverage_vs_result_sharing, (DATA_DIRECTORY, "Total infected"), {}),
        (result_sharing_vs_compliance, (DATA_DIRECTORY, "Total infected"), {}),
        (grid_search_panel, (DATA_DIRECTORY,), {}),
        (grid_search_panel_hm, (DATA_DIRECTORY, False), {}),
        (grid_search_panel_hm, (DATA_DIRECTORY, True), {}),
        (grid_search_panel_hm, (DATA_DIRECTORY, True, True), {}),
        (grid_ternary_heat_map, (DATA_DIRECTORY,), {}),
        (validation_panel, (DATA_DIRECTORY / "stage_1", POLYMOD_DATA_DIRECTORY, CONTACT_VALIDATION_DATA_DIRECTORY), {})
    )
    return other_visuals


def render_visual(fig_gen):
    func, args, kwargs = fig_gen
    return func(*args, **kwargs)


def main_text_tables():
    return {}


def supplement_text_tables():
    tables = (
        ("activity_validation_table", (activity_validation_table, (POLYMOD_DATA_DIRECTORY,), {})),
        ("rules_contact_validation_table", (rules_contact_validation_table, (), {})),
        ("tomori_overall_table", (contact_validation_table, (CONTACT_VALIDATION_DATA_DIRECTORY, "Overall"), {})),
        ("tomori_home_table", (contact_validation_table, (CONTACT_VALIDATION_DATA_DIRECTORY, "Home"), {})),
        ("tomori_work_table", (contact_validation_table, (CONTACT_VALIDATION_DATA_DIRECTORY, "Work"), {})),
        ("tomori_transport_table", (contact_validation_table, (CONTACT_VALIDATION_DATA_DIRECTORY, "Transport"), {})),
        ("tomori_others_table", (contact_validation_table, (CONTACT_VALIDATION_DATA_DIRECTORY, "Others"), {})),
        ("FP_FN_table_dct-optimal", (compute_values_in_text, (DATA_DIRECTORY, "perfect_behaviour"),
                                     {"metrics": ["False discovery rate", "False negative rate", "Fowlkes Mallows index"]})),
        ("FP_FN_table_dct-realistic", (compute_values_in_text, (DATA_DIRECTORY, "parameter_search"),
                                       {"metrics": ["False discovery rate", "False negative rate", "Fowlkes Mallows index"]})),

        ("full_metrics_table_dct-optimal", (compute_values_in_text, (DATA_DIRECTORY, "perfect_behaviour"),
                                     {"metrics": [
                                            "Generation Interval",
                                            "Serial Interval",
                                            "7-day Incidence",
                                            "7-day Hosp. Incidence",
                                            # 
                                            # "Incubation Period",
                                            # "Illness Duration",
                                            # "Days in Isolation",
                                            "Total infected",
                                            "Wave Duration",
                                            "Days in Quarantine",
                                            ]})),
        ("full_metrics_table_dct-realistic", (compute_values_in_text, (DATA_DIRECTORY, "parameter_search"),
                                       {"metrics": [
                                           "Generation Interval",
                                           "Serial Interval",
                                           "7-day Incidence",
                                           "7-day Hosp. Incidence",
                                           # 
                                           # "Incubation Period",
                                           # "Illness Duration",
                                           # "Days in Isolation",
                                           "Total infected",
                                           "Wave Duration",
                                           "Days in Quarantine",
                                       ]}))

    )

    supp_tables_dict = {}
    for num, (table_name, tab) in zip(range(1, len(tables) + 1), tables):
        supp_tables_dict[f"S{num}_{table_name}"] = tab

    return supp_tables_dict


def main(figures_names=None, table_names=None):
    if table_names is None:
        table_names = []

    if figures_names is None:
        figures_names = []

    figures = {}
    figures.update(main_text_images())
    figures.update(supplement_text_images())

    fig_to_save = {}
    for fig_name in figures_names:
        fig_to_save[fig_name] = render_visual(figures[fig_name])

    tables = {}
    tables.update(main_text_tables())
    tables.update(supplement_text_tables())

    tab_to_save = {}
    for tab_name in table_names:
        tab_to_save[tab_name] = render_visual(tables[tab_name])

    plt.interactive(True)
    plt.show(block=False)
    plt.pause(2)
    save_ = input("Save figures and tables?").lower()

    plt.close("all")

    if save_ in ["y", "yes"]:

        print("Saving figures")
        save_figures(fig_to_save)

        print("Saving Tables")
        save_tables(tab_to_save)


def list_available_items():
    print("# Main paper")
    print("## Tables")
    for item in main_text_tables():
        print(f"  {item}")

    print("\n## Images")
    for item in main_text_images():
        print(f"  {item}")

    print("\n# Supplemental material\n## Tables")
    for item in supplement_text_tables():
        print(f"  {item}")

    print("\n## Images")
    for item in supplement_text_images():
        print(f"  {item}")


if __name__ == "__main__":
    list_available_items()
    main(figures_names=[
        "2_grid_search_panel",
        "3_parallel_areas",

        "S1_world",
        "S2_world_infrastructure",

        "S3_activity_calibration_ld",
        "S4_tomori_overall",
        "S5_tomori_home",
        "S6_tomori_work",
        "S7_tomori_transport",
        "S8_tomori_others",
        "S9_viral_model_calibration",

        "S10_network_structures",
        "S11_network_metric_distribution",

        "S12_network_infection_time_vs_nx_metrics",
        "S13_network_infection_time_vs_nx_metrics_pre_infection",
        "S14_network_infection_time_vs_nx_metrics_7_day_pre_infection",
        "S15_network_7_day_incidence_vs_nx_metrics_7_day_pre_infection",

        "S16_network_daily_nx_metrics_wo_hospital",
        "S17_network_daily_nx_metrics_wo_hospital_ict_mct",
        "S18_network_daily_metrics_imperfect_dct",
        "S19_network_daily_metrics_perfect_dct",

        "S20_fdr_fnr_metric_analysis_ti"
    ],
        table_names=[
            "S1_activity_validation_table",
            "S2_rules_contact_validation_table",
            "S3_tomori_overall_table",
            "S4_tomori_home_table",
            "S5_tomori_work_table",
            "S6_tomori_transport_table",
            "S7_tomori_others_table",
            "S8_FP_FN_table_dct-optimal",
            "S9_FP_FN_table_dct-realistic",

            "S10_full_metrics_table_dct-optimal",
            "S11_full_metrics_table_dct-realistic"
             ])
    print("I am done")
