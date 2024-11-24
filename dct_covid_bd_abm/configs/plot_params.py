import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

from dct_covid_bd_abm.simulator.pathogen_utils import references

defaultRadarKwargs = {
    "selected_lw": 1.6,
    "color": [cm.Blues(0.5)],
}
stage1RadarPlotProperties = dict(
    perfect_behaviour={
        "title": "Perfect Behaviour",
        "data_dir": ["stage_1", "stage_2_perfect_behaviour"],
        "data_filters": [None, ["Both No Rnt No Qch Bnr"]],
        "selected": ["With Ict Mct", "Both No Rnt No Qch Bnr"],
        "cm": LinearSegmentedColormap.from_list("Truncated Greens", cm.Greens(np.linspace(0.5, 0.9, 10)))
    },
    coverage={
        "title": "Adoption",
        "data_dir": "stage_1_coverage_boundary",
        "selected": "Coverage 0.8",
        "refs": ["perfect_behaviour", "dropout"],
        "cm": LinearSegmentedColormap.from_list("Truncated Blues", cm.Blues(np.linspace(0.35, 0.9, 10)))
    },
    dropout={
        "title": "No-Compliance",
        "data_dir": "stage_1_ct_dropout_boundary",
        "data_filters": [[f"Dropout {dr}" for dr in [0.01, 0.05, 0.1, 0.2, 0.5, 0.75]]],
        "selected": "Dropout 0.2",
        "refs": ["perfect_behaviour"],
        "cm": LinearSegmentedColormap.from_list("Truncated Purples", cm.Purples(np.linspace(0.35, 1, 10))).reversed()
    },
    dct_intro={
        "title": "DCT Introduction",
        "data_dir": "stage_1_dct_introduction_time",
        "selected": "Introduction Time 0.1",
        "refs": ["perfect_behaviour", "rs_boundary"],
        "cm": LinearSegmentedColormap.from_list("Truncated Greys", cm.Greys(np.linspace(0.35, 1, 10))).reversed()
    },
    rs_boundary={
        "title": "Shared Test Results",
        "data_dir": "stage_1_result_sharing_boundary",
        "selected": "Result Sharing 0.8",
        "refs": ["perfect_behaviour", "coverage"],
        "cm": LinearSegmentedColormap.from_list("Truncated Reds", cm.Reds(np.linspace(0.35, 0.9, 10)))
    },
)
stage2DataPlotProperties = dict(
    perfect_behaviour={
        "data_dir": ["stage_1", "stage_2_perfect_behaviour"],
        "data_filters": [[
                          # ("None", "No Rnt", "No Qch", "Bnr"),
                          # ("Ict", "No Rnt", "No Qch", "Bnr"),
                          ("Mct", "No Rnt", "No Qch", "Bnr"),
                          ],
                          [("Both", "No Rnt", "Qch", "Bnr"),
                          ("Both", "No Rnt", "Qch", "No Bnr"),
                          ("Both", "No Rnt", "No Qch", "No Bnr"),
                          ("Both", "No Rnt", "No Qch", "Bnr"),
                          ("Both", "Rnt", "Qch", "No Bnr"),
                          ("Both", "Rnt", "No Qch", "Bnr"),
                          ("Both", "Rnt", "No Qch", "No Bnr"),
                          ("Both", "Rnt", "Qch", "Bnr"),

                           ]
                        # ("Both", slice(None), slice(None), slice(None))
                         ],
        "rename_index": [[slice(0, 3), ["None No Rnt No Qch Bnr",
                                        "Ict No Rnt No Qch Bnr",
                                        "Mct No Rnt No Qch Bnr"]], None]
    },

    parameter_search={
        "data_dir": ["stage_2_best_main_parameters"],
        "data_filters": [(["Both", "Mct"], slice(None), slice(None), slice(None)), slice(None)
                         ]
    }
)

__cm_dct_optimal = LinearSegmentedColormap.from_list("Truncated Greens", cm.Greens(np.linspace(0.5, 0.9, 10)))
__cm_baseline = LinearSegmentedColormap.from_list("Truncated Greens", cm.Oranges(np.linspace(0.5, 0.9, 3)))

stage2PanelProperties = dict(
    perfect_behaviour={
        "title": "Perfect Behaviour",
        "cm": [__cm_baseline(0.675)] + [__cm_dct_optimal(i) for i in np.linspace(0, 1, 8)],

        "fill_between_cm": [__cm_dct_optimal(0)],
        "fill_between_index": ("Dct", slice(None), slice(None), slice(None)),

        # Discrimination works on renamed index labels
        "discrimination": [("Dct", slice(None), slice(None), slice(None)),
                           (["None", "Ict", "No Dct"], slice(None), slice(None), slice(None))],

        # Selected works on renamed index labels
        "selected": [("No Dct", "No Rnt", "No Qch", "No Cbr"), ("Dct", "No Rnt", "No Qch", "No Cbr")]
    },

    dct_to_no_dct={
        "title": "DCT Effect",
        "cm": [LinearSegmentedColormap.from_list("Truncated Blues", cm.Blues(np.linspace(0.35, 0.9, 10)))(0.5),
               LinearSegmentedColormap.from_list("Truncated Reds", cm.Reds(np.linspace(0.35, 0.9, 10)))(0.5)],
        "discrimination": [("Dct", slice(None), slice(None), slice(None)),
                           ("No Dct", slice(None), slice(None), slice(None))],
        "refs": ["perfect_behaviour"]
    },

    rnt_to_no_rnt={
        "title": "RNT Effect with DCT",
        "cm": [LinearSegmentedColormap.from_list("Truncated Purple", cm.Blues(np.linspace(0.35, 0.9, 10)))(0.5),
               LinearSegmentedColormap.from_list("Truncated Oranges", cm.Reds(np.linspace(0.35, 0.9, 10)))(0.5)],
        "discrimination": [("Dct", "No Rnt", slice(None), slice(None)),
                           ("Dct", "Rnt", slice(None), slice(None))],
        "refs": ["perfect_behaviour"]
    },
    qch_to_no_qch={
        "title": "CBR - QCH Effect with DCT and without RNT",
        "cm": [LinearSegmentedColormap.from_list("Truncated Purple", cm.Blues(np.linspace(0.89, 0.9, 10))),
               LinearSegmentedColormap.from_list("Truncated Oranges", cm.Reds(np.linspace(0.89, 0.9, 10)))],

        "discrimination": [[("Dct", "No Rnt", "No Qch", "Cbr")],
                           [("Dct", "No Rnt", "Qch", "Cbr")]],
        "refs": ["perfect_behaviour"]
    },
    lbr_to_no_lbr={
        "title": "CBR Effect with DCT and no RNT",
        "cm": [LinearSegmentedColormap.from_list("Truncated Purple", cm.Blues(np.linspace(0.35, 0.9, 10))),
               LinearSegmentedColormap.from_list("Truncated Oranges", cm.Reds(np.linspace(0.35, 0.9, 10)))],
        "discrimination": [("Dct", "No Rnt", slice(None), "Cbr"),
                           ("Dct", "No Rnt", slice(None), "No Cbr")],
        "refs": ["perfect_behaviour"]
    },
    best_performance={
        "title": "Best combination",
        "cm": LinearSegmentedColormap.from_list("Truncated Purple", cm.Blues(np.linspace(0.8, 0.9, 10))),
        "lw": 2.5,
        "discrimination": [[("Dct", "No Rnt", "No Qch", "Cbr")],
                           ],
        "refs": ["perfect_behaviour"]
    }

)
validationPanelProperties = {
    "Serial Interval": {"refs": [references.serial_interval_he_et_al()]},
    "Generation Interval": {"refs": [references.generation_interval_ganyani_et_al()]},
    "Incubation Period": {"refs": [references.incubation_duration_distribution_lauer_et_al()]},
    "Illness Duration": {"refs": [references.clearance_period_kissler_et_al()],
                         "rename": "Clearance Period"}
}


def get_variable_units(variable):
    """
    Retrieves a variable's units. If variable is a list, returns the list of variables. If the variable is unknown,
    :meth:`get_variable_units` returns an empty string.
    Parameters
    ----------
    variable : str|lis[str]
        Name of the variable for which we want to retrieve the units.

    Returns
    -------
    units: str | list[units]
        Units of provided variable(s).

    """
    variable_units = {
        'Generation Interval': "[d]",
        'Serial Interval': "[d]",
        'Wave Duration': "[d]",
        'Incubation Period': "[d]",
        'Illness Duration': "[d]",
        'Days in Isolation': "[d]",
        'Total infected': "[%]",
        '7-day Incidence': "[cases/100,000 pop.]",
        '7-day Hosp. Incidence': "[cases/100,000 pop.]",
        'Days in Quarantine': "[d]",
    }

    if isinstance(variable, str):
        return variable_units.get(variable, "")

    return [variable_units.get(v, "") for v in variable]
