import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

from dashboard.plot_utils import overline
from dct_covid_bd_abm.simulator.analysis_utils.data_management import load_stage_data


def spider_perfect_behaviour(base_directory, ax=None, plot_props=None):
    internal_props = {
        "title": "Perfect Behaviour",
        "cm": LinearSegmentedColormap.from_list("Truncated Greens",
                                                cm.Greens(np.linspace(0.5, 0.9, 10))),

        # Selected works on renamed index labels
        "selected": [("No Dct", "No Rnt", "No Qch", "No Lbr"), ("Dct", "Rnt", "No Qch", "Lbr")],
        "data_props": {
            "perfect_behaviour": {
                "data_dir": ["stage_1", "stage_2_perfect_behaviour"],
                "data_filters": [[("Mct", "No Rnt", "No Qch", "Bnr")],
                                 [("Both", "No Rnt", "No Qch", "Bnr"),
                                  ("Both", "Rnt", "No Qch", "No Bnr"),
                                  ("Both", "Rnt", "Qch", "No Bnr")]
                                 ],
                "rename_index": [[slice(0, 3), ["No Ict No Rnt No Qch Bnr",
                                                "None No Rnt No Qch Bnr",
                                                "Mct No Rnt No Qch Bnr"]], None]}},
        "legend": {
            "legend_labels": [
                f"{overline('DCT')}, {overline('MCT')}, {overline('ICT')}",
                f"{overline('DCT')}, {overline('MCT')}, ICT",
                f"{overline('DCT')}, MCT, ICT",
                f"DCT, MCT, ICT"]
        },
    }

    if plot_props is not None:
        internal_props.update(plot_props)

    stage_2_data = load_stage_data(base_directory, internal_props["data_props"], True, prefix="stage_2")
    print(stage_2_data)


def dc_to_no_dct_spider(base_directory, ax=None):
    dct_to_no_dct = {
        "title": "DCT Effect",
        "cm": [
            LinearSegmentedColormap.from_list("Truncated Blues", cm.Blues(np.linspace(0.35, 0.9, 10))),
            LinearSegmentedColormap.from_list("Truncated Reds", cm.Reds(np.linspace(0.35, 0.9, 10)))],
        "discrimination": [("Dct", slice(None), slice(None), slice(None)),
                           ("No Dct", slice(None), slice(None), slice(None))],
        "refs": ["perfect_behaviour"]
    }


def rnt_to_no_rnt_spider(base_directory, ax=None):
    rnt_to_no_rnt = {
        "title": "DCT - RNT Effect",
        "cm": [
            LinearSegmentedColormap.from_list("Truncated Purple", cm.Blues(np.linspace(0.35, 0.9, 10))),
            LinearSegmentedColormap.from_list("Truncated Oranges",
                                              cm.Reds(np.linspace(0.35, 0.9, 10)))],
        "discrimination": [("Dct", "No Rnt", slice(None), slice(None)),
                           ("Dct", "Rnt", slice(None), slice(None))],
        "refs": ["perfect_behaviour"]
    }


def qch_to_no_qch_spider(base_directory, ax=None):
    qch_to_no_qch = {
        "title": "DCT, $\\mathrm{\\mathbf{\\overline{RNT}}}$, LBR - QCH Effect",
        "cm": [
            LinearSegmentedColormap.from_list("Truncated Purple", cm.Blues(np.linspace(0.89, 0.9, 10))),
            LinearSegmentedColormap.from_list("Truncated Oranges",
                                              cm.Reds(np.linspace(0.89, 0.9, 10)))],
        "discrimination": [[("Dct", "No Rnt", "No Qch", "Lbr")],
                           [("Dct", "No Rnt", "Qch", "Lbr")]],
        "refs": ["perfect_behaviour"]
    }


def lbr_to_no_lbr(base_directory, ax=None):
    lbr_to_no_lbr = {
        "title": "DCT, $\\mathrm{\\mathbf{\\overline{RNT}}}$ - LBR Effect",
        "cm": [
            LinearSegmentedColormap.from_list("Truncated Purple", cm.Blues(np.linspace(0.35, 0.9, 10))),
            LinearSegmentedColormap.from_list("Truncated Oranges",
                                              cm.Reds(np.linspace(0.35, 0.9, 10)))],
        "discrimination": [("Dct", "No Rnt", slice(None), "Lbr"),
                           ("Dct", "No Rnt", slice(None), "No Lbr")],
        "refs": ["perfect_behaviour"]
    }


def best_performance_spider(base_directory, ax=None):
    best_performance = {
        "title": "Best combination",
        "cm": [LinearSegmentedColormap.from_list("Truncated Purple", cm.Blues(np.linspace(0.8, 0.9, 10))),
               ],
        "lw": [2.],
        "discrimination": [[("Dct", "No Rnt", "No Qch", "Lbr")],
                           ],
        "refs": ["perfect_behaviour"]
    }
