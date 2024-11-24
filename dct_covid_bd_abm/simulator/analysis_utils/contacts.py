import os
from collections import Counter
from functools import partial
from glob import glob
from itertools import product
from pathlib import Path

import igraph as ig
from pyarrow import feather

import networkx as nx
import numpy as np
import pandas as pd

from dct_covid_bd_abm.simulator.analysis_utils.data_management import tokenize_experiment, expand_experiment_name
from dct_covid_bd_abm.simulator.analysis_utils.pathogen import load_infection_map, load_state_history, load_waves, \
    get_infection_incidence
from dct_covid_bd_abm.simulator.contact_utils.contact_networks import merge_nx_and_epi_graphs, get_chain_heads, build_graphs, \
    create_graph_attributes, create_joint_graph_attributes
from dct_covid_bd_abm.simulator.contact_utils.network_metrics import NxMetric
from dct_covid_bd_abm.simulator.contact_utils.contact_files import load_contact_history, default_aggregates
from dct_covid_bd_abm.simulator.utilities.file_utilities import load_metadata


def read_csv_file(file_name):
    df = pd.read_csv(file_name, header=None, names=["id_1", "id_2", "type", "start", "duration", "location"],
                     dtype={"type": "category", "location": "category"})

    # Remove leading spaces for categories
    df["type"].cat.categories = [s.strip() for s in df["type"].cat.categories]
    df["location"].cat.categories = [s.strip() for s in df["location"].cat.categories]

    return df


def read_feather_file(file_name):
    return feather.read_feather(file_name.replace(".csv", ".feather"))


def get_contact_types(data_frame):
    return data_frame["type"].cat.categories


def get_contact_locations(data_frame):
    return data_frame["location"].cat.categories


def select_interesting_contacts(data_frame, duration=3):
    interesting_contacts = data_frame["duration"] >= duration
    interesting_contacts = data_frame[interesting_contacts]
    return interesting_contacts


def generate_network(data_frame, contact_type=None):
    if contact_type is None:
        contacts_of_type = np.ones(len(data_frame), dtype=bool)
    else:
        contacts_of_type = data_frame["type"] == contact_type

    network = nx.Graph()
    network.add_edges_from(data_frame[contacts_of_type].loc[:, ["id_1", "id_2"]].to_numpy())

    return network


def generate_network_per_location(data_frame, location):
    contacts_at_location = data_frame["location"] == location
    network = nx.Graph()
    network.add_edges_from(data_frame[contacts_at_location].loc[:, ["id_1", "id_2"]].to_numpy())

    return network


def contact_histogram(network):
    histogram = Counter(dict(network.degree).values())
    return pd.DataFrame(histogram, index=[0]).sort_index(axis=1)


def assign_home_contacts(df):
    """Assign home locations to Home"""
    home_locations = [
        'Bathroom',
        'Corridor',
        'DiningRoom',
        'Kitchen',
        'LivingRoom']

    if "Home" not in df["location"].cat.categories:
        df["location"] = df["location"].cat.add_categories("Home")

    for location in home_locations:
        selector = df["location"] == location
        df["location"][selector] = "Home"

    df["location"] = df["location"].cat.remove_unused_categories()


def assign_other_contacts(df):
    """Assign home locations to Home"""
    other_locations = [
        'Bar',
        'Restaurant',
    ]

    if "Other" not in df["location"].cat.categories:
        df["location"] = df["location"].cat.add_categories("Other")

    for location in other_locations:
        selector = df["location"] == location
        df["location"][selector] = "Other"

    df["location"] = df["location"].cat.remove_unused_categories()


def generate_weighted_network_per_location(data_frame, location=None):
    if location is None:
        contacts_at_location = np.ones(len(data_frame), dtype=bool)
    else:
        contacts_at_location = data_frame["location"] == location

    network = nx.Graph()
    tuples = [(*s[0], s[1]) for s in data_frame.groupby(["id_1", "id_2"]).count().loc[:, "type"].iteritems()]
    network.add_weighted_edges_from(tuples)
    return network


def average_contact_per_person_unique(data_frame):
    network = generate_weighted_network_per_location(data_frame)
    distinct_contacts = np.array(network.degree)[:, 1]
    return pd.DataFrame(distinct_contacts, columns=["Contacts"])


def average_contact_per_person(data_frame):
    network = generate_weighted_network_per_location(data_frame)
    total_contacts = np.array(network.degree(weight="weight"))[:, 1]
    return pd.DataFrame(total_contacts, columns=["Contacts"])


def process_contacts(experiments, processor, pool, count_unique=True, complete=False, hospital_contacts=False):
    """
    Create a dataframe that merges all runs of an experiment vertically, and all experiments horizontally.  The
    parameter `experiments` contains a list of experiment folders. Each file is processed using the `processor`
    function. This function depends on a multiprocessing `pool` to process files in parallel. When setting the
    count_unique parameter only unique contacts are counted according to the grouping created by the processor function.
    When using `complete` the dataframe includes the agents without contacts.

    Parameters
    ----------
    experiments : List[str]
    processor : Callable
    pool : multiprocessing.Pool
    count_unique : bool
    complete : bool
    hospital_contacts : bool
    """
    results = []
    for experiment in experiments:
        file_list = glob(f"{experiment}/*_contact_history.csv")
        population_size, time_factor_ticks_day = read_metadata(file_list[0])
        e_name = get_experiment_name(experiment)

        if "Both" in e_name:
            e_name = "ICT MCT DCT"

        for file_name in file_list:
            results.append(pool.apply_async(process_contact_file, (file_name, count_unique, processor,
                                                                   time_factor_ticks_day, e_name)))

    sorter = {}
    for res in results:
        e_name, df = res.get()
        sorter.setdefault(e_name, []).append(df)
    print("sorted results")

    df_list = pool.map(partial(concat_experiment_data_frames, complete=complete,
                               population_size=population_size), sorter.items())

    print(f"Merged {len(df_list)} DataFrames")
    if hospital_contacts:
        return pd.concat(df_list, axis=1)
    else:
        merged_df = pd.concat(df_list, axis=1)
        return merged_df[merged_df["location"] != "Hospital"]


def get_experiment_name(experiment):
    c, e, s = tokenize_experiment(experiment)
    e_name = expand_experiment_name(e, ["validation_", "with_"])
    return e_name


def read_metadata(file_name):
    time_factor_ticks_day = load_metadata(file_name.replace(".csv", "npz"), "time_factor_ticks_day")
    population_size = load_metadata(file_name.replace(".csv", "npz"), "population_size")
    return population_size, time_factor_ticks_day


def concat_experiment_data_frames(sorter_item, complete, population_size):
    e_name, results = sorter_item
    series = pd.concat(results, axis=1, keys=range(len(results))).stack(0)
    if complete:
        new_index = pd.MultiIndex.from_product([series.index.levels[0], series.index.levels[1],
                                                range(population_size), series.index.levels[3]])
        print([len(l) for l in new_index.levels])
        series = series.reindex(new_index, fill_value=0)

    if type(results[0]) is pd.Series:
        series.name = e_name

    else:
        series.columns = pd.MultiIndex.from_product([[e_name], series.columns])
    return series


def process_contact_file(file_name, count_unique, processor, time_factor_ticks_day, e_name):
    df = read_feather_file(file_name)
    df["location"] = df["location"].astype("category")
    assign_home_contacts(df)
    assign_other_contacts(df)

    # Align to day
    df["start"] = df["start"] // time_factor_ticks_day
    df = select_interesting_contacts(df, 3)
    days = df["start"].max()
    df = processor(df, count_unique)  # type: pd.DataFrame

    return e_name, df


def get_averager_contact_per_person(count_unique):
    if count_unique:
        return average_contact_per_person_unique

    return average_contact_per_person


def overall_contacts_processor(df, count_unique):
    return df.groupby(["start"]).apply(get_averager_contact_per_person(count_unique)).reset_index(drop=True)


def location_contacts_processor(df, count_unique):
    partial = df.groupby(by=["start", "location"]).apply(get_averager_contact_per_person(count_unique))
    partial = partial.reset_index(level=1).reset_index(drop=True)
    return partial


def ts_location_contacts_processor(df, count_unique):
    partial = df.groupby(by=["start", "location"]).apply(get_averager_contact_per_person(count_unique))
    return partial


def ts_overall_contacts_processor(df, count_unique):
    return df.groupby(["start"]).apply(get_averager_contact_per_person(count_unique))


def melt_series(key, series, reindex=True, new_index=None, reset_idx_levels=None):
    if reindex:
        if new_index is None:
            days = max(series.index.get_level_values(0)) + 1
            new_index = pd.MultiIndex.from_product([range(days), range(1000), range(50)])
        else:
            new_index = pd.MultiIndex.from_product(new_index)

        series = series.reindex(new_index, fill_value=0)

    if reset_idx_levels is None:
        series = pd.DataFrame(series, columns=[key]).melt(var_name="Study", value_name="Contacts")
    else:
        series.name = "Contacts"
        series = series.reset_index(level=reset_idx_levels)
        series["Study"] = key

    series["Study"] = series["Study"].astype("category")
    return series


def load_raw_data(stages):
    from multiprocessing import Pool
    with Pool(10, maxtasksperchild=1) as pool:
        print("Processing data with unique contacts")
        contact_data_unique = process_contacts(stages, ts_location_contacts_processor, pool, count_unique=True)
        contact_data_unique = contact_data_unique.droplevel(1, axis=1)
        contact_data_unique.index.names = ["start", "location", "u_id", "run_id"]
        contact_data_unique.reset_index().to_feather("../../data/contact_validation/contact_data_unique.feather")

        print("Processing data with all contacts")
        contact_data_all = process_contacts(stages, ts_location_contacts_processor, pool, count_unique=False)
        contact_data_all = contact_data_all.droplevel(1, axis=1)
        contact_data_all.index.names = ["start", "location", "u_id", "run_id"]
        contact_data_all.reset_index().to_feather("../../data/contact_validation/contact_data_all.feather")


def load_raw_data_complete(base_dir):
    from multiprocessing import Pool
    with Pool(10, maxtasksperchild=1) as pool:
        print("Processing data with unique contacts, including no contacts")
        from dct_covid_bd_abm.simulator.contact_utils.tomori_model import studies
        from dct_covid_bd_abm.simulator.contact_utils.contact_files import stages
        tomori_stages = [stages[stage] for stage in studies]

        contact_data_unique_complete = process_contacts(tomori_stages, ts_location_contacts_processor, pool, count_unique=True,
                                                        complete=True)
        contact_data_unique_complete = contact_data_unique_complete.droplevel(1, axis=1)  # tpd.DatFrame
        contact_data_unique_complete.index.names = ["start", "location", "u_id", "run_id"]
        contact_data_unique_complete.reset_index().to_feather(
            f"{base_dir}/contact_data_unique_complete.feather")

        print("Processing data with all contacts, including no contacts")
        contact_data_all_complete = process_contacts(tomori_stages, ts_location_contacts_processor, pool, count_unique=False,
                                                     complete=True)
        contact_data_all_complete = contact_data_all_complete.droplevel(1, axis=1)
        contact_data_all_complete.index.names = ["start", "location", "u_id", "run_id"]
        contact_data_all_complete.reset_index().to_feather(
            f"{base_dir}/contact_data_all_complete.feather")


def fix_infection_map(infection_map, state_history, contact_history):
    def find_node_infectors(node):
        infection_time = state_history.loc[node, "infected"]
        indexer: pd.Series
        indexer = (((contact_history["id_1"] == node) | (contact_history["id_2"] == node)) &  # noqa
                   ((contact_history["start"] <= infection_time) & (contact_history["end"] >= infection_time)))  # noqa

        if indexer.any() == 0:
            # print(f"Found no possible infector contact for node `{node}`")
            raise RuntimeError(f"Found no possible infector contact for node `{node}`")

        candidates = contact_history[indexer]
        infector, codes = (candidates.loc[:, ["id_1", "id_2"]] != node).idxmax(axis=1).factorize()
        infectors = candidates.reindex(codes, axis=1).to_numpy()[range(len(candidates)), infector]

        # Select active contacts
        recovered_times = state_history.loc[infectors, ["immune", "deceased"]].min(axis=1)
        infected_times = state_history.loc[infectors, "infected"]
        active_contact = (infection_time > infected_times) & (infection_time < recovered_times)

        active_contact = active_contact[active_contact]
        return active_contact.index

    def find_orphan_infectees():
        def dict_walk(dict_):
            for key, items in dict_.items():
                if isinstance(items, dict):
                    yield key
                    yield from dict_walk(items)

                else:
                    yield key

        return set(state_history["infected"][state_history["infected"] < np.inf].index) - set(dict_walk(infection_map))

    def get_infection_chain_heads():
        g = ig.Graph.DictDict({str(k): {str(ki): {"duration": vi[0]}
                                        for ki, vi in v.items()}
                               for k, v in infection_map.items()},
                              directed=True)

        chain_heads = [(chain_head, state_history.loc[chain_head, "infected"]) for chain_head in get_chain_heads(g)]

        return chain_heads

    contact_history["end"] = contact_history["start"] + contact_history["duration"]

    # Remove infection interactions that are not in the contact history
    pairwise_infection_map = [(s, t) for s, v in infection_map.items() for t in v]
    for source, target in pairwise_infection_map:
        id_1 = min(source, target)
        id_2 = max(source, target)
        if len(contact_history[(contact_history["id_1"] == id_1) & (contact_history["id_2"] == id_2)]) == 0:
            del infection_map[source][target]
            if len(infection_map[source]) == 0:
                del infection_map[source]

    # Find orphans
    for orphan in find_orphan_infectees():
        infection_time = state_history.loc[orphan, "infected"]
        if int(infection_time) == 2:
            continue

        try:
            for infector in find_node_infectors(orphan):
                infection_map.setdefault(infector, {})[orphan] = [state_history.loc[orphan, "infected"] - 1]
        except RuntimeError as e:
            print(f"Failed processing orphans: {e}")

    # Trim infection heads
    for head, infection_time in get_infection_chain_heads():
        if int(infection_time) == 2:
            continue

        try:
            for infector in find_node_infectors(head):
                infection_map.setdefault(infector, {})[head] = [state_history.loc[head, "infected"] - 1]

        except RuntimeError as e:
            print(f"Failed processing chain heads: {e}")

    # Corroborate actual contacts


def compute_vertex_contact_metric_by_infection_edge(vertex_metric, edge_vertex="source"):
    def wrapped_metric(infection_graph, contact_graph, **kwargs):
        idx = infection_graph.get_edge_dataframe()[edge_vertex].values
        return vertex_metric(contact_graph, idx, **kwargs)

    return wrapped_metric


def apply_filters(contact_history, infection_map, states, time_factor_ticks_day, contact_history_filters):
    ch = contact_history
    for method, args, kwargs in contact_history_filters:
        kwargs.setdefault("time_factor_ticks_day", time_factor_ticks_day)
        kwargs.setdefault("infection_map", infection_map)
        kwargs.setdefault("states", states)

        ch = method(ch, *args, **kwargs)

    return ch


def contacts_until_infection(contact_history, *, states=None, **kwargs):
    def filter_function(df):
        id_1 = df["id_1"].iloc[0]
        id_2 = df["id_2"].iloc[0]
        inf_time_1 = states.loc[id_1, "infected"]
        inf_time_2 = states.loc[id_2, "infected"]

        pre_inf_1 = df["start"] < inf_time_1
        pre_inf_2 = df["start"] < inf_time_2

        # Although we do not know is a contact lead to an infection, we assume that the contact infected earlier
        # is the source of the infection. When combined with the infection graph, this fields will match with the
        # source and target.
        if inf_time_1 < inf_time_2:
            df["source duration pre infection"] = df.duration * pre_inf_1
            df["target duration pre infection"] = df.duration * pre_inf_2
            df["source contacts pre infection"] = pre_inf_1
            df["target contacts pre infection"] = pre_inf_2
        else:
            df["source duration pre infection"] = df.duration * pre_inf_2
            df["target duration pre infection"] = df.duration * pre_inf_1
            df["source contacts pre infection"] = pre_inf_2
            df["target contacts pre infection"] = pre_inf_1

        return df

    return contact_history.groupby(["id_1", "id_2"], as_index=False).apply(filter_function)


def contact_person_pre_infection(contact_history, *, time_factor_ticks_day=1, **kwargs):
    contact_history["day"] = contact_history["start"] // time_factor_ticks_day

    def _contact_person(df):
        columns = [f"{column} contact person pre infection" for column in ["source", "target"]]
        test_columns = ["source contacts pre infection", "target contacts pre infection"]

        df.loc[:, columns] = 0
        df.loc[:, columns].iloc[-1, :] = 1 * (df.loc[:, test_columns].max(axis=0) > 0).values

        return df

    ch = contact_history.groupby(["day", "id_1", "id_2"], as_index=False).apply(_contact_person)
    return ch


def average_daily_contacts(contact_history, *, time_factor_ticks_day=1, extra_fields=None, aggregates=None, **kwargs):
    if extra_fields is None:
        extra_fields = []

    if aggregates is None:
        aggregates = {}

    if "day" not in contact_history.colums:
        contact_history["day"] = contact_history["start"] // time_factor_ticks_day

    ch = contact_history.loc[:, ["id_1", "id_2", "day", "duration", "type"] + extra_fields]
    ch_grouper = ch.groupby(["id_1", "id_2"], as_index=False)

    def _weight_day(x):
        result_columns = ["duration"] + extra_fields
        results = np.zeros((len(x), len(result_columns)))
        for col_ix, column in enumerate(result_columns):
            results[:, col_ix] = x.loc[:, column] / (x["day"] * (x[column] > 0)).max()

        return results

    ch.loc[:, ["daily duration"] + extra_fields] = (ch_grouper.apply(_weight_day)).values
    ch = ch.drop("day", axis=1)

    return ch_grouper.agg(default_aggregates(ch.drop(["id_1", "id_2"], axis=1).update(aggregates)))


def convert_sim_time(contact_history, *, time_factor_ticks_day=1, **kwargs):
    contact_history.loc[:, ["start", "duration"]] = contact_history.loc[:,
                                                    ["start", "duration"]] / time_factor_ticks_day
    return contact_history


def degree_pre_infection(graph: ig.Graph, idx, weights=None):
    print(weights)
    new_edges = [edge for edge in graph.es if edge[weights] > 0]
    return ig.Graph(n=graph.vcount(), edges=new_edges).degree()


def load_contact_enriched_epidemiological_data(stage,
                                               contact_history_filters=None,
                                               contact_edge_attrs=None, contact_vertex_attrs=None,
                                               epi_edge_attrs=None, epi_vertex_attrs=None,
                                               joint_edge_attrs=None, joint_vertex_attrs=None, norm=None):
    """Examples of epi_metrix

    {"generation interval": get_generation_interval_map,
    "serial interval": get_serial_interval_map
    }

    """
    if joint_edge_attrs is None:
        joint_edge_attrs = [
            NxMetric("degree", compute_vertex_contact_metric_by_infection_edge(ig.Graph.degree)),
            NxMetric("strength", compute_vertex_contact_metric_by_infection_edge(ig.Graph.strength),
                     {"weights": "duration"}),
            NxMetric("degree_target", compute_vertex_contact_metric_by_infection_edge(ig.Graph.degree,
                                                                                      edge_vertex="target")),
            NxMetric("strength_target", compute_vertex_contact_metric_by_infection_edge(ig.Graph.strength,
                                                                                        edge_vertex="target"),
                     {"weights": "duration"}),
            NxMetric("source degree pre infection",
                     compute_vertex_contact_metric_by_infection_edge(degree_pre_infection),
                     {"weights": "source contacts pre infection"}),
            NxMetric("target degree pre infection",
                     compute_vertex_contact_metric_by_infection_edge(degree_pre_infection,
                                                                     edge_vertex="target"),
                     {"weights": "target contacts pre infection"})
        ]

    if epi_vertex_attrs is None:
        epi_vertex_attrs = [NxMetric("people infected", ig.Graph.degree)]

    if contact_vertex_attrs is None:
        contact_vertex_attrs = [NxMetric("degree", ig.Graph.degree),
                                NxMetric("strength", ig.Graph.strength, {"weights": "duration"})]

    run_ = 0
    stage_dir = os.path.dirname(stage)
    file_name_pattern = "{}_{}_i2bm_sim_data".format(*stage_dir.split("/")[-2:])
    pathogen_run_file_ = f"{os.path.join(stage_dir, file_name_pattern)}_{run_:04d}.npz"
    time_factor_ticks_day = load_metadata(pathogen_run_file_, "time_factor_ticks_day")

    dataframes = {"edges": [], "vertices": []}
    for run_ in range(1):
        print(run_, end=", ")
        contact_run_file_ = f"{os.path.join(stage_dir, file_name_pattern)}_{run_:04d}_contact_history.feather"
        pathogen_run_file_ = f"{os.path.join(stage_dir, file_name_pattern)}_{run_:04d}.npz"

        infection_map = load_infection_map(pathogen_run_file_)
        states = load_state_history(pathogen_run_file_)
        contact_history = load_contact_history(contact_run_file_)

        fix_infection_map(infection_map, states, contact_history)

        if contact_history_filters is not None:
            contact_history = apply_filters(contact_history,
                                            infection_map,
                                            states,
                                            time_factor_ticks_day, contact_history_filters)

        try:
            contacts_graph, infection_graph = build_graphs(contact_history, infection_map, states)

            # Compute individual attributes
            create_graph_attributes(contact_edge_attrs, contact_vertex_attrs, contacts_graph)
            create_graph_attributes(epi_edge_attrs, epi_vertex_attrs, infection_graph)

            # Merge contacts_graph into  infection_graph
            merge_nx_and_epi_graphs(contacts_graph, infection_graph)

            # Create joint attributes
            create_joint_graph_attributes(contacts_graph, infection_graph, joint_edge_attrs, joint_vertex_attrs)
            edge_frame = infection_graph.get_edge_dataframe()
            vertex_frame = infection_graph.get_vertex_dataframe()

            if len(edge_frame) > 0:
                dataframes.setdefault("edges", []).append(edge_frame)

            if len(vertex_frame) > 0:
                dataframes.setdefault("vertices", []).append(vertex_frame)

        except ig.InternalError as e:
            continue

    for t in ["edges", "vertices"]:
        dfs = dataframes[t]
        if not dfs:
            continue

        dataframes[t] = pd.concat(dfs, keys=range(len(dfs)))

        # make type categorical
        if "type" in dataframes[t].columns:
            cat_type = dataframes[t].type.astype("category")
            dataframes[t].type = cat_type

        if t == "edges" and type(norm) is str and norm.lower() == "average":
            for nx_l, nm_v in product(joint_edge_attrs, ["", "_target", "_by_type", "_target_by_type"]):
                try:
                    column = dataframes[t].loc[:, f"{nx_l}{nm_v}"]

                except KeyError as e:
                    print(dataframes[t].columns)
                    raise e

                dataframes[t].loc[:, f"{nx_l}{nm_v}"] = column / column.mean()

    return dataframes


def load_infection_map_enriched_with_contact_info(base_path, file_selector=None, refresh=False):
    if file_selector is None:
        file_selector = slice(None)

    if not isinstance(file_selector, slice):
        file_selector = slice(file_selector, file_selector + 1)

    # Load the saved file if it exists
    edges_path = os.path.join(base_path, "infection_map_with_contact_metrics_edges.feather")
    vertices_path = os.path.join(base_path, "infection_map_with_contact_metrics_vertices.feather")
    if os.path.exists(edges_path) and os.path.exists(vertices_path) and not refresh:
        edges = pd.read_feather(edges_path).set_index(['level_0', 'level_1'])
        vertices = pd.read_feather(vertices_path).set_index(['level_0'])
        return dict(edges=edges,
                    vertices=vertices)

    edges = []
    vertices = []
    print(Path(base_path).absolute(), sorted(Path(base_path).glob("*.npz")))
    file_names = sorted(Path(base_path).glob("*.npz"))
    for file_name in file_names[file_selector]:
        contact_history = load_contact_history(file_name)
        infection_map = load_infection_map(file_name)
        # infection_map = {}
        state_history = load_state_history(file_name)
        wave_duration = load_waves(file_name)[0][-1]
        time_factor_ticks_day = load_metadata(file_name, "time_factor_ticks_day")

        contact_history["day"] = contact_history["start"] // time_factor_ticks_day
        contact_history["end"] = contact_history["start"] + contact_history["duration"]

        # Complete infection tree
        fix_infection_map(infection_map, state_history, contact_history)

        # Compute infection time
        contact_history.loc[:, "id_1 infected"] = state_history.loc[contact_history.id_1, "infected"].values
        contact_history.loc[contact_history["id_1 infected"] == np.inf, "id_1 infected"] = wave_duration

        contact_history.loc[:, "id_2 infected"] = state_history.loc[contact_history.id_2, "infected"].values
        contact_history.loc[contact_history["id_2 infected"] == np.inf, "id_2 infected"] = wave_duration

        # Compute source and target
        source_id_in_1 = contact_history["id_1 infected"] < contact_history["id_2 infected"]
        source_id_in_2 = ~source_id_in_1
        contact_history["source"] = contact_history['id_1'] * source_id_in_1 + contact_history['id_2'] * source_id_in_2
        contact_history["target"] = contact_history['id_1'] * source_id_in_2 + contact_history['id_2'] * source_id_in_1

        # Organize infection time to match source and target
        contact_history["source infected"] = contact_history['id_1 infected'] * source_id_in_1 + contact_history[
            'id_2 infected'] * source_id_in_2
        contact_history["target infected"] = contact_history['id_1 infected'] * source_id_in_2 + contact_history[
            'id_2 infected'] * source_id_in_1

        # Compute relative date
        contact_history["source relative day"] = (
                (contact_history["start"] - contact_history["source infected"]) // time_factor_ticks_day).astype(int)
        contact_history["target relative day"] = (
                (contact_history["start"] - contact_history["target infected"]) // time_factor_ticks_day).astype(int)

        # Compute duration pre infection
        contact_history["source duration pre infection"] = contact_history["duration"] * (
                contact_history["source relative day"] <= 0)
        contact_history["target duration pre infection"] = contact_history["duration"] * (
                contact_history["target relative day"] <= 0)

        # Compute 7-day pre infection.
        contact_history["source 7-day duration pre infection"] = contact_history["duration"] * (
                (contact_history["source relative day"] <= 0) & (contact_history["source relative day"] >= -7))
        contact_history["target 7-day duration pre infection"] = contact_history["duration"] * (
                (contact_history["target relative day"] <= 0) & (contact_history["target relative day"] >= -7))

        # Compute Degree and Strength. We can sum them up, because there is no intersection between the sets where an
        # agent j is the source and an agent j is the target
        vertex_metrics = (contact_history.groupby(["source"])
                          .agg({"duration": [("Strength", "sum")], "target": [("Degree", "nunique")]})
                          .droplevel(0, axis=1)
                          .reindex(range(1000))
                          .fillna(0)) + (
                             contact_history.groupby(["target"])
                             .agg({"duration": [("Strength", "sum")], "source": [("Degree", "nunique")]})
                             .droplevel(0, axis=1)
                             .reindex(range(1000))
                             .fillna(0))

        vertex_metrics.loc[:, ["Strength pre infection", "Degree pre infection"]] = (
                contact_history[contact_history["source relative day"] <= 0].groupby(["source"])
                .agg({"source duration pre infection": [("Strength pre infection", "sum")],
                      "target": [("Degree pre infection", "nunique")]})
                .droplevel(0, axis=1)
                .reindex(range(1000))
                .fillna(0) +
                contact_history[contact_history["target relative day"] <= 0].groupby(["target"])
                .agg({"target duration pre infection": [("Strength pre infection", "sum")],
                      "source": [("Degree pre infection", "nunique")]})
                .droplevel(0, axis=1)
                .reindex(range(1000))
                .fillna(0))

        vertex_metrics.loc[:, ["7-day Strength pre infection", "7-day Degree pre infection"]] = (
                contact_history[contact_history["source 7-day duration pre infection"] > 0].groupby(["source"])
                .agg({"source 7-day duration pre infection": [("7-day Strength pre infection", "sum")],
                      "target": [("7-day Degree pre infection", "nunique")]})
                .droplevel(0, axis=1)
                .reindex(range(1000))
                .fillna(0) +
                contact_history[contact_history["target 7-day duration pre infection"] > 0].groupby(["target"])
                .agg({"target 7-day duration pre infection": [("7-day Strength pre infection", "sum")],
                      "source": [("7-day Degree pre infection", "nunique")]})
                .droplevel(0, axis=1)
                .reindex(range(1000))
                .fillna(0))

        infection_contacts = pd.DataFrame([[source, target, inf_time[0]]
                                           for source, targets in infection_map.items()
                                           for target, inf_time in targets.items()],
                                          columns=["source", "target", "infection time"])

        # Compute 7-day incidence
        infection_contacts["infection day"] = infection_contacts["infection time"] // time_factor_ticks_day
        infection_contacts["7-day incidence"] = get_infection_incidence(file_name).fillna(0).loc[
            infection_contacts["infection day"]].values

        # Compute number of infected individuals per source
        vertex_metrics["infected"] = (infection_contacts.groupby("source")
                                      .size().reindex(range(1000))
                                      .fillna(0))

        # Associate contact metrics to infection interactions
        for person in ["source", "target"]:
            for metric in ["Strength", "Strength pre infection", "Degree", "Degree pre infection",
                           "7-day Strength pre infection", "7-day Degree pre infection"]:
                infection_contacts[f"{person} {metric.lower()}"] = vertex_metrics.loc[
                    infection_contacts[person].values, metric].values

        # Add contact "type" information of interaction
        inf_map_indexer = list(infection_contacts.loc[:, ["source", "target"]].itertuples(index=False, name=None))
        infection_contacts["type"] = contact_history.groupby(["source", "target"])["type"].last().loc[
            inf_map_indexer].values

        edges.append(infection_contacts)
        vertices.append(vertex_metrics)

    edges = pd.concat(edges, keys=range(len(file_names)))
    vertices = pd.concat(vertices, keys=range(len(file_names)))

    # Save files
    edges.reset_index().to_feather(edges_path)
    vertices.reset_index().to_feather(vertices_path)

    print(edges.reset_index().columns)
    print(vertices.reset_index().columns)

    return dict(edges=edges, vertices=vertices)


def load_daily_contact_metrics_in_time_relative_to_infection(remove_hospital_contacts=True, base_path=None,
                                                             file_selector=None, refresh=False):

    if file_selector is None:
        file_selector = slice(None)

    if not isinstance(file_selector, slice):
        file_selector = slice(file_selector, file_selector + 1)

    # Load the saved file if it exists
    contacts_path = Path(base_path) / f"contacts_daily_metrics_rti{'_w_hc' if not remove_hospital_contacts else ''}.feather"
    if contacts_path.exists() and not refresh:
        print(f"Loading data from {contacts_path}")
        contacts = (pd.read_feather(contacts_path)
                    .set_index(["run", "id", "relative day", "Type"])
                    .unstack("Type")
                    .swaplevel(0, 1, axis=1))

        return contacts

    combined_results = []
    base_path = Path(base_path)
    if len(list(base_path.glob("*.npz"))) == 0:
        raise RuntimeError(f"{base_path} is empty.")


    for file_name in base_path.glob("*.npz"):
        contact_history = load_contact_history(file_name)
        contact_history["end"] = contact_history["start"] + contact_history["duration"]
        state_history = load_state_history(file_name)
        time_factor_ticks_day = load_metadata(file_name, "time_factor_ticks_day")
        wave_duration = load_waves(file_name)[0][-1]

        # Remove Hospital contacts
        if remove_hospital_contacts:
            contact_history = contact_history[contact_history["location"] != "Hospital"]

        # Add infection time
        contact_history.loc[:, "id_1 infected"] = state_history.loc[contact_history.id_1, "infected"].values
        contact_history.loc[contact_history["id_1 infected"] == np.inf, "id_1 infected"] = wave_duration

        contact_history.loc[:, "id_2 infected"] = state_history.loc[contact_history.id_2, "infected"].values
        contact_history.loc[contact_history["id_2 infected"] == np.inf, "id_2 infected"] = wave_duration

        # Compute relative date
        contact_history["id_1 relative day"] = (
                (contact_history["start"] - contact_history["id_1 infected"]) // time_factor_ticks_day).astype(int)
        contact_history["id_2 relative day"] = (
                (contact_history["start"] - contact_history["id_2 infected"]) // time_factor_ticks_day).astype(int)

        # Melt to focus on each individual
        ch_melted = contact_history.melt(id_vars=["duration", "id_1 relative day", "id_2 relative day", "type"],
                                         value_vars=["id_1", "id_2"], ignore_index=False)

        # Compute contact
        ch_melted["contact"] = 0
        ch_melted.loc[ch_melted["variable"] == "id_1", "contact"] = ch_melted.loc[
            ch_melted["variable"] == "id_2", "value"]
        ch_melted.loc[ch_melted["variable"] == "id_2", "contact"] = ch_melted.loc[
            ch_melted["variable"] == "id_1", "value"]

        # Remove unaffected individuals
        susceptible = (state_history.loc[ch_melted["value"], "infected"] != np.inf).values
        ch_melted = ch_melted.loc[susceptible, :]

        # Compute degree, contacts, strength, and type composition
        ch_melted["relative day"] = np.where(ch_melted["variable"] == "id_1", ch_melted["id_1 relative day"],
                                             ch_melted["id_2 relative day"])
        ch_melted = ch_melted.drop(["variable", "id_1 relative day", "id_2 relative day"], axis=1)
        grouper = ch_melted.groupby(["value", "relative day"])
        daily_results = (grouper.agg({"duration": [("Strength", "sum")],
                                      "contact": [("Degree", "nunique")]})
                         .rename({"duration": "complete", "contact": "complete"}, axis=1))

        daily_results_by_type = (ch_melted.groupby(["value", "relative day", "type"], observed=True)
                                 .agg({"duration": [("Strength", "sum")],
                                       "contact": [("Degree", "nunique")]})
                                 .droplevel(0, axis=1)
                                 .unstack("type", 0)
                                 .swaplevel(0, 1, axis=1)
                                 .sort_index(axis=1)
                                 )

        daily_results = pd.concat([daily_results, daily_results_by_type], axis=1)
        combined_results.append(daily_results)

    contacts = pd.concat(combined_results, keys=range(len(combined_results)), names=["run", "id", "relative day"])
    contacts.columns.names = ["Type", "Metric"]
    contacts.stack("Type").reset_index().to_feather(contacts_path)

    return contacts


def load_daily_contact_metrics(remove_hospital_contacts=True, base_path=None, refresh=False, file_selector=None):
    if file_selector is None:
        file_selector = slice(None)

    if not isinstance(file_selector, slice):
        file_selector = slice(file_selector, file_selector + 1)

    # Load the saved file if it exists
    contacts_path = Path(base_path) / f"contacts_daily_metrics{'_w_hc' if not remove_hospital_contacts else ''}.feather"
    if contacts_path.exists() and not refresh:
        print(f"Loading data from {contacts_path}")
        contacts = (pd.read_feather(contacts_path)
                    .set_index(["run", "id", "day", "Type"])
                    .unstack("Type")
                    .swaplevel(0, 1, axis=1))

        return contacts

    combined_results = []
    base_path = Path(base_path)
    for file_name in base_path.glob("*.npz"):
        contact_history = load_contact_history(file_name)
        contact_history["end"] = contact_history["start"] + contact_history["duration"]
        time_factor_ticks_day = load_metadata(file_name, "time_factor_ticks_day")
        contact_history["day"] = contact_history["start"] // time_factor_ticks_day

        # Remove hospital contacts
        if remove_hospital_contacts:
            contact_history = contact_history[contact_history["location"] != "Hospital"]

        # Melt to focus on each individual
        ch_melted = contact_history.melt(id_vars=["duration", "day", "type"],
                                         value_vars=["id_1", "id_2"], ignore_index=False)

        # Compute contact
        ch_melted["contact"] = 0
        ch_melted.loc[ch_melted["variable"] == "id_1", "contact"] = ch_melted.loc[
            ch_melted["variable"] == "id_2", "value"]
        ch_melted.loc[ch_melted["variable"] == "id_2", "contact"] = ch_melted.loc[
            ch_melted["variable"] == "id_1", "value"]

        # Compute degree, contacts, strength, and type composition
        ch_melted = ch_melted.drop(["variable"], axis=1)
        grouper = ch_melted.groupby(["value", "day"])
        daily_results = (grouper.agg({"duration": [("Strength", "sum")],
                                      "contact": [("Degree", "nunique")]})
                         .rename({"duration": "complete", "contact": "complete"}, axis=1))

        daily_results_by_type = (ch_melted.groupby(["value", "day", "type"], observed=True)
                                 .agg({"duration": [("Strength", "sum")],
                                       "contact": [("Degree", "nunique")]})
                                 .droplevel(0, axis=1)
                                 .unstack("type", 0)
                                 .swaplevel(0, 1, axis=1)
                                 .sort_index(axis=1)
                                 )

        daily_results = pd.concat([daily_results, daily_results_by_type], axis=1)
        combined_results.append(daily_results)

    contacts = pd.concat(combined_results, keys=range(len(combined_results)), names=["run", "id", "day"])
    contacts.columns.names = ["Type", "Metric"]
    contacts.stack("Type").reset_index().to_feather(contacts_path)

    return contacts


def load_contact_data(base_path, file_selector=None, refresh=False):
    if file_selector is None:
        file_selector = slice(None)

    if not isinstance(file_selector, slice):
        file_selector = slice(file_selector, file_selector + 1)

    # Load the saved file if it exists
    contacts_path = Path(base_path) / "contacts_network_metrics.feather"
    if contacts_path.exists() and not refresh:
        contacts = pd.read_feather(contacts_path).set_index(["run", "id", "metric"]).unstack("metric")

        return contacts

    combined_results = []
    for file_name in Path(base_path).glob("*.npz"):
        contact_history = load_contact_history(file_name)
        contact_history["end"] = contact_history["start"] + contact_history["duration"]

        # Melt to focus on each individual
        ch_melted = contact_history.melt(id_vars=["duration", "type"],
                                         value_vars=["id_1", "id_2"], ignore_index=False)

        # Compute contact
        ch_melted["contact"] = 0
        ch_melted.loc[ch_melted["variable"] == "id_1", "contact"] = ch_melted.loc[
            ch_melted["variable"] == "id_2", "value"]
        ch_melted.loc[ch_melted["variable"] == "id_2", "contact"] = ch_melted.loc[
            ch_melted["variable"] == "id_1", "value"]

        # Compute degree, contacts, strength, and type composition
        ch_melted = ch_melted.drop(["variable"], axis=1)
        grouper = ch_melted.groupby(["value"])
        daily_results = (grouper.agg({"duration": [("Strength", "sum")],
                                      "contact": [("Degree", "nunique")]})
                         .rename({"duration": "complete", "contact": "complete"}, axis=1))

        daily_results_by_type = (ch_melted.groupby(["value", "type"], observed=True)
                                 .agg({"duration": [("Strength", "sum")],
                                       "contact": [("Degree", "nunique")]})
                                 .droplevel(0, axis=1)
                                 .unstack("type", 0)
                                 .swaplevel(0, 1, axis=1)
                                 .sort_index(axis=1)
                                 )

        daily_results = pd.concat([daily_results, daily_results_by_type], axis=1)
        combined_results.append(daily_results)

    contacts = pd.concat(combined_results, keys=range(len(combined_results)), names=["run", "id"])
    contacts.columns.names = ["type", "metric"]

    contacts.stack("metric").reset_index().to_feather(contacts_path)

    return contacts

