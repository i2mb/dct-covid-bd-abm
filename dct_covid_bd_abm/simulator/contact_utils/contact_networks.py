import numpy as np
import pandas as pd
import igraph as ig

from dct_covid_bd_abm.simulator.contact_utils.contact_files import load_contact_history, default_aggregates
from dct_covid_bd_abm.simulator.contact_utils.network_metrics import NxMetric


def generate_contacts_per_time_stamps(individuals, num_samples):
    from itertools import combinations
    possible_contacts = np.array(list(combinations(individuals, 2)))
    contact_list = []

    for sample in range(num_samples):
        if len(possible_contacts) == 1:
            make_contact = np.random.random() < 0.5
            if make_contact:
                contact_list.append(possible_contacts)

        else:
            num_contacts = np.random.randint(0, len(possible_contacts))
            instant_contacts = np.random.choice(np.arange(len(possible_contacts)), num_contacts, replace=False)
            contact_list.append(possible_contacts[instant_contacts])

    contacts = pd.DataFrame(np.vstack(contact_list), columns=["id_1", "id_2"])
    contacts = contacts.groupby(["id_1", "id_2"]).size()
    contacts.name = "duration"
    contacts = contacts.reset_index()

    return contacts


def generate_contacts_per_time_stamps_with_mat(individuals, num_samples):
    individuals = np.array(individuals)
    contact_list = []

    for sample in range(num_samples):
        possible_contacts = np.random.random((len(individuals), len(individuals))) > 0.5
        # possible_contacts |= possible_contacts.T
        possible_contacts = np.triu(possible_contacts, 1)
        possible_contacts = np.where(possible_contacts)
        possible_contacts = np.vstack([individuals[possible_contacts[0]], individuals[possible_contacts[1]]]).T
        contact_list.append(possible_contacts)

    contacts = pd.DataFrame(np.vstack(contact_list), columns=["id_1", "id_2"])
    contacts = contacts.groupby(["id_1", "id_2"]).size()
    contacts.name = "duration"
    contacts = contacts.reset_index()

    return contacts


def generate_random_contacts(individuals, num_samples=1000):
    contacts = pd.DataFrame(np.random.choice(individuals, (num_samples, 2), replace=True), columns=["id_1", "id_2"])
    contacts = contacts.groupby(["id_1", "id_2"]).size()
    contacts.name = "duration"
    contacts = contacts.reset_index()
    different_id = contacts["id_1"] != contacts["id_2"]
    contacts = contacts[different_id]
    return contacts


def get_contacts_by_type(contact_run_file, contact_type=None):
    contacts_df = load_contact_history(contact_run_file)
    if contact_type is None:
        return contacts_df

    family_contacts = contacts_df[contacts_df.loc[:, "type"] == contact_type]
    return family_contacts


def get_contacts_from_individuals(contact_run_file, individuals):
    contacts_df = pd.read_feather(contact_run_file)
    selector = ((contacts_df.id_1.values.reshape(-1, 1) == individuals).any(axis=1) &
                (contacts_df.id_2.values.reshape(-1, 1) == individuals).any(axis=1))
    contacts = contacts_df[selector]
    return contacts


def get_chain_heads(graph: ig.Graph):
    chain_heads = []

    def __get_chains_directed_named():
        for v in graph.vs:
            if v.degree(mode="out") > 0 and v.degree(mode="in") == 0:
                chain_heads.append(int(v["name"]))

    def __get_chains_directed_indexed():
        for v in graph.vs:
            if v.degree(mode="out") > 0 and v.degree(mode="in") == 0:
                chain_heads.append(v.index)

    def __get_chains_indexed():
        for cc in graph.connected_components():
            if cc[0].degree() > 0:
                chain_heads.append(cc[0].index)

    def __get_chains_named():
        for cc in graph.connected_components():
            if cc[0].degree() > 0:
                chain_heads.append(int(cc[0]["name"]))

    if graph.is_directed:
        if "name" in graph.vs.attributes():
            __get_chains_directed_named()
        else:
            __get_chains_directed_indexed()
    else:
        if "name" in graph.vs.attributes():
            __get_chains_named()
        else:
            __get_chains_indexed()

    return chain_heads


def compute_vertex_metric_by_edge(vertex_metric, edge_vertex="source"):
    def wrapped_metric(graph, **kwargs):
        idx = graph.get_edge_dataframe()[edge_vertex].values
        return vertex_metric(graph, idx, **kwargs)

    return wrapped_metric


def get_edge_generation(graph, vertices=None, **kwargs):
    if vertices is None:
        vertices = slice(None)

    chain_heads = get_chain_heads(graph)

    generation = pd.DataFrame(graph.distances(chain_heads), index=chain_heads).T.min(axis=1)
    return generation.values.ravel()[vertices]


def compute_attributes(graph, component_sequence, attrs: [NxMetric]):
    for nx_l, nx_m, nx_kwargs in attrs:
        component_sequence[nx_l] = nx_m(graph, **nx_kwargs)


def compute_joint_attributes(infection_graph, contacts_graph, component_sequence, joint_attrs: [NxMetric]):
    for nx_l, nx_m, nx_kwargs in joint_attrs:
        component_sequence[nx_l] = nx_m(infection_graph, contacts_graph, **nx_kwargs)


def build_graphs(contacts, infection_map, states,):
    # Create graph
    contacts_graph = build_contact_graph(contacts)
    infection_graph = build_infection_graph(infection_map, status=states)
    return contacts_graph, infection_graph


def merge_nx_and_epi_graphs(contacts_graph, infection_graph):
    # Merge graphs and extract edge dataframe
    merge_graph_edges_properties(contacts_graph, infection_graph)
    merge_graph_vertex_properties(contacts_graph, infection_graph)
    return


def create_joint_graph_attributes(contacts_graph, infection_graph, joint_edge_attrs, joint_vertex_attrs):
    if joint_edge_attrs is None:
        joint_edge_attrs = []
    if joint_vertex_attrs is None:
        joint_vertex_attrs = []

    compute_joint_attributes(infection_graph, contacts_graph, infection_graph.es, joint_edge_attrs)
    compute_joint_attributes(infection_graph, contacts_graph, infection_graph.vs, joint_vertex_attrs)


def create_graph_attributes(contact_edge_attrs, contact_vertex_attrs, contacts_graph):
    if contact_edge_attrs is None:
        contact_edge_attrs = []

    if contact_vertex_attrs is None:
        contact_vertex_attrs = []

    compute_attributes(contacts_graph, contacts_graph.es,  contact_edge_attrs)
    compute_attributes(contacts_graph, contacts_graph.vs, contact_vertex_attrs)


def merge_graph_edges_properties(contacts_graph, infection_graph):
    for edge_attribute in contacts_graph.es.attributes():
        new_attribute = [contacts_graph.es[contacts_graph.get_eid(e.source, e.target)][edge_attribute]
                         for e in infection_graph.es]

        infection_graph.es[edge_attribute] = new_attribute

    return


def merge_graph_vertex_properties(contacts_graph, infection_graph):
    for vertex_attribute in contacts_graph.vs.attributes():
        new_attribute = [contacts_graph.vs[e.index][vertex_attribute]
                         for e in infection_graph.vs]

        infection_graph.vs[vertex_attribute] = new_attribute

    return


def build_infection_graph(infection_map, status):
    edge_attrs = {"infection time": [iv[0] for k, v in infection_map.items() for ik, iv in v.items()]}
    infection_graph = ig.Graph(n=1000, edges=[(k, ik) for k, v in infection_map.items() for ik, iv in v.items()],
                               vertex_attrs={"label": list(range(1000)),
                                             "recovered": (status["infected"].values < np.inf)},
                               edge_attrs=edge_attrs, directed=True)

    infection_graph.vs["infected"] = infection_graph.degree(mode="out")

    return infection_graph


def build_contact_graph(contacts):
    contact_grouper = contacts.groupby(["id_1", "id_2"])
    edge_attr = (contact_grouper.agg(default_aggregates(contacts, skip=["id_1", "id_2"]))
                                .to_dict("list"))
    full_contacts_graph = ig.Graph(contact_grouper.sum().index.to_list(),
                                   vertex_attrs={"name": list(range(1000))},
                                   edge_attrs=edge_attr)

    return full_contacts_graph


