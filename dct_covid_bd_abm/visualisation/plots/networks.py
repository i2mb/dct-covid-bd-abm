import matplotlib as mpl
import networkx as nx
import igraph as ig
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dashboard import mc_rcParams
from dct_covid_bd_abm.simulator.contact_utils.contact_networks import get_contacts_by_type, generate_contacts_per_time_stamps_with_mat
from dct_covid_bd_abm.simulator.world_utils import load_homes, load_offices, assign_building_ids


def create_figure_with_color_bar(step_size=5):
    """step_size is in minutes."""
    fig, axd = plt.subplot_mosaic([[0, 1, 2, "cb"],
                                   [3, 4, 5, "cb"]],
                                  # figsize=(figure_width, figure_height*2),
                                  width_ratios=[1, 1, 1, 0.1],
                                  gridspec_kw={"left": 0.01, "right": .88, "top": 0.97, "bottom": 0.03})

    for ax in axd.values():
        ax.axis("off")

    axd["cb"].axis("on")

    cmap = mc_rcParams.get("network.edge.cmap", plt.cm.jet)
    cbar = plt.colorbar(plt.cm.ScalarMappable(
        norm=mpl.colors.Normalize(0, 500),
        cmap=cmap), cax=axd["cb"], label="HH:MM")

    cbar.set_ticks(cbar.get_ticks(),
                   labels=[f"{td.days * 24 + td.components.hours:02}:{td.components.minutes:02}" for td in
                           pd.to_timedelta(cbar.get_ticks() * step_size, unit="m")])

    return fig, axd


def draw_network(g, ax: plt.Axes, labels):
    cmap = mc_rcParams.get("network.edge.cmap", plt.cm.jet)
    edge_color = mc_rcParams.get("network.edge.color", None)
    edge_vmax = mc_rcParams.get("network.edge.vmax", 500)
    edge_vmin = mc_rcParams.get("network.edge.vmin", 0)
    pos = nx.nx_agraph.graphviz_layout(g, prog="circo")
    if len(g.nodes) > 1:
        edge_labels = nx.get_edge_attributes(g, "duration")
        if edge_color is None:
            edge_color = [edge_labels[e] for e in g.edges]
        else:
            cmap = None
        nx.draw_networkx_edges(g, pos, ax=ax,
                               edge_color=edge_color,
                               edge_cmap=cmap,
                               edge_vmax=edge_vmax, edge_vmin=edge_vmin,
                               width=2.)
        # nx.draw_networkx_edge_labels(g, pos, edge_labels, ax=ax)

    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=mc_rcParams.get("network.node.color", None))
    nx.draw_networkx_labels(g, pos, ax=ax, labels=labels)
    ax.axis("off")


def draw_sample_contact_networks(home_size, contacts, building_sizes, buildings, id_field="home_id"):
    fig, axd = create_figure_with_color_bar()
    buildings_per_size = building_sizes.groupby("size")[id_field].unique()[home_size]
    houses = np.random.choice(buildings_per_size, min(len(buildings_per_size), 6), replace=False)
    for hid, house in enumerate(houses):
        individuals = buildings[buildings[id_field] == house].index
        house_contacts = (
            (contacts.loc[:, ("id_1", "id_2")].values.reshape(-1, 2, 1) == individuals.values.reshape(-1, 1).T)
            .any(axis=1)
            .any(axis=1))
        house_contacts = contacts[house_contacts]

        g = nx.from_pandas_edgelist(house_contacts.groupby(["id_1", "id_2"]).sum().reset_index(), source="id_1",
                                    target="id_2", edge_attr="duration")

        ax = axd[hid]
        draw_network(g, ax, building_sizes.loc[list(g.nodes), f"u_{id_field}"].to_dict())


def draw_complete_contact_networks(contacts, home_sizes):
    fig, axd = create_figure_with_color_bar()
    house_groups = home_sizes.reset_index().groupby("size")["u_id"]
    for house_size, individuals in house_groups:
        if house_size == 1:
            continue

        house_contacts = (
            (contacts.loc[:, ("id_1", "id_2")].values.reshape(-1, 2, 1) == individuals.values.reshape(-1, 1).T)
            .any(axis=1)
            .any(axis=1))
        house_contacts = contacts[house_contacts].replace(home_sizes["u_home_id"])
        g = nx.from_pandas_edgelist(house_contacts.groupby(["id_1", "id_2"]).sum().reset_index(), source="id_1",
                                    target="id_2", edge_attr="duration")

        ax = axd[house_size - 1]
        draw_network(g, ax, None)

    return fig, axd


def draw_random_contact_networks(home_size):
    fig, axd = create_figure_with_color_bar()
    for hid in range(6):
        individuals = np.arange(1, home_size+1)
        house_contacts = generate_contacts_per_time_stamps_with_mat(individuals, 500)
        g = nx.from_pandas_edgelist(house_contacts.reset_index(), source="id_1",
                                    target="id_2", edge_attr="duration")
        ax = axd[hid]
        draw_network(g, ax, None)


def draw_random_contact_networks_multi_size():
    fig, axd = create_figure_with_color_bar()
    for home_size in range(2, 6):
        individuals = np.arange(1, home_size+1)
        house_contacts = generate_contacts_per_time_stamps_with_mat(individuals, 500)
        g = nx.from_pandas_edgelist(house_contacts.reset_index(), source="id_1",
                                    target="id_2", edge_attr="duration")
        ax = axd[home_size]
        draw_network(g, ax, None)

    return fig, axd


def draw_networks_by_types(contact_run_file_):
    for contact_type in ['acquaintance', 'family', 'friend', 'random']:
    # for contact_type in ['random']:
        contacts = get_contacts_by_type(contact_run_file_, contact_type)
        if contact_type == "random":
            contacts = contacts.sample(1000).groupby(["id_1", "id_2"]).sum().reset_index()

        else:
            contacts.groupby(["id_1", "id_2"]).sum().reset_index()

        g = nx.from_pandas_edgelist(contacts, source="id_1", target="id_2", edge_attr="duration")

        plt.figure()
        draw_network(g, plt.gca(), None)
        plt.gcf().suptitle(contact_type.title())


def network_differences(contact_run_file, run_file):
    family_contacts = get_contacts_by_type(contact_run_file, "family")
    homes = load_homes(run_file)
    home_sizes = assign_building_ids(homes)

    for home_size in home_sizes["size"].unique():
        if home_size < 2:
            continue

        cmap = mc_rcParams.get("network.edge.cmap", plt.cm.jet)
        draw_sample_contact_networks(home_size, family_contacts, home_sizes, homes)
        draw_random_contact_networks(home_size)


def network_averages(contact_run_file, run_file):
    family_contacts = get_contacts_by_type(contact_run_file, "family")
    homes = load_homes(run_file)
    home_sizes = assign_building_ids(homes)
    draw_complete_contact_networks(family_contacts, home_sizes)
    draw_random_contact_networks_multi_size()


def network_other_types(contact_type, contact_run_file, run_file):
    contacts = get_contacts_by_type(contact_run_file, contact_type)
    office = load_offices(run_file)
    office_sizes = assign_building_ids(office, "office_id")

    for home_size in office_sizes["size"].unique():
        if home_size <= 2:
            continue

        draw_sample_contact_networks(home_size, contacts, office_sizes, office, "office_id")
        # draw_random_contact_networks(home_size)


# Define styles this probably could go elsewhere
mc_rcParams.update({
    "network.edge.cmap": plt.cm.inferno_r,
    "network.edge.v_min": 10,
    "network.edge.v_max": 500,
    "network.node.color": "#a3f799"
})


def component_consumer(component_per_size, g, max_items=None):
    if max_items is None:
        max_items = slice(None)

    elif isinstance(max_items, int):
        max_items = slice(max_items)

    for component_size in sorted(component_per_size, key=lambda x: len(component_per_size[x])):
        components = component_per_size[component_size]
        for c_idx, sg in sorted(components, key=lambda x: g.subgraph(x[0]).ecount(), reverse=True)[max_items]:
            yield c_idx, g.subgraph(sg)


def layout_wrapped(g, max_columns=None, max_items=None, pad=None):
    if pad is None:
        pad = 0.3

    component_per_size = {}
    components = g.connected_components(mode='weak')
    for c_idx, sg in enumerate(components):
        if len(sg) == 1:
            continue

        component_per_size.setdefault(len(sg), []).append((c_idx, sg))

    # we need to automate this but later
    memberships = np.array(components.membership)
    global_coordinates = np.full((g.vcount(), 2), np.nan)

    # Top aligned
    total_consumed_height = 0
    total_consumed_width = 0

    x_ref, y_ref = 0, 0

    component_iter = component_consumer(component_per_size, g, max_items)

    shell = 1
    break_ = False
    while not break_:
        row_queue = []
        col_queue = []

        row_size = shell
        col_size = shell - 1

        try:
            if shell > 1:
                cumulative_height = 0
                for i, ci in enumerate(component_iter, start=1):
                    height = ci[1].layout_circle().bounding_box(pad).height
                    if i >= shell and cumulative_height + height >= total_consumed_height:
                        row_queue.append(ci)
                        row_size -= 1
                        break

                    col_queue.append(ci)
                    cumulative_height += height

            row_queue.extend([next(component_iter) for _ in range(row_size)])

        except StopIteration:
            break_ = True

        max_row_height = 0
        row_dims = []
        for c_ix, sg_r in row_queue:
            l = sg_r.layout_circle()
            bbox_r = l.bounding_box(pad)
            row_dims.append(bbox_r.shape)
            max_row_height = max(max_row_height, bbox_r.height)

        max_col_width = 0
        col_dims = []
        for c_ix, sg in col_queue:
            l = sg.layout_circle()
            bbox = l.bounding_box(pad)
            col_dims.append([bbox.width, bbox.height])
            max_col_width = max(max_col_width, bbox.width)

        # Include the width of the last row element
        if row_queue:
            max_col_width = max(max_col_width, bbox_r.width)

        row_dims = np.array(row_dims)
        col_dims = np.array(col_dims)

        row_length = total_consumed_width + max_col_width
        col_length = total_consumed_height

        if row_queue:
            row_centers = np.linspace(row_dims[0, 0] / 2, row_length - row_dims[-1, 0] / 2, len(row_queue))
            row_centers = np.vstack(
                [row_centers, np.full_like(row_centers, y_ref - (total_consumed_height + max_row_height / 2)), ]).T

            for (c_ix, sg), center in zip(row_queue, row_centers):
                l = sg.layout_circle()
                l.center(center)
                global_coordinates[memberships == c_ix, :] = l

        if col_queue:
            f_h = col_dims[:, 1] / 2
            f_h[1:] += f_h[:-1]
            if len(col_queue) == shell - 1:
                col_heights = y_ref - np.linspace(col_dims[0, 1] / 2, col_length - col_dims[-1, 1] / 2, len(col_queue))
            else:
                col_heights = y_ref - np.cumsum(f_h)

            col_centers = np.vstack(
                [np.full_like(col_heights, total_consumed_width + max_col_width / 2), col_heights]).T

            for (c_ix, sg), center in zip(col_queue, col_centers):
                l = sg.layout_circle()
                l.center(center)
                global_coordinates[memberships == c_ix, :] = l

        total_consumed_width += max_col_width
        total_consumed_height += max_row_height

        shell += 1

    return ig.Layout(global_coordinates)
