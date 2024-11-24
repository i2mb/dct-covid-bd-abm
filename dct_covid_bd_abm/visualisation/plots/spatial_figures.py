import matplotlib
import matplotlib.pyplot as plt

from i2mb.engine.agents import AgentList
from i2mb.engine.configuration import Configuration
from i2mb.utils.visualization.scale_bar import scale_bar
from i2mb.worlds import BusMBCitaroK

from dct_covid_bd_abm.configs.experiment_setup import available_scenarios
from dct_covid_bd_abm.simulator.utilities.config_utilitites import apply_alternate_config
from dct_covid_bd_abm.simulator.scenarios.scenario_complex_world import ApartmentsScenario


def draw_component(world, ax, reference_scale=10):
    world.draw_world(ax)
    padding = (world.dims - world.origin) * 0.0
    ax.set_xlim(world.origin[0] + padding[0], world.origin[0] + world.dims[0] - padding[0])
    ax.set_ylim(world.origin[1] + padding[1], world.origin[1] + world.dims[1] - padding[1])


@matplotlib.rc_context({
    "axes.titlesize": "small",
    "axes.titleweight": "bold",
})
def draw_scenario(population_size=100, config_file="base_configuration.py", world=True, inventory=True):
    experiment_config = Configuration(population_size=population_size, config_file=config_file)
    population = AgentList(population_size)
    apply_alternate_config(experiment_config["scenario"], available_scenarios["cw_ohb"])
    scenario_cfg = experiment_config.get("scenario", {})
    scenario_class = experiment_config.get("class", ApartmentsScenario)
    scenario_parameters = scenario_cfg.get("kwargs", {})
    scenario = scenario_class(population, **scenario_parameters)
    bus = BusMBCitaroK(orientation="horizontal")

    w, h = (scenario.world.width, scenario.world.height)
    page_width = 8.5  # in inches
    margins = 1.5  # in inches
    figure_width = page_width - margins
    figure_height = figure_width * h / w
    # fig = plt.figure(figsize=(figure_width, figure_height))

    if not world and not inventory:
        raise ValueError("'world' or 'inventory' must be True")

    labels = list("abcdef")
    if world and inventory:
        mosaic = [['world', "world", "world"],
                  ['world', "world", "world"],
                  ['home', 'bus', "bus"],
                  ['office', 'bar', "restaurant"]]
    elif world:
        labels[0] = ""
        figure_height /= 2
        mosaic = [['world', "world", "world"],
                  ['world', "world", "world"]]
    else:
        figure_height /= 2
        mosaic = [['home', 'bus', "bus"],
                  ['office', 'bar', "restaurant"]]

    fig, axd = plt.subplot_mosaic(mosaic,
                                  figsize=(figure_width, figure_height * 2),
                                  gridspec_kw={"left": 0.01, "right": .99, "top": 1, "bottom": 0.03})

    for ax in axd.values():
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_frame_on(False)
        ax.set_aspect("equal")

    if world:
        scenario.draw_world(axd["world"], 0.01, draw_population=False)
        scale_bar(axd["world"], reference_div=10)
        if labels[0] != "":
            label = f"{labels.pop(0)}. "
            axd["world"].set_title(f"{label}World infrastructure for 100 simulated agents", loc="left", y=0.925, )

    if inventory:
        draw_component(scenario.homes[0], axd["home"])
        draw_component(scenario.offices[0], axd["office"])
        draw_component(bus, axd["bus"], 1)
        draw_component(scenario.bar, axd["bar"])
        draw_component(scenario.restaurants, axd["restaurant"])

    plt.show(block=False)

    if inventory:
        for label, ax_name in zip(labels, ['home', 'bus', 'office', "bar", "restaurant"]):
            scale_bar(axd[ax_name], reference_div=5 if ax_name == "bus" else 10)
            if ax_name == "office":
                axd[ax_name].set_title(f"{label}. Office/School", loc="center", y=1.0)
            else:
                axd[ax_name].set_title(f"{label}. {ax_name.title()}", loc="center", y=1.0)

    return fig
