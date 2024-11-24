from collections import namedtuple
from matplotlib import transforms
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def create_process_flow(ax):
    from dct_covid_bd_abm.visualisation.plots.plot_params import stage1RadarPlotProperties
    ax.axis("off")
    ax.set_title("Process Flow for Stage 1", fontweight="bold")
    ax.set_xlim(0, 1.1)
    face_colors = ["w", "k", "k", "w"]
    seen_values = []
    seen_colors = []
    Point = namedtuple("Point", "x, y")
    anchor = Point(0.012, 0.83)
    offset = Point(0.2, 0.2)
    box_width = 0.25
    box_height = 0.1
    for i, test in enumerate(["dropout", "coverage", "rs_boundary", "dct_intro"]):
        cm_ = stage1RadarPlotProperties[test]["cm"]
        color = cm_(0.3)
        title = stage1RadarPlotProperties[test]["title"]
        selected = stage1RadarPlotProperties[test]["selected"]
        box_x = anchor.x + i * offset.x
        box_y = anchor.y - i * offset.y
        ax.add_patch(FancyBboxPatch((box_x, box_y), box_width, box_height,
                                    color=color, boxstyle="Round, pad=0.01"))
        label_y = anchor.y + box_height / 2 - i * offset.y
        label_x = anchor.x + box_width / 2 + i * offset.x
        label = ax.text(label_x, label_y,
                        title, ha="center", va="center", color=face_colors[i], wrap=True)
        label._get_wrap_line_width = lambda: 100

        arrow_tail = Point(box_x + box_width, label_y)
        arrow_point = Point(box_x + offset.x + box_width / 2, box_y - offset.y + box_height)
        ax.add_patch(FancyArrowPatch(arrow_tail,
                                     arrow_point,
                                     arrowstyle="-|>",
                                     mutation_scale=15,
                                     connectionstyle="angle,angleA=0,angleB=90,rad=2"))

        seen_values.append(selected.split()[-1])
        seen_colors.append(cm_(float(selected.split()[-1])))
        label_marker = Point(arrow_point.x, (arrow_tail.y - arrow_point.y) / 2 + arrow_point.y)
        text = ax.text(*label_marker, " [", va="center", fontsize="small", fontweight="bold")
        width_pad_ = None
        height_pad = 0
        for v, color, pos in zip(seen_values, seen_colors, range(4)):
            # determine how far below the axis to place the first number
            text.draw(ax.figure.canvas.get_renderer())
            ex = text.get_window_extent()
            width_pad = ex.width
            if width_pad_ is None:
                width_pad_ = 0
            else:
                width_pad_ += ex.width

            if len(seen_values) > 3 and pos == 2:
                height_pad = ex.height
                width_pad -= width_pad_
            else:
                height_pad = 0

            tr = transforms.offset_copy(text.get_transform(), x=width_pad, y=-height_pad, units='dots')
            v_ = pos < len(seen_colors) - 1 and f"{v}, " or v
            text = ax.text(*label_marker, v_, transform=tr, color=color, va="center", fontsize="small",
                           fontweight="bold")

        text.draw(ax.figure.canvas.get_renderer())
        ex = text.get_window_extent()
        tr = transforms.offset_copy(text.get_transform(), x=ex.width, units='dots')
        ax.text(*label_marker, "]", transform=tr, va="center", fontsize="small", fontweight="bold")

        # params_label = f" [{', '.join(seen_values)}]"
        # if len(seen_values) > 3:
        #     params_label = f" [{', '.join(seen_values[:2])},\n  {', '.join(seen_values[2:])}]"

        # ax.text(*label_marker, params_label, va="center", fontsize="small")

    i = 4
    box_x = anchor.x + i * offset.x
    box_y = anchor.y - i * offset.y
    ax.add_patch(FancyBboxPatch((box_x, box_y), box_width, box_height,
                                color="C5", alpha=0.5, boxstyle="Round, pad=0.01"))

    label_y = anchor.y + box_height / 2 - i * offset.y
    label_x = anchor.x + box_width / 2 + i * offset.x
    label = ax.text(label_x, label_y, "Stage 2 Evaluation",
                    ha="center", va="center", wrap=True)
    label._get_wrap_line_width = lambda: 100
