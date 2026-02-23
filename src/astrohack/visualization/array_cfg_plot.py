import numpy as np

from astrohack.utils import convert_unit
from astrohack.utils.constants import fontsize
from astrohack.utils.tools import get_telescope_lat_lon_rad
from astrohack.utils.algorithms import compute_antenna_relative_off
from astrohack.antenna import get_proper_telescope
from astrohack.visualization.plot_tools import (
    create_figure_and_axes,
    close_figure,
    plot_boxes_limits_and_labels,
)


def plot_array_configuration(input_dict, xdtree, caller, box_default_size=0.2):
    telescope_name = xdtree.attrs["telescope_name"]
    telescope = get_proper_telescope(telescope_name)
    stations = input_dict["stations"]
    display = input_dict["display"]
    figure_size = input_dict["figure_size"]
    dpi = input_dict["dpi"]
    filename = f"{input_dict["destination"]}/{caller}_array_configuration.png"
    length_unit = input_dict["unit"]
    box_size = input_dict["box_size"]  # In user input unit
    plot_zoff = input_dict["zoff"]

    len_fac = convert_unit("m", length_unit, "length")
    tel_lon, tel_lat, tel_rad = get_telescope_lat_lon_rad(telescope)
    ant_offs = []
    ant_texts = []
    for ant_xdtree in xdtree.values():
        ant_info = ant_xdtree.attrs["antenna_info"]
        ew_off, ns_off, el_off, _ = compute_antenna_relative_off(
            ant_info, tel_lon, tel_lat, tel_rad, len_fac
        )
        text = f'  {ant_info["name"]}'
        if stations:
            text += f'@{ant_info["station"]}'
        if plot_zoff:
            text += f" {el_off:.1f} {length_unit}"
        ant_offs.append([ew_off, ns_off])
        ant_texts.append(text)

    ant_offs = np.array(ant_offs)
    if box_size is None:
        ew_min, ew_max = np.min(ant_offs[:, 0]), np.max(ant_offs[:, 0])
        ew_range = ew_max - ew_min
        ns_min, ns_max = np.min(ant_offs[:, 1]), np.max(ant_offs[:, 1])
        ns_range = ns_max - ns_min
        box_size = box_default_size * np.max([ew_range, ns_range])

    fig, axes = create_figure_and_axes(figure_size, [1, 2], default_figsize=[10, 5])
    inner_ax = axes[1]
    outer_ax = axes[0]
    for i_ant, text in enumerate(ant_texts):
        ew_off, ns_off = ant_offs[i_ant]
        plot_one_antenna_position(outer_ax, inner_ax, ew_off, ns_off, text, box_size)

    # axes labels
    xlabel = f"East [{length_unit}]"
    ylabel = f"North [{length_unit}]"

    plot_boxes_limits_and_labels(
        outer_ax, inner_ax, xlabel, ylabel, box_size, "Outer array", "Inner array"
    )

    title = f"{len(xdtree.keys())} antennas during observation"
    close_figure(fig, title, filename, dpi, display)
    return


def plot_one_antenna_position(
    outerax, innerax, xpos, ypos, text, box_size, marker="+", color="black"
):
    """
    Plot an antenna to either the inner or outer array boxes
    Args:
        outerax: Plotting axis for the outer array box
        innerax: Plotting axis for the inner array box
        xpos: X antenna position (east-west)
        ypos: Y antenna position (north-south)
        text: Antenna label
        box_size: Size of the inner array box
        marker: Antenna position marker
        color: Color for the antenna position marker
    """
    half_box = box_size / 2
    if abs(xpos) > half_box or abs(ypos) > half_box:
        outerax.plot(xpos, ypos, marker=marker, color=color)
        outerax.text(xpos, ypos, text, fontsize=fontsize, ha="left", va="center")
    else:
        outerax.plot(xpos, ypos, marker=marker, color=color)
        innerax.plot(xpos, ypos, marker=marker, color=color)
        innerax.text(xpos, ypos, text, fontsize=fontsize, ha="left", va="center")
