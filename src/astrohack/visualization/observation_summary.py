from astrohack.utils import (
    format_observation_summary,
    make_header,
)


def generate_observation_summary(parm_dict):
    antenna = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    data_type = parm_dict["dtype"]
    xds = parm_dict["xdt_data"]
    obs_sum = xds.attrs["summary"]
    tab_size = parm_dict["tab_size"]

    tab_count = 1
    spc = " "

    if data_type == "holog":
        map_id = parm_dict["this_map"]
        header = f"{antenna}, {ddi}, {map_id}"
    else:
        header = f"{antenna}, {ddi}"

    outstr = make_header(header, "#", 60, 3)

    outstr += (
        format_observation_summary(
            obs_sum,
            tab_size,
            tab_count,
            az_el_key=parm_dict["az_el_key"],
            phase_center_unit=parm_dict["phase_center_unit"],
            az_el_unit=parm_dict["az_el_unit"],
            time_format=parm_dict["time_format"],
        )
        + "\n"
    )

    if data_type == "beamcut":
        for cut in xds.children.values():
            outstr += f"{tab_count*tab_size*spc}{cut.name}:\n"
            outstr += f'{(tab_count+1)*tab_size*spc}{cut.attrs["direction"]} at {cut.attrs["time_string"]} UTC\n\n'

    return outstr
