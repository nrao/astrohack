import os
import warnings
from importlib.metadata import version

__version__ = version("astrohack")

# This installs a slick, informational tracebacks logger
from rich.traceback import install
from toolviper.utils.logger import setup_logger

from astrohack.antenna.telescope import get_proper_telescope as get_proper_telescope
from astrohack.beamcut import beamcut as beamcut
from astrohack.cassegrain_ray_tracing import (
    apply_holog_phase_fitting_to_rt_xds as apply_holog_phase_fitting_to_rt_xds,
)
from astrohack.cassegrain_ray_tracing import (
    cassegrain_ray_tracing_pipeline as cassegrain_ray_tracing_pipeline,
)
from astrohack.cassegrain_ray_tracing import (
    create_ray_tracing_telescope_parameter_dict as create_ray_tracing_telescope_parameter_dict,
)
from astrohack.cassegrain_ray_tracing import (
    plot_2d_maps_from_rt_xds as plot_2d_maps_from_rt_xds,
)
from astrohack.cassegrain_ray_tracing import (
    plot_radial_projection_from_rt_xds as plot_radial_projection_from_rt_xds,
)
from astrohack.combine import combine as combine
from astrohack.estimate_nf_distance import estimate_nf_distance as estimate_nf_distance
from astrohack.extract_alma_nf_holog import (
    extract_alma_nf_holog as extract_alma_nf_holog,
)
from astrohack.extract_holog import extract_holog as extract_holog
from astrohack.extract_holog import generate_holog_obs_dict as generate_holog_obs_dict
from astrohack.extract_locit import extract_locit as extract_locit
from astrohack.extract_pointing import extract_pointing as extract_pointing
from astrohack.fringefit_locit import fringefit_locit as fringefit_locit
from astrohack.holog import holog as holog
from astrohack.image_comparison_tool import compare_fits_images as compare_fits_images
from astrohack.image_comparison_tool import (
    rms_table_from_zarr_datatree as rms_table_from_zarr_datatree,
)
from astrohack.io.beamcut_mds import AstrohackBeamcutFile as AstrohackBeamcutFile
from astrohack.io.dio import open_beamcut as open_beamcut
from astrohack.io.dio import open_holog as open_holog
from astrohack.io.dio import open_image as open_image
from astrohack.io.dio import open_locit as open_locit
from astrohack.io.dio import open_panel as open_panel
from astrohack.io.dio import open_pointing as open_pointing
from astrohack.io.dio import open_position as open_position
from astrohack.io.holog_mds import AstrohackHologFile as AstrohackHologFile
from astrohack.io.image_mds import AstrohackImageFile as AstrohackImageFile
from astrohack.io.locit_mds import AstrohackLocitFile as AstrohackLocitFile
from astrohack.io.panel_mds import AstrohackPanelFile as AstrohackPanelFile
from astrohack.io.point_mds import AstrohackPointFile as AstrohackPointFile
from astrohack.io.position_mds import AstrohackPositionFile as AstrohackPositionFile
from astrohack.locit import locit as locit
from astrohack.panel import panel as panel
from astrohack.simulate_alma_nf_holog import (
    simulate_alma_nf_holog as simulate_alma_nf_holog,
)

install(show_locals=False)

# astrohack writes consolidated metadata on purpose; the Zarr format 3
# specification does not cover it yet.
warnings.filterwarnings(
    "ignore",
    message="Consolidated metadata is currently not part in the Zarr format 3 specification",
)

if not os.getenv("LOGGER_NAME"):
    os.environ["LOGGER_NAME"] = "astrohack"
    setup_logger(
        logger_name="astrohack",
        log_to_term=True,
        log_to_file=False,
        log_file="astrohack-logfile",
        log_level="INFO",
    )
