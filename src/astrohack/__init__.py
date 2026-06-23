import os
from importlib.metadata import version

__version__ = version("astrohack")

from .extract_holog import (
    extract_holog,
    generate_holog_obs_dict,
)
from .beamcut import beamcut
from .extract_pointing import extract_pointing
from .holog import holog
from .panel import panel
from .combine import combine
from .locit import locit
from .extract_locit import extract_locit
from .cassegrain_ray_tracing import (
    cassegrain_ray_tracing_pipeline,
    create_ray_tracing_telescope_parameter_dict,
    plot_2d_maps_from_rt_xds,
    plot_radial_projection_from_rt_xds,
    apply_holog_phase_fitting_to_rt_xds,
)
from .image_comparison_tool import compare_fits_images, rms_table_from_zarr_datatree
from .antenna.telescope import get_proper_telescope

# This installs a slick, informational tracebacks logger
from rich.traceback import install
from toolviper.utils.logger import setup_logger

install(show_locals=False)

if not os.getenv("LOGGER_NAME"):
    os.environ["LOGGER_NAME"] = "astrohack"
    setup_logger(
        logger_name="astrohack",
        log_to_term=True,
        log_to_file=False,
        log_file="astrohack-logfile",
        log_level="INFO",
    )
