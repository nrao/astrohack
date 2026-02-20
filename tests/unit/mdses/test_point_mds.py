import shutil
import matplotlib

from toolviper.utils import data

from astrohack import AstrohackPointFile, open_pointing
from astrohack.utils.verification_tools import (
    add_data_folder_to_names_in_class,
    are_png_files_close,
)

matplotlib.use("Agg")


class TestPointMDS:
    data_dir = "point_data"
    destination_folder = "point_exports"
    ref_products_name = f"ref_point_products"

    pnt_name = "ea25_cal_small_before_reference.point.zarr"

    ant_id = "ea25"
    ant_key = f"ant_{ant_id}"

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        data.download(file=cls.pnt_name, folder=cls.data_dir)
        data.download(file=cls.ref_products_name, folder=cls.data_dir)

        add_data_folder_to_names_in_class(cls)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.data_dir)
        shutil.rmtree(cls.destination_folder)
        return

    def test_init(self):
        pnt_mds = AstrohackPointFile(self.pnt_name)
        assert isinstance(pnt_mds, AstrohackPointFile)

    def test_plot_exports(self):
        pnt_mds = open_pointing(self.pnt_name)

        pnt_mds.plot_array_configuration(self.destination_folder)
        plot_name = "point_array_configuration.png"
        assert are_png_files_close(
            f"{self.destination_folder}/{plot_name}",
            f"{self.ref_products_name}/{plot_name}",
        ), "Array configuration plot is failing closeness test"

        pnt_mds.plot_pointing_in_time(
            self.destination_folder, plot_antennas_separately=False
        )
        plot_name = "point_directional_cosines_combined.png"
        assert are_png_files_close(
            f"{self.destination_folder}/{plot_name}",
            f"{self.ref_products_name}/{plot_name}",
        ), "All antennas combined directional cosines plot is failing closeness test"

        pnt_mds.plot_pointing_in_time(
            self.destination_folder,
            plot_antennas_separately=True,
            ant=self.ant_id,
        )
        plot_name = "point_directional_cosines_ant_ea25.png"
        assert are_png_files_close(
            f"{self.destination_folder}/{plot_name}",
            f"{self.ref_products_name}/{plot_name}",
        ), "All antennas combined directional cosines plot is failing closeness test"
