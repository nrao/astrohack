import json
import toolviper
import shutil

import numpy as np

from astrohack.core.holog_obs_dict import HologObsDict
from astrohack.io.dio import open_panel
from astrohack.extract_holog import extract_holog
from astrohack.extract_pointing import extract_pointing
from astrohack.holog import holog
from astrohack.panel import panel
from astrohack.utils.conversion import convert_unit
from astrohack.utils.verification_tools import relative_difference

base_name = "ea25_cal_small_"


def verify_panel_shifts(
    before_panel_filename,
    after_panel_filename,
    panel_list=None,
    expected_shift=np.array([-100, 75, 0, 150]),
    ref_mean_shift=np.array([-91.47636864, 60.34743659, 4.16119043, 122.40537789]),
    antenna="ant_ea25",
    ddi="ddi_0",
):
    if panel_list is None:
        panel_list = ["3-4", "5-27", "5-37", "5-38"]
    m_to_mils = convert_unit("m", "mils", "length")

    before_mds = open_panel(before_panel_filename)
    after_mds = open_panel(after_panel_filename)

    before_shift = (
        before_mds[antenna][ddi].dataset.sel(labels=panel_list).PANEL_SCREWS.values
        * m_to_mils
    )
    after_shift = (
        after_mds[antenna][ddi].dataset.sel(labels=panel_list).PANEL_SCREWS.values
        * m_to_mils
    )

    difference = after_shift - before_shift

    mean_shift = np.mean(difference, axis=1)
    print(mean_shift)

    delta_mean_shift = np.abs(mean_shift - expected_shift)
    delta_ref_shift = np.abs(ref_mean_shift - expected_shift)

    # New corrections - old corrections --> delta if delta < 0 ==> we improved.
    relative_shift = relative_difference(delta_mean_shift, delta_ref_shift)
    print(relative_shift)

    return np.all(relative_shift < 6e-2)


def verify_center_pixels(
    image_filename, antenna, ddi, reference_center_pixels, tolerance=1e-6
):
    from astrohack.io.dio import open_image

    mds = open_image(image_filename)[antenna][ddi]

    aperture_shape = mds.APERTURE.values.shape[-2], mds.APERTURE.values.shape[-1]
    beam_shape = mds.BEAM.values.shape[-2], mds.BEAM.values.shape[-1]

    aperture_center_pixels = np.squeeze(
        mds.APERTURE.values[..., aperture_shape[0] // 2, aperture_shape[1] // 2]
    )
    beam_center_pixels = np.squeeze(
        mds.BEAM.values[..., beam_shape[0] // 2, beam_shape[1] // 2]
    )

    aperture_ref = list(map(complex, reference_center_pixels["aperture"]))
    beam_ref = list(map(complex, reference_center_pixels["beam"]))
    real_check = True
    imag_check = True

    for i in range(len(aperture_ref)):
        aperture_check = (
            relative_difference(aperture_ref[i].real, aperture_center_pixels[i].real)
            < tolerance
        )

        beam_check = (
            relative_difference(beam_ref[i].real, beam_center_pixels[i].real)
            < tolerance
        )

        real_check = real_check and (aperture_check and beam_check)

        aperture_check = (
            relative_difference(aperture_ref[i].imag, aperture_center_pixels[i].imag)
            < tolerance
        )

        beam_check = (
            relative_difference(beam_ref[i].imag, beam_center_pixels[i].imag)
            < tolerance
        )

        imag_check = imag_check and (aperture_check and beam_check)

    return real_check and imag_check


class TestStakeholder:
    data_dir = "stakeholder_test_data"
    before_ms = "ea25_cal_small_before_fixed.split.ms"
    after_ms = "ea25_cal_small_after_fixed.split.ms"
    before_root = "ea25_before."
    after_root = "ea25_after."
    ext_holog_dict_ref = "extract_holog_verification.json"
    hol_num_dict_ref = "holog_numerical_verification.json"
    extensions = {
        "pnt": "point.zarr",
        "hlg": "holog.zarr",
        "img": "image.zarr",
        "pnl": "panel.zarr",
    }

    @classmethod
    def get_name(cls, is_before, ext):
        out_name = f"{cls.data_dir}/"
        if is_before:
            out_name += f"{cls.before_root}"
        else:
            out_name += f"{cls.after_root}"
        out_name += cls.extensions[ext]
        return out_name

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""

        # Data files
        toolviper.utils.data.download(file=cls.before_ms, folder=cls.data_dir)
        toolviper.utils.data.download(file=cls.after_ms, folder=cls.data_dir)

        # Verification json information
        toolviper.utils.data.download(file=cls.ext_holog_dict_ref, folder=cls.data_dir)
        toolviper.utils.data.download(file=cls.hol_num_dict_ref, folder=cls.data_dir)

        for ms_name, status in [[cls.before_ms, True], [cls.after_ms, False]]:
            ms_name = cls.data_dir + "/" + ms_name
            pnt_name = cls.get_name(status, "pnt")
            hlg_name = cls.get_name(status, "hlg")
            img_name = cls.get_name(status, "img")
            pnl_name = cls.get_name(status, "pnl")

            extract_pointing(
                ms_name=ms_name,
                point_name=pnt_name,
                parallel=False,
                overwrite=True,
            )

            extract_holog(
                ms_name=ms_name,
                point_name=pnt_name,
                holog_name=hlg_name,
                data_column="CORRECTED_DATA",
                parallel=False,
                overwrite=True,
            )

            holog(
                holog_name=hlg_name,
                image_name=img_name,
                overwrite=True,
                parallel=False,
            )

            panel(
                image_name=img_name,
                panel_name=pnl_name,
                panel_model="rigid",
                parallel=False,
                overwrite=True,
            )

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        shutil.rmtree(cls.data_dir, ignore_errors=True)

    def test_holog_obs_dict(self):
        ref_json = f"{self.data_dir}/{self.ext_holog_dict_ref}"
        with open(ref_json, "r") as ref_json_file:
            full_ref_dict = json.load(ref_json_file)

        for epoch in ["before", "after"]:
            is_before = epoch == "before"
            hlg_name = self.get_name(is_before, "hlg")

            test_hlg_obs_dict = HologObsDict.from_holog_file(hlg_name)
            ref_hlg_obs_dict = HologObsDict(full_ref_dict["vla"][epoch])
            assert (
                test_hlg_obs_dict == ref_hlg_obs_dict
            ), f"Verify {epoch} holog obs dictionary"

    def test_center_pixels(self):
        ref_json = f"{self.data_dir}/{self.hol_num_dict_ref}"
        with open(ref_json, "r") as ref_json_file:
            full_ref_dict = json.load(ref_json_file)

        for epoch in ["before", "after"]:
            is_before = epoch == "before"
            img_name = self.get_name(is_before, "img")

            assert verify_center_pixels(
                image_filename=img_name,
                antenna="ant_ea25",
                ddi="ddi_0",
                reference_center_pixels=full_ref_dict["vla"]["pixels"][epoch],
                tolerance=1.5e-6,
            ), f"Verifiy center pixels {epoch}"

    def test_panel_shifts(self):
        ref_json = f"{self.data_dir}/{self.hol_num_dict_ref}"
        with open(ref_json, "r") as ref_json_file:
            full_ref_dict = json.load(ref_json_file)

        bef_pnl_name = self.get_name(True, "pnl")
        aft_pnl_name = self.get_name(False, "pnl")

        assert verify_panel_shifts(
            before_panel_filename=bef_pnl_name,
            after_panel_filename=aft_pnl_name,
            ref_mean_shift=full_ref_dict["vla"]["offsets"],
        ), "Verify panel shifts"
