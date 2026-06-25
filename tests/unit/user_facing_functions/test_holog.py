import os
import pytest
import pathlib
import shutil
import toolviper

import numpy as np

from astrohack import open_image
from astrohack.holog import holog
from astrohack.utils.verification_tools import (
    add_data_folder_to_names_in_class,
    execute_cleanup,
)


class TestHolog:
    data_dir = "holog_data"
    hlg_name = "ea25_cal_small_before_reference.holog.zarr"

    def_img_name = "ea25_cal_small_before_reference.image.zarr"
    ref_img_name = "ea25_cal_before_reference.image.zarr"

    ant_id = "ea25"
    ddi_id = 0
    ant_key = f"ant_{ant_id}"
    ddi_key = f"ddi_{ddi_id}"

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        toolviper.utils.data.download(file=cls.hlg_name, folder=cls.data_dir)
        toolviper.utils.data.download(file=cls.ref_img_name, folder=cls.data_dir)

        add_data_folder_to_names_in_class(cls)

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        if execute_cleanup():
            shutil.rmtree(cls.data_dir)

    def test_defaults(self):
        """
        test holog defaults
        """
        new_img_mds = holog(
            holog_name=self.hlg_name,
            overwrite=True,
        )
        assert pathlib.Path(
            self.def_img_name
        ).is_dir(), f"A .image.zarr file named {self.def_img_name} does not exist."

        ref_img_mds = open_image(self.ref_img_name)
        assert new_img_mds.is_close_to(
            ref_img_mds
        ), "Reference and new mdses are different."

    def test_data_selection(self):
        image_mds = holog(
            holog_name=self.hlg_name,
            ant=self.ant_id,
            ddi=self.ddi_id,
            overwrite=True,
        )

        ant_list = list(image_mds.keys())
        assert len(ant_list) == 1, "A single antenna should be present."
        assert (
            ant_list[0] == self.ant_key
        ), "Ant name should be the same as the one given."

        ddi_list = list(image_mds[self.ant_key].keys())
        assert len(ddi_list) == 1, "A single ddi should be present."
        assert (
            ddi_list[0] == self.ddi_key
        ), "DDI key should be the same as the one given."

    def test_padding_factor(self):
        """
        Specify a padding factor to use in the image creation; check that image size is created.
        """

        pad_list = [[5, 256], [10, 512]]

        for pad_fac, ap_size in pad_list:
            image_mds = holog(
                holog_name=self.hlg_name,
                padding_factor=pad_fac,
                ant=self.ant_id,
                ddi=self.ddi_id,
                overwrite=True,
                parallel=False,
            )
            ap_shape = (
                1,
                1,
                4,
                ap_size,
                ap_size,
            )
            for ant_key in image_mds.keys():
                for ddi_key in image_mds[ant_key].keys():
                    this_ap_shape = image_mds[ant_key][ddi_key].APERTURE.shape
                    assert this_ap_shape == ap_shape, (
                        f"Aperture for {ant_key} {ddi_key} for a padding factor of {pad_fac} should be {ap_shape} "
                        f"but is {this_ap_shape}."
                    )

    def test_chan_average(self):
        """
        Check that channel average flag was set holog is run.
        """
        ref_nchan = 1
        image_mds = holog(
            holog_name=self.hlg_name,
            chan_average=True,
            ant=self.ant_id,
            ddi=self.ddi_id,
            overwrite=True,
            parallel=False,
        )

        for ant_key in image_mds.keys():
            for ddi_key in image_mds[ant_key].keys():
                this_ap_shape = image_mds[ant_key][ddi_key].APERTURE.shape
                assert (
                    this_ap_shape[1] == ref_nchan
                ), f"Non chan_average aperture for {ant_key} {ddi_key} should have {ref_nchan} channels"

    def test_to_stokes(self):
        """
        Check that to_stokes flag was set holog is run.
        """
        stokes_axis = np.array(["I", "Q", "U", "V"])
        image_mds = holog(
            holog_name=self.hlg_name,
            ant=self.ant_id,
            ddi=self.ddi_id,
            to_stokes=True,
            overwrite=True,
            parallel=False,
        )
        pol_axis = image_mds[self.ant_key][self.ddi_key].pol.values
        assert np.all(
            pol_axis == stokes_axis
        ), f"If to_stokes is set to True output data should have {stokes_axis} as the polarization axis."

    def test_overwrite(self):
        """
        Specify the output file should be overwritten; check that it WAS.
        """
        initial_time = os.path.getctime(self.def_img_name)

        holog(
            holog_name=self.hlg_name,
            image_name=self.def_img_name,
            ant=self.ant_id,
            ddi=self.ddi_id,
            overwrite=True,
            parallel=False,
        )
        modified_time = os.path.getctime(self.def_img_name)
        assert (
            initial_time != modified_time
        ), "Recreated file has to have a different time from the original file."

        with pytest.raises(FileExistsError):
            holog(
                holog_name=self.hlg_name,
                image_name=self.def_img_name,
                ant=self.ant_id,
                ddi=self.ddi_id,
                overwrite=False,
                parallel=False,
            )

    def test_perturbation_phase_fit(self):
        image_mds = holog(
            holog_name=self.hlg_name,
            image_name=self.def_img_name,
            phase_fit_engine="perturbations",
            ant=self.ant_id,
            ddi=self.ddi_id,
            grid_size=[31, 31],
            cell_size=[-0.0006386556122807017, 0.0006386556122807017],
            overwrite=True,
            parallel=False,
        )
        keys = [
            "phase_offset",
            "x_cassegrain_offset",
            "x_focus_offset",
            "x_point_offset",
            "x_subreflector_tilt",
            "y_cassegrain_offset",
            "y_focus_offset",
            "y_point_offset",
            "y_subreflector_tilt",
            "z_focus_offset",
        ]
        references = [
            0.07578374993954257,
            -28.033780511487777,
            -1.9620592050595538,
            0.00016673100624246893,
            0.00036714280075938257,
            -22.752401475110595,
            -3.3596837733268057,
            0.00032344494918384674,
            -0.0006101436903899218,
            0.07222802059408939,
        ]

        pha_fit_res = image_mds[self.ant_key][self.ddi_key].attrs["phase_fitting"][
            "map_0"
        ]["14167000000.0"]["I"]

        for ikey, key in enumerate(keys):
            assert np.isclose(pha_fit_res[key]["value"], references[ikey]), (
                f"Phase fitting values differ from " f"reference for {key}"
            )

        image_mds = holog(
            holog_name=self.hlg_name,
            image_name=self.def_img_name,
            ant=self.ant_id,
            ddi=self.ddi_id,
            phase_fit_engine="perturbations",
            phase_fit_control=[False, False, False, False, False],
            overwrite=True,
            parallel=False,
        )
        pha_fit_res = image_mds[self.ant_key][self.ddi_key].attrs["phase_fitting"]

        assert (
            pha_fit_res is None
        ), "When phase_fit_control is a 5-way False tuple, phase fit results should be None"

        image_mds = holog(
            holog_name=self.hlg_name,
            image_name=self.def_img_name,
            ant=self.ant_id,
            ddi=self.ddi_id,
            phase_fit_engine="perturbations",
            phase_fit_control=[False, True, False, True, False],
            overwrite=True,
            parallel=False,
        )

        pha_fit_res = image_mds[self.ant_key][self.ddi_key].attrs["phase_fitting"][
            "map_0"
        ]["14167000000.0"]["I"]

        assert np.isnan(
            pha_fit_res["x_point_offset"]["error"]
        ), "If pointing offset is not fitted x_point_offset error should be NaN"
        assert np.isnan(
            pha_fit_res["z_focus_offset"]["error"]
        ), "If focus is not fitted z_focus_offset error should be NaN"
        assert np.isnan(
            pha_fit_res["x_cassegrain_offset"]["error"]
        ), "If cassegrain offset is not fitted x_cassegrain_offset error should be NaN"

    def test_no_phase_fit(self):
        image_mds = holog(
            holog_name=self.hlg_name,
            image_name=self.def_img_name,
            phase_fit_engine="none",
            ant=self.ant_id,
            ddi=self.ddi_id,
            overwrite=True,
            parallel=False,
        )

        pha_fit_res = image_mds[self.ant_key][self.ddi_key].attrs["phase_fitting"]
        assert (
            pha_fit_res is None
        ), "When phase_fit_engine is set to 'none', phase fit results should be None"

    def test_zernike_phase_fitting(self):
        image_mds = holog(
            holog_name=self.hlg_name,
            image_name=self.def_img_name,
            ant=self.ant_id,
            ddi=self.ddi_id,
            grid_size=[31, 31],
            cell_size=[-0.0006386556122807017, 0.0006386556122807017],
            phase_fit_engine="zernike",
            zernike_n_order=4,
            overwrite=True,
            parallel=False,
        )

        pha_fit_res = image_mds[self.ant_key][self.ddi_key].attrs["phase_fitting"]

        assert pha_fit_res is None

        ref_phase = [
            [[125, 125], -0.17758619948993593],
            [[213, 430], -0.1459607430199923],
            [[432, 195], -0.034865251933011265],
        ]
        phase_img = image_mds[self.ant_key][self.ddi_key].CORRECTED_PHASE.values[
            0, 0, 0
        ]

        for idx, phase in ref_phase:
            assert np.isclose(
                phase_img[*idx], phase
            ), f"Phase is different from reference at {idx}"

    #
    def test_holog_zernike_coeffs(self):
        image_mds = holog(
            holog_name=self.hlg_name,
            image_name=self.def_img_name,
            ant=self.ant_id,
            ddi=self.ddi_id,
            grid_size=[31, 31],
            cell_size=[-0.0006386556122807017, 0.0006386556122807017],
            phase_fit_engine="none",
            zernike_n_order=10,
            overwrite=True,
            parallel=False,
        )
        ref_zernike_coeffs = [
            3.63853142e-01 + 3.22356554e-02j,
            4.23001476e00 + 4.11000920e01j,
            -1.75638037e01 - 6.11203902e01j,
            5.44624567e00 - 2.71889063e00j,
            -2.07518837e-01 - 1.07029883e-02j,
            -5.32112029e00 + 4.64863302e00j,
            -4.44881637e00 - 4.13055928e01j,
            4.26979653e00 + 3.99186278e01j,
            -8.98318910e00 - 3.19533343e01j,
            1.14296814e01 + 4.01677823e01j,
            3.22839441e00 - 1.57685943e00j,
            3.50888476e00 - 1.55562367e00j,
            -3.57103888e-02 + 1.19658639e-02j,
            -2.27098458e-02 + 2.00348932e-02j,
            9.35219135e-03 - 1.41071395e-02j,
            1.83504535e00 + 1.73453819e01j,
            -1.95266260e00 - 1.77442134e01j,
            1.85819383e00 + 1.71604139e01j,
            -1.83994376e-01 - 7.02351715e-01j,
            -1.17611308e00 - 4.27534468e00j,
            1.07348587e00 + 3.73876032e00j,
            2.45203816e00 - 1.12858846e00j,
            3.60288701e-01 - 2.00945408e-01j,
            7.99358267e-01 - 3.78177411e-01j,
            6.87383554e-04 - 3.05145445e-02j,
            -6.70306575e-02 + 2.45662583e-03j,
            2.51829138e-02 + 2.42719024e-03j,
            1.51297987e-03 - 2.09583627e-02j,
            -2.18272416e-01 - 2.49254874e00j,
            3.18745948e-01 + 2.90377897e00j,
            -3.48360089e-01 - 2.98705808e00j,
            2.82767637e-01 + 2.65015144e00j,
            -6.45378731e-02 - 2.32601543e-01j,
            -1.78847682e-01 - 6.28803632e-01j,
            1.98984840e-01 + 6.45562396e-01j,
            -5.75574487e-01 - 2.03482119e00j,
            -1.79071119e-01 + 9.75425501e-02j,
            1.33041282e-01 - 6.08107273e-02j,
            -1.04930214e-01 + 1.98600004e-02j,
            1.02529432e-01 - 4.56826111e-02j,
            -9.10502057e-02 + 4.00437485e-02j,
            -4.55895027e-02 + 3.88348143e-02j,
            4.98863504e-03 - 2.49634561e-02j,
            1.81918402e-02 + 1.18122844e-02j,
            -1.60219775e-02 - 2.28458691e-02j,
            1.47953573e-02 + 1.31553475e-01j,
            7.77135087e-03 + 3.36702956e-02j,
            2.47718592e-02 - 3.50457522e-02j,
            -2.67166187e-02 + 1.18180934e-02j,
            -1.46191264e-02 - 7.72276561e-02j,
            -3.00941394e-02 - 9.87108070e-02j,
            8.96918830e-03 - 2.84022568e-02j,
            -1.65357311e-03 - 4.38318775e-02j,
            1.50658709e-02 - 6.14483136e-02j,
            4.87811290e-02 + 1.75694507e-01j,
            -4.81118972e-03 + 3.73321157e-03j,
            4.60629264e-02 - 3.62603959e-02j,
            -1.21179544e-03 + 2.53104896e-02j,
            1.75069126e-02 + 1.04703864e-02j,
            8.61753013e-03 + 2.54347275e-03j,
            5.40815261e-02 - 2.62438937e-02j,
            4.45687952e-02 - 4.29741834e-02j,
            4.25176984e-02 + 4.59819354e-02j,
            -1.46957058e-02 - 3.46945617e-02j,
            3.38637874e-02 - 1.08213929e-02j,
            -2.06739538e-02 + 1.35215571e-03j,
        ]

        zer_coeffs = image_mds[self.ant_key][self.ddi_key].ZERNIKE_COEFFICIENTS.values[
            0, 0, 0
        ]

        expected_n_coeff = 66
        assert zer_coeffs.shape[0] == expected_n_coeff

        assert np.allclose(
            ref_zernike_coeffs, zer_coeffs
        ), "Fitted Zernike coefficients do not match references"
