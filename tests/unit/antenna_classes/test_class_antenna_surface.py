from astrohack.antenna.antenna_surface import AntennaSurface
from astrohack.antenna.telescope import get_proper_telescope
from astrohack import extract_holog, extract_pointing, holog, open_panel, open_image
from astrohack.utils.conversion import convert_unit

import numpy as np
import toolviper
import shutil
import xarray as xr

from astrohack.utils.verification_tools import add_data_folder_to_names_in_class

datafolder = "paneldata/"


class TestClassAntennaSurface:
    data_dir = "ant_class_data"
    img_name = "ea25_cal_before_reference.image.zarr"

    ant_key = "ant_ea25"
    ddi_key = "ddi_0"

    datashape = (510, 510)
    middle_pix = 255
    tolerance = 1e-6
    sigma = 20
    rand = sigma * np.random.randn(*datashape)
    zero = np.zeros(datashape)

    @classmethod
    def setup_class(cls):
        toolviper.utils.data.download(cls.img_name, cls.data_dir)

        add_data_folder_to_names_in_class(cls)

        input_xds = open_image(cls.img_name)[cls.ant_key][cls.ddi_key].dataset
        input_xds.attrs["ant_name"] = "ea00"
        input_xds.attrs["ddi"] = "test"

        cls.tant = AntennaSurface(input_xds, panel_margins=0.2)

        return

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        shutil.rmtree(cls.data_dir)

    def test_init(self):
        """
        Tests the initialization of a AntennaSurface object
        """
        assert np.isnan(self.tant.in_rms), "RMS is not properly initialized"
        assert np.isnan(self.tant.ingains), "Gains are not properly initialized"
        assert self.tant.telescope.ringed, "Currently only ringed telescopes supported"
        assert self.tant.panelmodel == "rigid", "Default panel kind should be rigid"
        # Tests _build_polar
        assert (
            self.tant.rad.shape == self.datashape
        ), "Radius image does not have the expected dimensions"
        assert abs(self.tant.rad[self.middle_pix, self.middle_pix]) < 15e-1, (
            "Radius at the center of the image " "is more than 15 cm from zero"
        )
        assert (
            abs(
                self.tant.phi[self.middle_pix, int(3 * self.datashape[0] / 4)]
                - np.pi / 2
            )
            / np.pi
            < 0.01
        ), "Azimuth at the horizontal axis is more than 1% different from pi/2"
        # tests _build_ring_panels
        assert len(self.tant.panels) == np.sum(self.tant.telescope.n_panel_per_ring), (
            "Number of panels do not " "match the expected number"
        )
        # tests _build_ring_mask
        assert (
            self.tant.mask.shape == self.datashape
        ), "Mask image does not have the expected dimensions"
        assert not self.tant.mask[
            0, 0
        ], "Mask is True at edges, where it should be False"

    def test_fit_surface(self):
        """
        Tests that fitting results for two panels match the reference
        """
        expected_len = 3
        solved_pars = [
            [0, [-0.00050669, 0.00029443, -0.00094982]],
            [30, [-0.00031, -0.00024618, -0.00012443]],
        ]
        self.tant.fit_surface()

        assert len(self.tant.panels[0].model.parameters) == expected_len, (
            "Fitted results have a different length" " from reference"
        )
        for idx, solved_par_list in solved_pars:
            assert np.allclose(
                self.tant.panels[idx].model.parameters,
                solved_par_list,
                atol=self.tolerance,
            ), f"Fitting results for Panel {idx} do not match reference within tolerance"

    def test_correct_surface(self):
        """
        Tests that surface corrections and residuals combined properly reconstruct the original deviations
        """
        self.tant.correct_surface()
        reconstruction = self.tant.residuals - self.tant.corrections
        assert (
            np.nansum((reconstruction - self.tant.deviation)[self.tant.mask])
            < self.tolerance
        ), "Reconstruction is not faithful to original data"

    def test_gains_array(self):
        """
        Tests gain computations by using a zero array and a random array
        """
        self.tant.phase = self.zero
        z_gains = self.tant.gains()
        # If the antenna has not been corrected, gains returns a [2] list, if it has been corrected it returns a [2,2]
        # list containing the corrected gains. This try and except assures that this test works in both situations.
        try:
            len(z_gains[0])
            assert z_gains[0][0] == z_gains[0][1]
        except TypeError:
            assert (
                z_gains[0] == z_gains[1]
            ), "Theoretical gains should be equal to real gains for a perfect antenna"
        self.tant.phase = self.rand
        r_gains = self.tant.gains()
        assert (
            r_gains[0] < r_gains[1]
        ), "Real gains need to be inferior to theoretical gains on a noisy surface"

    def test_get_rms(self):
        """
        Tests RMS computations by using a zero array and a random array
        """
        self.tant.residuals = self.zero
        z_rms = self.tant.get_rms()
        assert z_rms[1] == 0, "RMS should be zero when computed over a zero array"
        self.tant.residuals = self.rand
        self.tant.mask[:, :] = True
        fac = convert_unit("mm", "m", "length")
        r_rms = self.tant.get_rms()[1] * fac
        assert (
            abs(r_rms - self.sigma) / self.sigma < 0.01
        ), "Computed RMS does not match expected RMS within 1%"
