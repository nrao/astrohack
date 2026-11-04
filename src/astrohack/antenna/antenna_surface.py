import numpy as np
import toolviper.utils.logger as logger
import xarray as xr
from matplotlib import patches

from astrohack.antenna.telescope import get_proper_telescope
from astrohack.utils.algorithms import (
    data_statistics,
    phase_wrapping,
)
from astrohack.utils.constants import clight, fourpi
from astrohack.utils.conversion import convert_unit, to_antenna_tapering
from astrohack.utils.fits import (
    put_axis_in_fits_header,
    put_resolution_in_fits_header,
    write_fits,
)
from astrohack.utils.text import (
    add_prefix,
    bool_to_str,
    create_dataset_label,
    create_informative_label_from_summary,
    format_frequency,
    format_value_unit,
    lnbr,
    spc,
    statistics_to_text,
    string_to_ascii_file,
)
from astrohack.visualization.plot_tools import (
    close_figure,
    compute_extent,
    create_figure_and_axes,
    get_proper_color_map,
    simple_imshow_map_plot,
    well_positioned_colorbar,
)

SUPPORTED_POL_STATES = ["I", "RR", "LL", "XX", "YY"]


class AntennaSurface:
    def __init__(
        self,
        inputxds,
        clip_type="sigma",
        clip_level=3,
        panel_model="rigid",
        nan_out_of_bounds=True,
        panel_margins=0.05,
        reread=False,
        pol_state="I",
        patch_phase=False,
        use_detailed_mask=True,
    ):
        """
        Antenna Surface description capable of computing RMS, Gains, and fitting the surface to obtain screw adjustments
        Args:
            inputxds: Input xarray dataset
            clip_type: Type of clipping to be applied to amplitude
            clip_level: Level of clipping
            panel_model: model of panel surface fitting, if is None defaults to telescope default
            nan_out_of_bounds: Should the region outside the dish be replaced with NaNs?
            panel_margins: Margin to be ignored at edges of panels when fitting, defaults to 20% if None
            reread: Read a previously processed holography
            pol_state: Polarization state to select
            patch_phase: Phase data from inputxds needs to be wrapped to -pi to pi interval
            use_detailed_mask: use detailed mask (arms shadows, ngvla outer panels)
        """
        self.reread = reread
        self.phase = None
        self.deviation = None
        self.residuals = None
        self.corrections = None
        self.phase_corrections = None
        self.phase_residuals = None
        self.solved = False
        self.in_gains = np.nan
        self.ou_gains = np.nan
        self.in_rms = np.nan
        self.out_rms = np.nan
        self.fitted = False
        self.pol_state = pol_state

        # Read the data in an XDS
        self._read_xds(inputxds)

        if patch_phase:
            self.phase = phase_wrapping(self.phase)

        self._create_aperture_mask(clip_type, clip_level, use_detailed_mask)
        self.deviation = self.telescope.phase_to_deviation(
            self.u_axis, self.v_axis, self.mask, self.phase, self.wavelength
        )

        self.panels = self.telescope.build_panel_list(panel_model, panel_margins)
        if self.reread:
            for i_panel, panel in enumerate(self.panels):
                panel.reread_solution(
                    self.panel_pars[i_panel, :], self.screw_adjustments[i_panel, :]
                )
        else:
            self.panel_model = panel_model
            self.panel_margins = panel_margins
            self.panel_distribution = self.telescope.attribute_pixels_to_panels(
                self.panels,
                self.u_axis,
                self.v_axis,
                self.rad,
                self.phi,
                self.deviation,
                self.mask,
            )

        if nan_out_of_bounds:
            self._nan_out_of_bounds()

    def _read_holog_xds(self, input_xds):
        if "chan" in input_xds.dims:
            if input_xds.sizes["chan"] != 1:
                raise RuntimeError("Only single channel holographies supported")
            self.wavelength = clight / input_xds.chan.values[0]
        else:
            self.wavelength = input_xds.attrs["wavelength"]

        if self.pol_state not in input_xds.coords["pol"]:
            msg = (
                f"Polarization state {self.pol_state} is not present in the data (available states: "
                f"{input_xds.coords['pol']})"
            )
            logger.error(msg)
            raise ValueError(msg)

        self.amplitude = (
            input_xds["AMPLITUDE"].sel(pol=self.pol_state).isel(time=0, chan=0).values
        )
        self.phase = (
            input_xds["CORRECTED_PHASE"]
            .sel(pol=self.pol_state)
            .isel(time=0, chan=0)
            .values
        )

        self.amp_unit = "V"
        self.u_axis = input_xds.u_prime.values
        self.v_axis = input_xds.v_prime.values

    def _read_panel_xds(self, input_xds):
        self.wavelength = input_xds.attrs["wavelength"]
        self.amp_unit = input_xds.attrs["amp_unit"]
        self.panel_model = input_xds.attrs["panel_model"]
        self.panel_margins = input_xds.attrs["panel_margin"]
        self.clip = input_xds.attrs["clip"]
        self.solved = input_xds.attrs["solved"]
        self.fitted = input_xds.attrs["fitted"]
        if "pol_state" not in input_xds.attrs:
            # Here I assume no one was doing panel fitting on something that is not Stokes I
            self.pol_state = "I"
        else:
            self.pol_state = input_xds.attrs["pol_state"]
        # Arrays
        self.amplitude = input_xds["AMPLITUDE"].values
        self.phase = input_xds["PHASE"].values
        self.deviation = input_xds["DEVIATION"].values
        self.mask = input_xds["MASK"].values
        self.u_axis = input_xds.u.values
        self.v_axis = input_xds.v.values
        self.panel_distribution = input_xds["PANEL_DISTRIBUTION"].values
        self.amplitude_noise = input_xds["AMP_NOISE"].values

        if self.solved:
            self.panel_fallback = input_xds["PANEL_FALLBACK"].values
            self.panel_model_array = input_xds["PANEL_MODEL"].values
            self.phase_residuals = input_xds["PHASE_RESIDUALS"].values
            self.residuals = input_xds["RESIDUALS"].values
            self.phase_corrections = input_xds["PHASE_CORRECTIONS"].values
            self.corrections = input_xds["CORRECTIONS"].values
            self.panel_pars = input_xds["PANEL_PARAMETERS"].values
            self.screw_adjustments = input_xds["PANEL_SCREWS"].values
            self.in_gains = [
                input_xds.attrs["input_gain"],
                input_xds.attrs["theoretical_gain"],
            ]
            self.ou_gains = [
                input_xds.attrs["output_gain"],
                input_xds.attrs["theoretical_gain"],
            ]
            self.panel_labels = input_xds.labels.values

    def _read_xds(self, input_xds):
        """
        Read input XDS, the reading function depending on if it is a reread or a new processing
        Args:
            input_xds: X array dataset
        """
        # Origin dependant reading

        if self.reread:
            self._read_panel_xds(input_xds)
        else:
            self._read_holog_xds(input_xds)

        # Common elements
        self.summary = input_xds.attrs["summary"]
        self.antenna_name = input_xds.attrs["summary"]["general"]["antenna name"]
        self.resolution = input_xds.summary["aperture"]["resolution"]
        self.ddi = input_xds.attrs["ddi"]
        self.label = create_dataset_label(self.antenna_name, input_xds.attrs["ddi"])
        self.telescope = get_proper_telescope(
            self.summary["general"]["telescope name"], self.antenna_name
        )

    def _define_amp_clip(self, clip_type, clip_level):
        self.amplitude_noise = np.where(self.base_mask, np.nan, self.amplitude)
        if clip_type is None or clip_type == "none":
            clip = np.nanmin(self.amplitude)
        elif clip_type == "relative":
            clip = clip_level * np.nanmax(self.amplitude)
        elif clip_type == "absolute":
            clip = clip_level
        elif clip_type == "sigma":
            noise_stats = data_statistics(self.amplitude_noise)
            clip = noise_stats["mean"] + clip_level * noise_stats["rms"]
        elif clip_type == "noise_threshold":
            clip = self._compute_noise_threshold_clip(clip_level)
        else:
            msg = f"Unrecognized clipping type: {clip_type}"
            raise ValueError(msg)
        return clip

    def _compute_noise_threshold_clip(self, threshold, step_multiplier=0.95):
        noise_stats = data_statistics(self.amplitude_noise)

        in_disk = np.where(self.rad < self.telescope.diameter / 2.0, 1.0, np.nan)
        in_disk = np.where(
            self.rad < self.telescope.inner_radial_limit, np.nan, in_disk
        )
        n_in_disk = np.nansum(in_disk)
        in_disk_amp = in_disk * self.amplitude

        fraction_in = 0
        clip = noise_stats["max"]
        multiplier = 1.0
        while fraction_in < threshold:
            clip *= multiplier
            data_in = np.where(in_disk_amp > clip, 1.0, 0.0)
            fraction_in = float(np.sum(data_in)) / n_in_disk
            multiplier *= step_multiplier

        return clip

    def _create_aperture_mask(self, clip_type, clip_level, use_detailed_mask):

        self.base_mask, self.rad, self.phi = self.telescope.create_aperture_mask(
            self.u_axis,
            self.v_axis,
            use_detailed_mask=use_detailed_mask,
            return_polar_meshes=True,
            use_outer_limit=True,
        )

        if self.reread:
            pass
        else:
            self.clip = self._define_amp_clip(clip_type, clip_level)
            self.mask = np.where(self.amplitude < self.clip, False, self.base_mask)
            self.mask = np.where(~np.isfinite(self.phase), False, self.mask)

    def _nan_out_of_bounds(self):
        self.phase = np.where(self.base_mask, self.phase, np.nan)
        self.amplitude = np.where(self.base_mask, self.amplitude, np.nan)
        self.deviation = np.where(self.base_mask, self.deviation, np.nan)

    def gains(self):
        """
        Computes antenna gains in decibels before and after panel surface fitting
        Returns:
        Gains before panel fitting OR Gains before and after panel fitting
        """
        self.in_gains = self.gain_at_wavelength(False, self.wavelength)
        if not self.solved:
            return self.in_gains

        else:
            self.ou_gains = self.gain_at_wavelength(True, self.wavelength)
            return self.in_gains, self.ou_gains

    def gain_at_wavelength(self, corrected, wavelength):
        # This is valid for the VLA not sure if valid for anything else...
        wavelength_scaling = self.wavelength / wavelength

        dish_mask = self.base_mask

        if corrected:
            if self.fitted:
                scaled_phase = wavelength_scaling * self.phase_residuals
            else:
                msg = (
                    "Cannot computed gains for corrected dish if panels are not fitted."
                )
                logger.error(msg)
                raise RuntimeError(msg)
        else:
            scaled_phase = wavelength_scaling * self.phase

        cossum = np.nansum(np.cos(scaled_phase[dish_mask]))
        sinsum = np.nansum(np.sin(scaled_phase[dish_mask]))
        real_factor = np.sqrt(cossum**2 + sinsum**2) / np.sum(dish_mask)

        theo_gain = fourpi * self.telescope.diameter / wavelength
        real_gain = theo_gain * real_factor
        return to_antenna_tapering(real_gain), to_antenna_tapering(theo_gain)

    def get_rms(self, unit="mm"):
        """
        Computes antenna surface RMS before and after panel surface fitting
        Returns:
        RMS before panel fitting OR RMS before and after panel fitting
        """
        fac = convert_unit("m", unit, "length")
        self.in_rms = self._compute_rms_array(self.deviation)
        if self.residuals is None:
            return fac * self.in_rms
        else:
            self.out_rms = self._compute_rms_array(self.residuals)
        return fac * self.in_rms, fac * self.out_rms

    def _compute_rms_array(self, array):
        """
        Factorized the computation of the RMS of an array
        Args:
            array: Input data array

        Returns:
            RMS of the input array
        """
        return np.sqrt(np.nanmean(array[self.mask] ** 2))

    def fit_surface(self):
        """
        Loops over the panels to fit the panel surfaces
        """
        panels = []
        for panel in self.panels:
            if not panel.solve():
                panels.append(panel.label)

        self.fitted = True
        if len(panels) > 0:
            msg = (
                f"{self.label}: Fit failed with the {self.panel_model} model and a simple mean has been used instead "
                f"for the following panels:"
            )

            logger.warning(msg)
            msg = str(panels)
            logger.warning(msg)

    def correct_surface(self):
        """
        Apply corrections determined by the panel surface fitting methods to the antenna surface
        """
        if not self.fitted:
            raise RuntimeError("Panels must be fitted before atempting a correction")
        self.corrections = np.where(self.mask, 0, np.nan)
        for panel in self.panels:
            corrections = panel.get_corrections()
            for corr in corrections:
                ix, iy = int(corr[0]), int(corr[1])
                self.corrections[ix, iy] = -corr[-1]

        self.residuals = self.deviation + self.corrections
        self.phase_corrections = self.telescope.deviation_to_phase(
            self.u_axis, self.v_axis, self.mask, self.corrections, self.wavelength
        )

        self.phase_residuals = self.telescope.deviation_to_phase(
            self.u_axis, self.v_axis, self.mask, self.residuals, self.wavelength
        )

        self._build_panel_data_arrays()
        self.solved = True

    def print_misc(self):
        """
        Print miscelaneous information on the panels in the antenna surface
        """
        for panel in self.panels:
            panel.print_misc()

    def plot_mask(self, basename, caller, cmap, parm_dict):
        """
        Plot mask used in the selection of points to be fitted
        Args:
            basename: basename for the plot, the prefix 'ancillary_mask' will be added to it
            caller: Which mds called this plotting function
            parm_dict: dictionary with plotting parameters
            cmap: colormap name
        """
        plotmask = np.where(self.mask, 1, np.nan)
        plotname = add_prefix(basename, f"{caller}_mask")
        parm_dict["z_lim"] = [0, 1]
        parm_dict["unit"] = spc
        self._plot_map(plotname, plotmask, "Mask", "Mask", cmap, parm_dict)

    def plot_amplitude(self, basename, caller, cmap, parm_dict):
        """
        Plot Amplitude map
        Args:
            basename: basename for the plot, the prefix 'ancillary_amplitude' will be added to it
            caller: Which mds called this plotting function
            parm_dict: dictionary with plotting parameters
            cmap: colormap name
        """
        if (
            parm_dict["amplitude_limits"] is None
            or parm_dict["amplitude_limits"] == "None"
        ):
            parm_dict["z_lim"] = np.nanmin(self.amplitude), np.nanmax(self.amplitude)
        else:
            parm_dict["z_lim"] = parm_dict["amplitude_limits"]

        amp_stats = data_statistics(np.where(self.mask, self.amplitude, np.nan))
        noise_stats = data_statistics(self.amplitude_noise)
        title = (
            "Amplitude, "
            + statistics_to_text(amp_stats)
            + lnbr
            + "Noise, "
            + statistics_to_text(noise_stats)
        )

        plotname = add_prefix(basename, f"{caller}_amplitude")
        parm_dict["unit"] = self.amp_unit
        self._plot_map(plotname, self.amplitude, title, "Amplitude", cmap, parm_dict)

    def plot_phase(self, basename, caller, cmap, parm_dict):
        """
        Plot phase map(s)
        Args:
            basename: basename for the plot(s), the prefix 'phase_{original|corrections|residuals}' will be added to
                      it/them
            caller: Which mds called this plotting function
            parm_dict: dictionary with plotting parameters
            cmap: colormap name
        """

        if parm_dict["phase_unit"] is None:
            parm_dict["unit"] = "deg"
        else:
            parm_dict["unit"] = parm_dict["phase_unit"]
        parm_dict["z_lim"] = parm_dict["phase_limits"]
        fac = convert_unit("rad", parm_dict["unit"], "trigonometric")
        prefix = "phase"
        if caller == "image":
            prefix = "corrected"
            maps = [self.phase]
            labels = ["phase"]
            z_labels = ["Phase"]
        else:
            if self.residuals is None:
                maps = [self.phase]
                labels = ["original"]
                z_labels = ["Phase"]
            else:
                maps = [self.phase, self.phase_corrections, self.phase_residuals]
                labels = ["original", "correction", "residual"]
                z_labels = ["Phase", "Phase correction", "Phase residual"]
        self._multi_plot(
            maps, labels, z_labels, cmap, prefix, basename, fac, parm_dict, caller
        )

    def plot_deviation(self, basename, caller, cmap, parm_dict):
        """
        Plot deviation map(s)
        Args:
            basename: basename for the plot(s), the prefix 'deviation_{original|corrections|residuals}' will be added
                      to it/them
            caller: Which mds called this plotting function
            parm_dict: dictionary with plotting parameters
            cmap: colormap name
        """
        if parm_dict["deviation_unit"] is None:
            parm_dict["unit"] = "mm"
        else:
            parm_dict["unit"] = parm_dict["deviation_unit"]
        parm_dict["z_lim"] = parm_dict["deviation_limits"]
        fac = convert_unit("m", parm_dict["unit"], "length")
        prefix = "deviation"
        rms = self.get_rms(unit=parm_dict["unit"])
        if caller == "image":
            prefix = "original"
            maps = [self.deviation]
            labels = ["deviation"]
            z_labels = ["Deviation"]
        else:
            if self.residuals is None:
                maps = [self.deviation]
                labels = [f"original RMS={rms:.2f} {parm_dict['unit']}"]
                z_labels = ["Deviation"]
            else:
                maps = [self.deviation, self.corrections, self.residuals]
                labels = [
                    f"original RMS={rms[0]:.2f} {parm_dict['unit']}",
                    "correction",
                    f"residual RMS={rms[1]:.2f} {parm_dict['unit']}",
                ]
                z_labels = ["Deviation", "Deviation correction", "Deviation residual"]
        self._multi_plot(
            maps, labels, z_labels, cmap, prefix, basename, fac, parm_dict, caller
        )

    def _multi_plot(
        self, maps, labels, z_labels, cmap, prefix, basename, factor, parm_dict, caller
    ):
        if len(maps) != len(labels):
            raise ValueError("Map list and label list must be of the same size")
        nplots = len(maps)
        if parm_dict["z_lim"] is None or parm_dict["z_lim"] == "None":
            vmax = float(
                np.nanmax(np.abs(factor * maps[0]))
            )  # Gotten from the original map (displays the biggest variation)
            parm_dict["z_lim"] = [-vmax, vmax]
        for iplot in range(nplots):
            title = f"{prefix.capitalize()} {labels[iplot]}"
            plotname = add_prefix(basename, labels[iplot].split()[0])
            plotname = add_prefix(plotname, prefix)
            plotname = add_prefix(plotname, caller)
            self._plot_map(
                plotname, factor * maps[iplot], title, z_labels[iplot], cmap, parm_dict
            )

    def _plot_map(
        self, filename, data, title, z_label, cmap, parm_dict, add_colorbar=True
    ):
        fig, ax = create_figure_and_axes(parm_dict["figure_size"], [1, 1])
        simple_imshow_map_plot(
            ax,
            fig,
            self.u_axis,
            self.v_axis,
            np.where(self.mask, data, np.nan),
            title,
            cmap,
            parm_dict["z_lim"],
            z_label=f"{z_label} [{parm_dict['unit']}]",
            add_colorbar=add_colorbar,
        )

        self._add_resolution_to_plot(ax)
        ax.set_xlabel("X axis [m]")
        ax.set_ylabel("Y axis [m]")
        for panel in self.panels:
            panel.plot(
                ax, screws=parm_dict["plot_screws"], label=parm_dict["panel_labels"]
            )

        suptitle = create_informative_label_from_summary(self.summary, "deg")
        suptitle += f", Pol. state: {self.pol_state}"
        close_figure(fig, suptitle, filename, parm_dict["dpi"], parm_dict["display"])

    def _add_resolution_to_plot(self, ax, xpos=0.9, ypos=0.1):
        lw = 0.5
        if self.resolution is None:
            return
        minx = self.u_axis[0]
        miny = self.v_axis[1]
        dx = self.u_axis[-1] - minx
        dy = self.v_axis[-1] - miny
        center = (minx + xpos * dx, miny + ypos * dy)
        resolution = patches.Ellipse(
            center,
            self.resolution[0],
            self.resolution[1],
            angle=0.0,
            linewidth=lw,
            color="black",
            zorder=2,
            fill=False,
        )
        ax.add_patch(resolution)
        halfbeam = self.resolution / dy / 2
        ax.axvline(
            x=center[0],
            ymin=ypos - halfbeam[1],
            ymax=ypos + halfbeam[1],
            color="black",
            lw=lw / 2,
        )
        ax.axhline(
            y=center[1],
            xmin=xpos - halfbeam[0],
            xmax=xpos + halfbeam[0],
            color="black",
            lw=lw / 2,
        )

    def plot_screw_adjustments(self, filename, parm_dict):
        """
        Plot screw adjustments as circles over a blank canvas with the panel layout
        Args:
            filename: Name of the output filename for the plot
            parm_dict: Dictionary with plotting parameters
        """
        unit = parm_dict["unit"]
        threshold = parm_dict["threshold"]
        combined_threshold = parm_dict["combined_threshold"]
        cmap = get_proper_color_map(parm_dict["colormap"], default_cmap="RdBu_r")
        fig, ax = create_figure_and_axes(parm_dict["figure_size"], [1, 1])

        fac = convert_unit("m", unit, "length")
        vmax = float(np.nanmax(np.abs(fac * self.screw_adjustments)))
        vmin = -vmax
        if threshold is None or threshold == "None":
            threshold = 0.1 * vmax
        else:
            threshold = np.abs(threshold)
        if combined_threshold is None or combined_threshold == "None":
            combined_threshold = 0.3 * vmax
        else:
            combined_threshold = np.abs(combined_threshold)

        ax.set_title(f"{lnbr}Threshold = {threshold:.2f} {unit}", fontsize="small")
        # set the limits of the plot to the limits of the data
        extent = compute_extent(self.u_axis, self.v_axis)
        im = ax.imshow(
            np.full_like(self.deviation, fill_value=np.nan),
            cmap=cmap,
            interpolation="nearest",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )

        self._add_resolution_to_plot(ax)
        colorbar = well_positioned_colorbar(
            ax, fig, im, "Screw adjustments [" + unit + "]"
        )
        if threshold > 0:
            line = threshold

            while line < vmax:
                colorbar.ax.axhline(y=line, color="black", linestyle="-", lw=0.2)
                colorbar.ax.axhline(y=-line, color="black", linestyle="-", lw=0.2)
                line += threshold

        ax.set_xlabel("X axis [m]")
        ax.set_ylabel("Y axis [m]")

        filter_shadowed_panels = (
            parm_dict["exclude_shadowed_panels"]
            and self.telescope.list_of_shadowed_panels is not None
        )

        for ipanel in range(len(self.panels)):
            panel = self.panels[ipanel]
            panel.plot(ax, screws=False, label=parm_dict["panel_labels"])
            if filter_shadowed_panels:
                if panel.label in self.telescope.list_of_shadowed_panels:
                    continue
            panel.plot_screw_corrections(
                ax, cmap, unit, threshold, combined_threshold, vmin, vmax
            )

        superior_title = create_informative_label_from_summary(self.summary, "deg")
        superior_title += f", Pol. state: {self.pol_state}"
        close_figure(
            fig, superior_title, filename, parm_dict["dpi"], parm_dict["display"]
        )

    def _build_panel_data_arrays(self):
        """
        Build arrays with data from the panels so that they can be stored on the XDS
        Returns:
            List with panel labels, panel fitting parameters, screw_adjustments
        """
        npanels = len(self.panels)
        # First panel might fail hence we need to check npar for all panels
        max_n_par = 0
        max_n_screws = 0
        for panel in self.panels:
            max_n_par = np.max((panel.model.npar, max_n_par))
            max_n_screws = np.max((panel.screw_positions.shape[0], max_n_screws))

        self.panel_labels = np.ndarray([npanels], dtype="U22")
        self.panel_model_array = np.ndarray([npanels], dtype="U22")
        self.panel_pars = np.full((npanels, max_n_par), np.nan, dtype=float)
        self.screw_adjustments = np.ndarray((npanels, max_n_screws), dtype=float)
        self.panel_fallback = np.ndarray([npanels], dtype=bool)

        for i_panel, panel in enumerate(self.panels):
            self.panel_labels[i_panel] = panel.label
            self.panel_pars[i_panel, :] = panel.model.parameters
            self.screw_adjustments[i_panel, :] = panel.export_screws(unit="m")
            self.panel_model_array[i_panel] = panel.model_name
            self.panel_fallback[i_panel] = panel.fall_back_fit

    def export_screws(
        self,
        filename,
        unit="mm",
        comment_char="#",
        filter_shadowed_panels=False,
        threshold=None,
        combined_threshold=None,
        apply_thresholds=False,
    ):
        """
        Export screw adjustments for all panels onto an ASCII file
        Args:
            filename: ASCII file name/path
            unit: unit for panel screw adjustments ['mm','miliinches']
            comment_char: Character used for comments
            filter_shadowed_panels: Filter panels in the shadows of feed arms from appearing in the list
            threshold: Threshold for individual screw adjustments to be included in the list
            combined_threshold: threshold for sum(abs(screw_adjustments)) above which to include all screws in a panel
            apply_thresholds: Apply thresholds to produce a filtered list
        """
        date_str = create_informative_label_from_summary(self.summary, "deg").split(
            ","
        )[-1]
        outfile = f"{comment_char} Screw adjustments for {self.telescope.name}'s {self.label}, pol. state {self.pol_state}{lnbr}"
        outfile += f"{comment_char} Observation date:{date_str}{lnbr}"
        freq = clight / self.wavelength
        rmses = self.get_rms(unit)
        outfile += f"{comment_char} Frequency = {format_frequency(freq)}{lnbr}"
        if unit == "mm":
            outfile += f"{comment_char} Antenna surface RMS before adjustment: {format_value_unit(rmses[0], unit)}{lnbr}"
            outfile += f"{comment_char} Antenna surface RMS after adjustment: {format_value_unit(rmses[1], unit)}{lnbr}"
        else:
            mmrms = self.get_rms("mm")
            outfile += (
                f"{comment_char} Antenna surface RMS before adjustment: {format_value_unit(rmses[0], unit)} or "
                f"{format_value_unit(mmrms[0], 'mm')}{lnbr}"
            )
            outfile += (
                f"{comment_char} Antenna surface RMS after adjustment: {format_value_unit(rmses[1], unit)} or "
                f"{format_value_unit(mmrms[1], 'mm')}{lnbr}"
            )
        outfile += f"{comment_char} Lower means away from subreflector" + lnbr
        outfile += f"{comment_char} Raise means toward the subreflector" + lnbr
        outfile += f"{comment_char} LOWER the panel if the number is POSITIVE" + lnbr
        outfile += f"{comment_char} RAISE the panel if the number is NEGATIVE" + lnbr
        outfile += f"{comment_char} Adjustments are in " + unit + lnbr
        outfile += lnbr
        outfile += f"{comment_char} Panel{2 * spc}"
        nscrews = len(self.telescope.screw_description)
        for screw in self.telescope.screw_description:
            outfile += f"{4 * spc}{screw:2s}{4 * spc}"
        outfile += f"Fallback{4 * spc}Model{lnbr}"

        fac = convert_unit("m", unit, "length")

        if apply_thresholds:
            screw_max = float(np.nanmax(np.abs(fac * self.screw_adjustments)))
            if threshold is None:
                threshold = 0.1 * screw_max
            if combined_threshold is None:
                combined_threshold = 0.3 * screw_max
        else:
            threshold = 0
            combined_threshold = 0

        for i_panel in range(len(self.panel_labels)):
            this_panel_abs_screw = np.abs(fac * self.screw_adjustments[i_panel, :])
            # HERE COMES THE CHECK FOR BLACKLISTED PANELS IF BLACKLISTED AND FILTER ACTIVE CONTINUE
            this_panel_line = ""
            if np.sum(this_panel_abs_screw) > combined_threshold:
                this_panel_line += f"{self.panel_labels[i_panel]:>5s}"
                for i_screw in range(nscrews):
                    this_panel_line += (
                        f" {fac * self.screw_adjustments[i_panel, i_screw]:>9.2f}"
                    )
            else:
                if np.any(this_panel_abs_screw > threshold):
                    this_panel_line += f"{self.panel_labels[i_panel]:>5s}"
                    for i_screw in range(nscrews):
                        if this_panel_abs_screw[i_screw] > threshold:
                            this_panel_line += f" {fac * self.screw_adjustments[i_panel, i_screw]:>9.2f}"
                        else:
                            this_panel_line += 10 * " "

            if this_panel_line != "":
                this_panel_line += (
                    f"{5 * spc}{bool_to_str(self.panel_fallback[i_panel]):>3s}{7 * spc}{self.panel_model_array[i_panel]}"
                    + lnbr
                )
            outfile += this_panel_line

        if filter_shadowed_panels:
            outfile += (
                f"{lnbr}{comment_char} Applied filtering of shadowed panels{lnbr}"
            )
        if combined_threshold > 0:
            outfile += f"{comment_char} Applied a combined threshold of {combined_threshold:.3f} {unit}{lnbr}"
        if threshold > 0:
            outfile += (
                f"{comment_char} Applied a threshold of {threshold:.3f} {unit}{lnbr}"
            )
        string_to_ascii_file(outfile, filename)

    def export_xds(self):
        """
        Export all the data to Xarray dataset
        Returns:
            XarrayDataSet containing all the relevant information
        """
        xds = xr.Dataset()
        gains = self.gains()
        rms = self.get_rms(unit="m")
        xds.attrs["ddi"] = self.ddi
        xds.attrs["wavelength"] = self.wavelength
        xds.attrs["amp_unit"] = self.amp_unit
        xds.attrs["panel_model"] = self.panel_model
        xds.attrs["panel_margin"] = self.panel_margins
        xds.attrs["clip"] = self.clip
        xds.attrs["solved"] = self.solved
        xds.attrs["fitted"] = self.fitted
        xds.attrs["aperture_resolution"] = self.resolution
        xds.attrs["pol_state"] = self.pol_state
        xds.attrs["summary"] = self.summary

        xds["AMPLITUDE"] = xr.DataArray(self.amplitude, dims=["u", "v"])
        xds["PHASE"] = xr.DataArray(self.phase, dims=["u", "v"])
        xds["DEVIATION"] = xr.DataArray(self.deviation, dims=["u", "v"])
        xds["MASK"] = xr.DataArray(self.mask, dims=["u", "v"])
        xds["PANEL_DISTRIBUTION"] = xr.DataArray(
            self.panel_distribution, dims=["u", "v"]
        )
        xds["AMP_NOISE"] = xr.DataArray(self.amplitude_noise, dims=["u", "v"])
        xds["RADIUS"] = xr.DataArray(self.rad, dims=["u", "v"])

        coords = {"u": self.u_axis, "v": self.v_axis}

        if self.solved:
            xds["PHASE_RESIDUALS"] = xr.DataArray(self.phase_residuals, dims=["u", "v"])
            xds["RESIDUALS"] = xr.DataArray(self.residuals, dims=["u", "v"])
            xds["PHASE_CORRECTIONS"] = xr.DataArray(
                self.phase_corrections, dims=["u", "v"]
            )
            xds["CORRECTIONS"] = xr.DataArray(self.corrections, dims=["u", "v"])
            xds.attrs["input_rms"] = rms[0]
            xds.attrs["output_rms"] = rms[1]
            xds.attrs["input_gain"] = gains[0][0]
            xds.attrs["output_gain"] = gains[1][0]
            xds.attrs["theoretical_gain"] = gains[0][1]
            xds["PANEL_PARAMETERS"] = xr.DataArray(
                self.panel_pars, dims=["labels", "pars"]
            )
            xds["PANEL_SCREWS"] = xr.DataArray(
                self.screw_adjustments, dims=["labels", "screws"]
            )
            xds["PANEL_MODEL"] = xr.DataArray(self.panel_model_array, dims=["labels"])
            xds["PANEL_FALLBACK"] = xr.DataArray(self.panel_fallback, dims=["labels"])

            coords = {
                **coords,
                "labels": self.panel_labels,
                "screws": self.telescope.screw_description,
                "pars": np.arange(self.panel_pars.shape[1]),
            }

        else:
            xds.attrs["input_rms"] = rms
            xds.attrs["input_gain"] = gains[0]
            xds.attrs["theoretical_gain"] = gains[1]

        xds = xds.assign_coords(coords)
        return xds

    def export_to_fits(self, basename):
        """
        Data to export: Amplitude, mask, phase, phase_corrections, phase_residuals, deviations, deviation_corrections,
                        deviation_residuals
        conveniently all data are on the same grid!
        Returns:
        """

        head = {
            "PMODEL": self.panel_model,
            "PMARGIN": self.panel_margins,
            "CLIP": self.clip,
            "TELESCOP": self.antenna_name,
            "INSTRUME": self.telescope.name,
            "WAVELENG": self.wavelength,
            "FREQUENC": clight / self.wavelength,
        }
        head = put_axis_in_fits_header(head, self.u_axis, 1, "X----LIN", "m")
        head = put_axis_in_fits_header(head, self.v_axis, 2, "Y----LIN", "m")
        head = put_resolution_in_fits_header(head, self.resolution)

        write_fits(
            head,
            "Amplitude",
            self.amplitude,
            add_prefix(basename, "amplitude") + ".fits",
            self.amp_unit,
            "panel",
        )
        write_fits(
            head,
            "Mask",
            np.where(self.mask, 1.0, np.nan),
            add_prefix(basename, "mask") + ".fits",
            "",
            "panel",
        )
        write_fits(
            head,
            "Original Phase",
            self.phase,
            add_prefix(basename, "phase_original") + ".fits",
            "rad",
            "panel",
        )
        write_fits(
            head,
            "Phase Corrections",
            self.phase_corrections,
            add_prefix(basename, "phase_correction") + ".fits",
            "rad",
            "panel",
        )
        write_fits(
            head,
            "Phase residuals",
            self.phase_residuals,
            add_prefix(basename, "phase_residual") + ".fits",
            "rad",
            "panel",
        )
        write_fits(
            head,
            "Original Deviation",
            self.deviation,
            add_prefix(basename, "deviation_original") + ".fits",
            "m",
            "panel",
        )
        write_fits(
            head,
            "Deviation Corrections",
            self.corrections,
            add_prefix(basename, "deviation_correction") + ".fits",
            "m",
            "panel",
        )
        write_fits(
            head,
            "Deviation residuals",
            self.residuals,
            add_prefix(basename, "deviation_residual") + ".fits",
            "m",
            "panel",
        )
