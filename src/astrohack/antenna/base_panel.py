import numpy as np
import toolviper.utils.logger as logger
from matplotlib.colors import Normalize
from matplotlib.patches import Circle

from astrohack.antenna.panel_fitting import PANEL_MODEL_DICT, PanelModel, PanelPoint
from astrohack.utils.constants import fontsize, markersize, pi
from astrohack.utils.conversion import convert_unit
from astrohack.utils.text import bool_to_str, lnbr, spc


class BasePanel:
    markers = ["X", "o", "*", "P", "D"]
    colors = ["g", "g", "r", "r", "b"]
    linewidth = 0.5
    linecolor = "black"

    def __init__(
        self,
        model,
        screw_positions,
        plot_screw_pos,
        plot_screw_size,
        label,
        center=None,
        zeta=None,
        ref_points=None,
    ):
        """
        Initializes the base panel with the appropriated fitting methods and providing basic functionality
        Fitting method models are:
        AIPS fitting models:
            mean: The panel is corrected by the mean of its samples
            rigid: The panel samples are fitted to a rigid surface
        Corotated Paraboloids (the two bending axes are parallel and perpendicular to the radius of the antenna crossing
        the middle of the panel):
            corotated_scipy: Paraboloid is fitted using scipy.optimize, robust but slow
            corotated_lst_sq: Paraboloid is fitted using the linear algebra least squares method, fast but unreliable
            corotated_robust: Tries corotated_lst_sq, if it diverges falls back to corotated_scipy
        Experimental fitting models:
            xy_paraboloid: fitted using scipy.optimize, bending axes are parallel to the x and y axes
            rotated_paraboloid: fitted using scipy.optimize, bending axes can be rotated by an arbitrary angle
            full_paraboloid_lst_sq: Full 9 parameter paraboloid fitted using least_squares method, heavily overfits
        Args:
            model: What model of surface fitting method to be used
            label: Panel label
            screw_positions: position of the screws
            center: Panel center
            zeta: panel center angle
            ref_points: panel ref_points (relevant for flexible model)
        """
        self.model_name = model
        self.solved = False
        self.fall_back_fit = False
        self.label = label
        self.screw_positions = screw_positions
        self.plot_screw_pos = plot_screw_pos
        self.plot_screw_size = plot_screw_size
        self.samples = []
        self.margins = []
        self.corrections = None
        self.screw_corrections = None

        if center is None:
            self.center = PanelPoint(0, 0)
        else:
            self.center = center
        if zeta is None:
            self.zeta = 0
        else:
            self.zeta = zeta
        if ref_points is None:
            self.ref_points = [0, 0, 0]
        else:
            self.ref_points = ref_points

        try:
            model_dict = PANEL_MODEL_DICT[self.model_name]
        except KeyError:
            msg = f"Unknown model {self.model_name}"
            logger.error(msg)
            raise ValueError(msg)

        self.model = PanelModel(model_dict, self.zeta, self.ref_points, self.center)

    def add_sample(self, sample):
        """
        Add a point to the panel's list of points to be fitted
        Args:
            sample: tuple/list containing point description [xcoor,ycoor,xidx,yidx,value]
        """
        if self.model_name in PANEL_MODEL_DICT.keys():
            self.samples.append(PanelPoint(*sample))
        else:
            self.samples.append(sample)

    def add_margin(self, sample):
        """
        Add a point to the panel's list of points to be corrected, but not fitted
        Args:
            sample: tuple/list containing point description [xcoor,ycoor,xidx,yidx,value]
        """
        if self.model_name in PANEL_MODEL_DICT.keys():
            self.margins.append(PanelPoint(*sample))
        else:
            self.margins.append(sample)

    def solve(self):
        if len(self.samples) < self.model.npar:
            self._fallback_solve()
            status = False
        else:
            try:
                self.model.solve(self.samples)
                status = True
            except np.linalg.LinAlgError:
                self._fallback_solve()
                status = False
        self.solved = True
        return status

    def _fallback_solve(self):
        """
        Changes the method association to mean surface fitting, and fits the panel with it
        """
        self.model = PanelModel(
            PANEL_MODEL_DICT["mean"], self.zeta, self.ref_points, self.center
        )
        self.model.solve(self.samples)
        self.fall_back_fit = True

    def get_corrections(self):
        if not self.solved:
            msg = "Cannot correct a panel that is not solved"
            logger.error(msg)
            raise RuntimeError(msg)
        self.corrections = self.model.correct(self.samples, self.margins)
        nscrew = len(self.screw_positions)
        self.screw_corrections = np.zeros(nscrew)
        for iscrew, screw in enumerate(self.screw_positions):
            self.screw_corrections[iscrew] = self.model.correct_point(screw)
        return self.corrections

    def plot_label(self, ax, rotate=True):
        """
        Plots panel label to ax
        Args:
            ax: matplotlib axes instance
            rotate: Rotate label for better display
        """
        if rotate:
            angle = (-self.zeta % pi - pi / 2) * convert_unit(
                "rad", "deg", "trigonometric"
            )
        else:
            angle = 0
        ax.text(
            self.center.xc,
            self.center.yc,
            self.label,
            fontsize=fontsize,
            ha="center",
            va="center",
            rotation=angle,
        )

    def plot_screws(self, ax):
        """
        Plots panel screws to ax
        Args:
            ax: matplotlib axes instance
        """
        for iscrew, screw in enumerate(self.screw_positions):
            ax.scatter(
                screw.xc,
                screw.yc,
                marker=self.markers[iscrew],
                lw=self.linewidth,
                s=markersize,
                color=self.colors[iscrew],
            )

    def _any_screw_above_threshold(self, unit_factor, threshold):
        return np.any(unit_factor * np.abs(self.screw_corrections) > threshold)

    def _above_combined_threshold(self, unit_factor, combined_threshold):
        return np.sum(unit_factor * np.abs(self.screw_corrections)) > combined_threshold

    def plot_screw_corrections(
        self, ax, cmap, unit, threshold, combined_threshold, vmin, vmax
    ):
        """
        Plot screw corrections onto an axis
        Args:
            ax: axis for plot
            cmap: Colormap of the corrections to be applied to each screw
            unit: unit for corrections
            threshold: Threshold below which data is considered negligible
            combined_threshold: Same as threshold but for the sum of all corrections in this panel
            vmin: bottom of the colormap
            vmax: top of the colormap
        """

        def plot_a_screw(screw_pos, screw_corr):
            circle = Circle(
                (screw_pos.xc, screw_pos.yc),
                self.plot_screw_size,
                color=cmap(norm(screw_corr)),
                fill=True,
            )
            ax.add_artist(circle)

        fac = convert_unit("m", unit, "length")
        norm = Normalize(vmin=vmin, vmax=vmax)
        if self._any_screw_above_threshold(fac, threshold):
            for i_screw in range(self.plot_screw_pos.shape[0]):
                screw_correction = fac * self.screw_corrections[i_screw]
                if screw_correction > threshold:
                    plot_a_screw(self.plot_screw_pos[i_screw], screw_correction)

        elif self._above_combined_threshold(fac, combined_threshold):
            for i_screw in range(self.plot_screw_pos.shape[0]):
                plot_a_screw(
                    self.plot_screw_pos[i_screw], fac * self.screw_corrections[i_screw]
                )
        else:
            # Nothing to plot!
            pass

    def reread_solution(self, model_name, parameters, fallback, screw_adjustments):
        """
        Simple re-read of values pertinent for export production
        Args:
            model_name: Panel surface model
            parameters: model parameters
            fallback: is the panel model a fallback?
            screw_adjustments: screw adjustments
        """
        self.model.parameters = parameters
        self.screw_corrections = screw_adjustments
        self.model_name = model_name
        self.fall_back_fit = fallback

    def get_panel_adjustment_string(self, unit_factor, threshold, combined_threshold):
        """
        Get a string of panel adjustments
        Args:
            unit_factor: unit conversion factor
            threshold: threshold for single corrections
            combined_threshold: Threshold for cor the combined corrections in a panel
        Returns:
            Empty string if nothing to report, string with panel adjustments if successful
        """
        out_str = f"{self.label:>5s}"
        if self._above_combined_threshold(unit_factor, combined_threshold):
            for screw_correction in self.screw_corrections:
                out_str += f" {unit_factor * screw_correction:>9.2f}"
        elif self._any_screw_above_threshold(unit_factor, threshold):
            for screw_correction in self.screw_corrections:
                if unit_factor * np.abs(screw_correction) > threshold:
                    out_str += f" {unit_factor * screw_correction:>9.2f}"
                else:
                    out_str += 10 * " "
        else:
            return ""

        out_str += f"{5 * spc}{bool_to_str(self.fall_back_fit):>3s}{7 * spc}{self.model_name}{lnbr}"
        return out_str
