import numpy as np

from astrohack.core import beamcalc


def _antenna(band):
    geometry = beamcalc.EVLA_GEOMETRY[band]
    return beamcalc.new_antenna(geometry, geometry["reffreq"])


class TestBeamcalc:
    def test_band_for_frequency(self):
        assert beamcalc.band_for_frequency(1.5) == "EVLA_L"
        assert beamcalc.band_for_frequency(2.0) == "EVLA_L"
        assert beamcalc.band_for_frequency(15.0) == "EVLA_U"

    def test_dish_value_reproduces_surface_samples(self):
        a = _antenna("EVLA_L")
        r = np.arange(1, a["ngeom"] - 1) * a["deltar"]
        z, m = beamcalc.dish_value(a, r)
        assert np.abs(z - a["z"][1:-1]).max() < 1e-6
        assert np.abs(m - a["m"][1:-1]).max() < 1e-12

    def test_rays_are_focused(self):
        r, phi = np.meshgrid(
            np.linspace(2.0, 12.4, 20), np.linspace(0, 2 * np.pi, 36, endpoint=False)
        )
        for band in beamcalc.EVLA_GEOMETRY:
            ray = beamcalc.trace(_antenna(band), r * np.cos(phi), r * np.sin(phi))
            assert np.all(ray["valid"])
            assert np.ptp(beamcalc.ray_length(ray)) < 1e-9
            assert (
                np.abs(beamcalc._norm(ray["aper"] - ray["dish"])[..., :2]).max() < 1e-9
            )

    def test_feed_gain_at_subangle_is_the_taper(self):
        for band, geometry in beamcalc.EVLA_GEOMETRY.items():
            a = _antenna(band)
            side = beamcalc._norm(np.cross(a["feeddir"], [0.0, 0.0, 1.0]))
            theta = np.radians(geometry["subangle"])
            ray = {
                "feed": a["feed"][None],
                "sub": (
                    a["feed"] + np.cos(theta) * a["feeddir"] + np.sin(theta) * side
                )[None],
            }
            assert (
                abs(10 * np.log10(beamcalc.feed_gain(a, ray)[0]) + a["taper"])
                < 1e-3 * a["taper"]
            )

    def test_reflections_keep_the_field_strength(self):
        a = _antenna("EVLA_C")
        r, phi = np.meshgrid(
            np.linspace(2.0, 12.4, 10), np.linspace(0, 2 * np.pi, 12, endpoint=False)
        )
        ray = beamcalc.trace(a, r * np.cos(phi), r * np.sin(phi))
        hhat, vhat, _ = beamcalc.feed_basis(a)
        e1 = beamcalc.trace_pol((hhat + 1j * vhat) / np.sqrt(2), ray)
        assert np.allclose(np.linalg.norm(e1, axis=-1), 1.0)

    def test_fixes(self):
        geometry = beamcalc.EVLA_GEOMETRY["EVLA_C"]
        a = _antenna("EVLA_C")
        foot_z = np.interp(
            geometry["legfoot"], np.arange(a["ngeom"]) * a["deltar"], a["z"]
        )
        assert abs(a["legfootz"] - foot_z) < 1e-6
        r, phi = np.meshgrid(
            np.linspace(2.0, 12.4, 10), np.linspace(0, 2 * np.pi, 12, endpoint=False)
        )
        ray = beamcalc.trace(a, r * np.cos(phi), r * np.sin(phi))
        hhat, vhat, _ = beamcalc.feed_basis(a)
        e0 = beamcalc.feed_field(a, (hhat + 1j * vhat) / np.sqrt(2), ray)
        k = beamcalc._norm(ray["sub"] - ray["feed"])
        assert np.abs(np.sum(k * e0, axis=-1)).max() < 1e-12
        assert np.allclose(np.linalg.norm(e0, axis=-1), 1.0)
        assert np.abs(beamcalc.trace_pol(e0, ray)[..., 2]).max() < 1e-8

    def test_width_and_squint_against_casa(self):
        # CASA 6.7.6 A-projection at 1.040 GHz: HWHM 20.64 arcmin GHz, RR-LL separation 2.35 arcmin GHz
        freq, dx, n = 1.04, 0.1, 253
        x0 = -(n // 2) * dx
        ap = beamcalc.calculate_aperture(
            beamcalc.EVLA_GEOMETRY["EVLA_L"],
            freq,
            x0,
            x0,
            dx,
            dx,
            n,
            n,
            fix_leg_foot=False,
            fix_feed_field=False,
        )
        grid = np.arange(-30.0, 30.001, 0.25)
        e = np.exp(
            2j
            * np.pi
            * freq
            * 1e9
            / 299792458.0
            * np.outer(np.radians(grid / freq / 60.0), x0 + np.arange(n) * dx)
        )
        p_rr, p_ll = (np.abs(e @ ap[k] @ e.T) ** 2 for k in (0, 3))
        p_i = 0.5 * (p_rr + p_ll)
        p_i /= p_i.max()
        r = np.hypot(*np.meshgrid(grid, grid, indexing="ij")).ravel()
        edges = np.arange(0.0, 30.0, 0.5)
        idx = np.digitize(r, edges)
        prof = np.array([p_i.ravel()[idx == k].mean() for k in range(1, edges.size)])
        k = int(np.argmax(prof < 0.5))
        centres = 0.5 * (edges[1:] + edges[:-1])
        hwhm = np.interp(0.5, [prof[k], prof[k - 1]], [centres[k], centres[k - 1]])
        centroids = []
        for p in (p_rr, p_ll):
            w = p > 0.5 * p.max()
            ii, jj = np.nonzero(w)
            centroids.append(
                (np.average(ii, weights=p[w]), np.average(jj, weights=p[w]))
            )
        squint = np.hypot(*np.subtract(*centroids)) * 0.25
        assert abs(hwhm / 20.64 - 1) < 0.005
        assert abs(squint - 2.35) < 0.1
