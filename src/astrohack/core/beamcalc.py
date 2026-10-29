"""Aperture illumination of the EVLA antennas by ray tracing: a port of CASA's BeamCalc.

Ported from casa6 casatools/src/code/synthesis/TransformMachines/BeamCalc.cc and BeamCalcParams.cc (Copyright (C)
Associated Universities, Inc., GNU Library General Public License, version 2 or later). It is the ray tracing of CASA's
VLA A-projection, as CASA runs it for the EVLA: the band geometries compiled into BeamCalcParams.cc, with no pathology
and no astigmatism (both are zero there). Lengths are in metres, frequencies in GHz.

The main reflector is axisymmetric, given by ``VLA.surface`` (radius, height and slope; casa-data nrao/VLA, commit
1b0506ed). The subreflector is not given. For each dish point, its point is the one from which a ray from the feed
reflects to the dish point and then along the axis, with the same path from the feed to the aperture plane for every
ray (``sub_from_dish``).

The aperture field of a ray is the feed's field, reflected at the subreflector and the dish (E' = 2 (n.E) n - E). The
feed's field is one polarization vector in the feed's basis, with a Gaussian power pattern of ``taper`` dB at the
angle ``subangle``. The amplitude is sqrt(feed gain x solid angle at the feed per aperture area), and the phase is
2 pi (L - L0) / lambda. Rays inside the central hole or blocked by the quadrupod legs are dropped.

Two BeamCalc defects are fixed by default, and both fixes off reproduce CASA: BeamCalc takes the leg feet's height at
its default leg-foot radius, radius / 2, not the geometry's (``fix_leg_foot``), and uses the same feed field for every
ray, which is not transverse to the ray (``fix_feed_field``).
"""

import numpy as np

from astrohack.utils.package_info import get_astrohack_path

NS_METER = 0.299792458

# BeamCalcParams.cc: EVLABeamCalcGeometryDefaults, EVLABandMinFreqDefaults and EVLABandMaxFreqDefaults
_COMMON = {
    "surface": "VLA",
    "sub_h": 8.47852,
    "subangle": 9.26,
    "legwidth": 0.27,
    "legfoot": 7.55,
    "legapex": 10.93876,
    "rant": 12.5,
}
EVLA_GEOMETRY = {
    "EVLA_L": {
        **_COMMON,
        "freq_range": (0.9, 2.0),
        "feedpos": (-0.10026, 0.97019, 1.67640),
        "rhole": 2.0,
        "reffreq": 1.5,
        "taperpoly": (10.0, 2.0),
    },
    "EVLA_S": {
        **_COMMON,
        "freq_range": (2.0, 4.0),
        "feedpos": (0.196123, -0.955439, 1.67640),
        "rhole": 3.87,
        "reffreq": 3.0,
        "taperpoly": (15.837, -0.004, 0.00023),
    },
    "EVLA_C": {
        **_COMMON,
        "freq_range": (4.0, 8.0),
        "feedpos": (0.94300, -0.249152, 1.67640),
        "rhole": 2.0,
        "reffreq": 6.0,
        "taperpoly": (12.75, 0.375),
    },
    "EVLA_X": {
        **_COMMON,
        "freq_range": (8.0, 12.0),
        "feedpos": (0.893100, 0.392044, 1.67640),
        "rhole": 2.0,
        "reffreq": 10.0,
        "taperpoly": (13.0,),
    },
    "EVLA_U": {
        **_COMMON,
        "freq_range": (12.0, 18.0),
        "feedpos": (-0.657688, -0.720260, 1.67640),
        "rhole": 2.0,
        "reffreq": 15.0,
        "taperpoly": (13.0,),
    },
    "EVLA_K": {
        **_COMMON,
        "freq_range": (18.0, 26.5),
        "feedpos": (-0.877393, -0.426039, 1.67640),
        "rhole": 2.0,
        "reffreq": 22.0,
        "taperpoly": (13.0,),
    },
    "EVLA_A": {
        **_COMMON,
        "freq_range": (26.5, 40.0),
        "feedpos": (-0.933238, 0.283539, 1.67640),
        "rhole": 2.0,
        "reffreq": 33.0,
        "taperpoly": (13.0,),
    },
    "EVLA_Q": {
        **_COMMON,
        "freq_range": (40.0, 50.0),
        "feedpos": (-0.9723533, -0.0765259, 1.67640),
        "rhole": 2.0,
        "reffreq": 45.0,
        "taperpoly": (13.0,),
    },
}


def _norm(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def band_for_frequency(freq):
    """The first band whose frequency range holds freq (GHz), as BeamCalc::getBandID without a band name."""
    for band, geometry in EVLA_GEOMETRY.items():
        if geometry["freq_range"][0] <= freq <= geometry["freq_range"][1]:
            return band
    raise ValueError(f"No EVLA band covers {freq} GHz")


def new_antenna(geometry, freq, fix_leg_foot=True):
    """BeamCalc::newAntenna and newAntennafromApertureCalcParams for one band geometry at freq (GHz)."""
    r, z, m = np.loadtxt(
        get_astrohack_path() / f"data/beamcalc/{geometry['surface']}.surface",
        unpack=True,
    )
    a = {
        "z": z,
        "m": m,
        "radius": r[-1],
        "ngeom": r.size,
        "zedge": z[-1],
        "deltar": r[-1] / (r.size - 1.0),
        "bestparabola": z[-1] / r[-1] ** 2,
    }
    feed = np.array(
        [-geometry["feedpos"][0], geometry["feedpos"][1], geometry["feedpos"][2]]
    )
    sub_h = geometry["sub_h"]
    d = np.sqrt(feed[0] ** 2 + feed[1] ** 2 + (sub_h - feed[2]) ** 2)
    a["K"] = sub_h + d if sub_h - feed[2] > 0 else np.linalg.norm(feed)
    a["feeddir"] = np.array([-feed[0], -feed[1], sub_h - feed[2]]) / d
    taper = sum(
        c * (freq - geometry["reffreq"]) ** i
        for i, c in enumerate(geometry["taperpoly"])
    )
    a["fa2pi"] = (
        2.0 * np.pi * np.sqrt(taper) * 0.1874 / np.sin(np.radians(geometry["subangle"]))
    )
    a.update(
        sub_h=sub_h,
        feed=feed,
        taper=taper,
        freq=freq,
        wavelength=NS_METER / freq,
        legwidth=geometry["legwidth"],
        legfoot=geometry["legfoot"],
        legapex=geometry["legapex"],
        hole_radius=geometry["rhole"],
    )
    # newAntenna sets legfootz at its default leg foot, radius / 2, before the geometry's leg foot replaces it
    a["legfootz"] = dish_value(
        a, np.array(a["legfoot"] if fix_leg_foot else a["radius"] / 2.0)
    )[0]
    return a


def dish_value(a, r):
    """BeamCalc::dishvalue: height and slope of the dish at radius r, quadratic in the slope about the nearest sample."""
    s = np.where(r < 0, -1.0, 1.0)
    r = np.abs(r)
    d = a["deltar"]
    n = np.minimum(np.floor(r / d + 0.5).astype(int), a["ngeom"] - 2)
    x = r - n * d
    first = n == 0
    nm1 = np.maximum(n - 1, 0)
    ma = np.where(first, -a["m"][1], a["m"][nm1])
    mb = np.where(first, 0.0, a["m"][n])
    mc = a["m"][np.where(first, 1, n + 1)]
    zav = np.where(
        first, 2.0 * a["z"][1] + a["z"][0], a["z"][nm1] + a["z"][n] + a["z"][n + 1]
    )
    big_b = 0.5 * (mc - ma) / d
    big_c = 0.5 * (mc - 2.0 * mb + ma) / d**2
    big_d = (zav - big_b * d**2) / 3.0
    z = s * (big_d + mb * x + big_b * x**2 / 2.0 + big_c * x**3 / 3.0)
    m = s * (mb + big_b * x + big_c * x**2)
    return np.where(r == 0, a["z"][0], z), np.where(r == 0, 0.0, m)


def sub_from_dish(a, x, y):
    """BeamCalc::subfromdish: subreflector points [..., 3] and normals [..., 3] for the dish points (x, y)."""
    r = np.hypot(x, y)
    z, m = dish_value(a, r)
    m = np.tan(2.0 * np.arctan(m))
    w = 1.0 / np.sqrt(1.0 + m * m)
    rs = np.where(r == 0, 1.0, r)
    u, v = -m * (x / rs) * w, -m * (y / rs) * w
    dish = np.stack([x, y, z], axis=-1)
    dx = a["feed"] - dish
    dl = a["K"] + z
    t = (
        0.5
        * (np.sum(dx * dx, axis=-1) - dl * dl)
        / (-dl + u * dx[..., 0] + v * dx[..., 1] + w * dx[..., 2])
    )
    sub = dish + t[..., None] * np.stack([u, v, w], axis=-1)
    sub = np.where((r == 0)[..., None], [0.0, 0.0, a["sub_h"]], sub)
    normal = _norm(_norm(dish - sub) + _norm(a["feed"] - sub))
    return sub, normal


def intersect_dish(a, sub, unitdir, niter):
    """BeamCalc::intersectdish: the dish point and normal along unitdir from sub, Newton steps from the best paraboloid."""
    bp = a["bestparabola"]
    big_a = bp * (unitdir[..., 0] ** 2 + unitdir[..., 1] ** 2)
    big_b = (
        2.0 * bp * (unitdir[..., 0] * sub[..., 0] + unitdir[..., 1] * sub[..., 1])
        - unitdir[..., 2]
    )
    big_c = bp * (sub[..., 0] ** 2 + sub[..., 1] ** 2) - sub[..., 2]
    t = 0.5 * (np.sqrt(big_b**2 - 4.0 * big_a * big_c) - big_b) / big_a
    for it in range(niter + 1):
        x = sub[..., :2] + t[..., None] * unitdir[..., :2]
        r = np.hypot(x[..., 0], x[..., 1])
        z, m = dish_value(a, r)
        n2 = 1.0 / np.sqrt(1.0 + m * m)
        point = np.concatenate([x, z[..., None]], axis=-1)
        normal = np.stack(
            [-m * (x[..., 0] / r) * n2, -m * (x[..., 1] / r) * n2, n2], axis=-1
        )
        if it < niter:
            t = np.sum(normal * (point - sub), axis=-1) / np.sum(
                normal * unitdir, axis=-1
            )
    return point, normal


def trace(a, x, y):
    """BeamCalc::trace without pathology: rays from the feed through the dish points (x, y) to the aperture plane.

    Returns the feed, subreflector, dish and aperture points, the two normals and whether each ray is valid.
    """
    sub, sub_normal = sub_from_dish(a, x, y)
    valid = (sub_normal[..., 2] >= -1.0) & (sub_normal[..., 2] <= -0.0)
    to_feed = _norm(a["feed"] - sub)
    to_dish = (
        2.0 * sub_normal * np.sum(sub_normal * to_feed, axis=-1, keepdims=True)
        - to_feed
    )
    dish, dish_normal = intersect_dish(a, sub, to_dish, niter=7)
    to_aperture = to_dish - 2.0 * dish_normal * np.sum(
        dish_normal * to_dish, axis=-1, keepdims=True
    )
    t = (a["zedge"] - dish[..., 2]) / to_aperture[..., 2]
    aperture = dish + t[..., None] * to_aperture
    return {
        "feed": np.broadcast_to(a["feed"], sub.shape),
        "sub": sub,
        "sub_normal": sub_normal,
        "dish": dish,
        "dish_normal": dish_normal,
        "aper": aperture,
        "valid": valid,
    }


def ray_length(ray):
    """BeamCalc::Raylen: feed to subreflector to dish to aperture plane."""
    return sum(
        np.linalg.norm(ray[p] - ray[q], axis=-1)
        for p, q in (("feed", "sub"), ("sub", "dish"), ("dish", "aper"))
    )


def feed_gain(a, ray):
    """BeamCalc::feedgain: Gaussian power pattern about the feed direction."""
    costheta = np.sum(a["feeddir"] * _norm(ray["sub"] - ray["feed"]), axis=-1)
    return np.exp(2.0 * (-0.083) * a["fa2pi"] ** 2 * (1.0 - costheta**2))


def solid_angle(ray1, ray2, ray3):
    """BeamCalc::dOmega: solid angle at the feed of the triangle of three rays' subreflector points."""
    n1, n2, n3 = (_norm(r["sub"] - r["feed"]) for r in (ray1, ray2, ray3))
    return 0.5 * np.linalg.norm(np.cross(n1 - n3, n2 - n3), axis=-1)


def feed_field(a, e0, ray):
    """The feed field e0 [3] turned from the feed direction onto each ray, about their common normal.

    This is the field of a feed with a rotationally symmetric pattern and no cross-polarization (Ludwig's third
    definition). BeamCalc uses e0 for every ray.
    """
    d = a["feeddir"]
    k = _norm(ray["sub"] - ray["feed"])
    return e0 - np.sum(k * e0, axis=-1, keepdims=True) / (
        1.0 + np.sum(k * d, axis=-1, keepdims=True)
    ) * (d + k)


def trace_pol(e0, ray):
    """BeamCalc::tracepol: the feed field e0 [3] reflected at the subreflector and at the dish, per ray."""
    v1, v2, v3 = (
        _norm(ray[q] - ray[p])
        for p, q in (("feed", "sub"), ("sub", "dish"), ("dish", "aper"))
    )
    e1 = np.broadcast_to(e0, v1.shape).astype(complex)
    for r in (_norm(v1 - v2), _norm(v2 - v3)):
        e1 = 2.0 * r * np.sum(r * e1, axis=-1, keepdims=True) - e1
    return e1


def feed_basis(a):
    """BeamCalc::getfeedbasis: hhat, vhat and the feed direction."""
    d = a["feeddir"]
    if d[0] == 0.0 and d[1] == 0.0:
        return np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), d
    vhat = _norm(np.array([d[1], -d[0], 0.0]))
    return np.cross(vhat, d), vhat, d


def leg_block(a, ray, spherical):
    """BeamCalc::legplanewaveblock2 (spherical False) and legsphericalwaveblock (True), "+" or "x" shaped legs.

    A ray is blocked when the line through its dish point and its aperture point (plane wave, dish points inside the
    leg foot circle) or its subreflector point (spherical wave, outside) passes within half the leg width of a leg's line.
    """
    if a["legwidth"] == 0.0:
        return np.zeros(ray["dish"].shape[:-1], dtype=bool)
    thetas = (
        np.array([0.25, 0.75, 1.25, 1.75]) * np.pi
        if a["legwidth"] < 0
        else np.array([0.0, 0.5, 1.0, 1.5]) * np.pi
    )
    r0 = ray["dish"]
    rr = r0[..., 0] ** 2 + r0[..., 1] ** 2
    inside = rr < a["legfoot"] ** 2
    dr = (ray["sub"] if spherical else ray["aper"]) - r0
    blocked = np.zeros(rr.shape, dtype=bool)
    for theta in thetas:
        l0 = np.array(
            [a["legfoot"] * np.cos(theta), a["legfoot"] * np.sin(theta), a["legfootz"]]
        )
        near = (l0[0] * r0[..., 0] + l0[1] * r0[..., 1]) / np.sqrt(
            a["legfoot"] ** 2 * rr
        ) >= 0.7
        dl = np.array([0.0, 0.0, a["legapex"]]) - l0
        n = _norm(np.cross(dr, dl))
        blocked |= near & (
            np.abs(np.sum((r0 - l0) * n, axis=-1)) <= 0.5 * abs(a["legwidth"])
        )
    return blocked & (~inside if spherical else inside)


def calculate_aperture(
    geometry,
    freq,
    x0,
    y0,
    dx,
    dy,
    nx,
    ny,
    pa=0.0,
    oversamp=1,
    fix_leg_foot=True,
    fix_feed_field=True,
):
    """BeamCalc::calculateAperture for circular feeds: the RR, RL, LR and LL aperture planes [4, ny, nx].

    As in BeamCalc, cell (j, i) has the centre (X, Y) = (x0 + i dx, y0 + j dy) and sums oversamp x oversamp points; the
    point (X, Y) is the dish point x = -(X cos(pa + pi/2) - Y sin(pa + pi/2)), y = X sin(pa + pi/2) + Y cos(pa + pi/2).
    fix_leg_foot and fix_feed_field False reproduce CASA.
    """
    a = new_antenna(geometry, freq, fix_leg_foot)
    l0 = ray_length(trace(a, np.array(0.0), np.array(0.00001)))
    hhat, vhat, _ = feed_basis(a)
    phase = np.arctan2(hhat[1], hhat[0])
    rot = np.array([[np.cos(phase), np.sin(phase)], [-np.sin(phase), np.cos(phase)]])
    pol_r, pol_l = (
        rot @ (np.array([1.0, 1j]) / np.sqrt(2.0)),
        rot @ (np.array([1.0, -1j]) / np.sqrt(2.0)),
    )
    e_r, e_l = hhat * pol_r[0] + vhat * pol_r[1], hhat * pol_l[0] + vhat * pol_l[1]

    sdx, sdy = dx / oversamp, dy / oversamp
    big_x = x0 - dx / 2.0 + sdx / 2.0 + np.arange(nx * oversamp) * sdx
    big_y = y0 - dy / 2.0 + sdy / 2.0 + np.arange(ny * oversamp) * sdy
    big_y, big_x = np.meshgrid(big_y, big_x, indexing="ij")
    pac, pas = np.cos(pa + np.pi / 2), np.sin(pa + np.pi / 2)
    x = -(pac * big_x - pas * big_y)
    y = pas * big_x + pac * big_y
    r2 = x * x + y * y
    use = (
        (np.abs(x) <= a["radius"])
        & (np.abs(y) <= a["radius"])
        & (r2 <= a["radius"] ** 2)
        & (r2 >= a["hole_radius"] ** 2)
    )
    x, y = x[use], y[use]

    x1, y1 = x.copy(), y.copy()
    ok = np.ones(x.shape, dtype=bool)
    for _ in range(6):
        ray = trace(a, x1, y1)
        ok &= ray["valid"]
        x1 += x - ray["aper"][..., 0]
        y1 += y - ray["aper"][..., 1]
    ray = trace(a, x1, y1)
    ok &= (
        ray["valid"]
        & ~leg_block(a, ray, spherical=False)
        & ~leg_block(a, ray, spherical=True)
    )
    eps = sdx / 4.0
    ray_y = trace(a, x1, np.where(y < 0, y1 + eps, y1 - eps))
    ray_x = trace(a, np.where(x < 0, x1 + eps, x1 - eps), y1)
    ok &= ray_x["valid"] & ray_y["valid"]
    d1, d2 = (
        ray_x["aper"][..., :2] - ray["aper"][..., :2],
        ray_y["aper"][..., :2] - ray["aper"][..., :2],
    )
    area = 0.5 * np.abs(d1[..., 0] * d2[..., 1] - d2[..., 0] * d1[..., 1])
    amp = np.sqrt(solid_angle(ray_x, ray_y, ray) / area * sdx * sdx * feed_gain(a, ray))
    fp = np.exp(1j * 2.0 * np.pi * (ray_length(ray) - l0) / a["wavelength"]) * amp
    planes = []
    for e0 in (e_r, e_l):
        if fix_feed_field:
            e0 = feed_field(a, e0, ray)
        e1 = trace_pol(e0, ray) * fp[..., None]
        planes += [e1[..., 0] - 1j * e1[..., 1], e1[..., 0] + 1j * e1[..., 1]]
    out = np.zeros((4,) + use.shape, dtype=complex)
    out[:, use] = np.where(ok, np.array(planes), 0.0)
    return out.reshape(4, ny, oversamp, nx, oversamp).sum(axis=(2, 4))
