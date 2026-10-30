"""Independent slope simulator used to validate :mod:`pyTomoAO.slodar`.

It shares no code with the SLODAR response model: von Karman phase screens are drawn by FFT
(with Lane et al. 1992 subharmonics), sampled through each guide star's cone/cylinder with
cubic interpolation, and converted to Shack-Hartmann slopes as the exact mean gradient
(G-tilt) over each subaperture, i.e. the edge-to-edge phase difference averaged with the
trapezoid rule over ``pix + 1`` points per edge.
"""

import numpy as np
from scipy.ndimage import map_coordinates

ARCSEC2RAD = np.pi / (180 * 3600)

#: Maunakea Ridge median 7-layer model, Keck Adaptive Optics Note 503 (Dekany 2007, v1.1),
#: r0 = 0.16 m at 500 nm (zenith). Altitudes [m], fractional Cn2, wind [m/s].
MKR_MEDIAN = {
    "altitude": np.array([0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]),
    "fraction": np.array([0.517, 0.119, 0.063, 0.061, 0.105, 0.081, 0.054]),
    "windSpeed": np.array([6.8, 6.9, 7.1, 7.5, 10.0, 26.9, 18.5]),
    "r0": 0.16,
}


def kapa_valid_map():
    """The 20x20 KAPA valid-subaperture map shipped with pyTomoAO."""
    import yaml

    from pyTomoAO import example_config

    with open(example_config("kapa")) as f:
        cfg = yaml.safe_load(f)
    return np.array(cfg["lgs_wfs_parameters"]["validLLMap"], dtype=bool)


def ring_asterism(n, radius_arcsec, start_deg=0.0):
    """``(n, 2)`` Cartesian directions [rad] of ``n`` stars evenly spaced on a ring."""
    az = np.radians(start_deg + 360.0 * np.arange(n) / n)
    return radius_arcsec * ARCSEC2RAD * np.stack([np.cos(az), np.sin(az)], axis=1)


def ft_phase_screens(n, dx, r0, L0, rng, *, nSub=3):
    """Two independent von Karman phase screens [rad at the r0 wavelength].

    Parameters
    ----------
    n : int
        Screen size in pixels.
    dx : float
        Pixel pitch [m].
    r0, L0 : float
        Fried parameter and outer scale [m].
    rng : numpy.random.Generator
    nSub : int
        Number of subharmonic levels (Lane et al. 1992).

    Returns
    -------
    tuple of numpy.ndarray
        Two ``(n, n)`` screens (real and imaginary parts of one complex draw).
    """
    df = 1.0 / (n * dx)
    fx = np.fft.fftfreq(n, dx)
    fxx, fyy = np.meshgrid(fx, fx)
    psd = 0.023 * r0 ** (-5 / 3) * (fxx**2 + fyy**2 + 1.0 / L0**2) ** (-11 / 6)
    psd[0, 0] = 0.0
    c = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * np.sqrt(psd) * df
    scr = np.fft.ifft2(c) * n * n

    # Subharmonics for the low-order power missed by the FFT grid; exp(2i pi (fx x + fy y))
    # is separable, so each costs one outer product.
    x = (np.arange(n) - n / 2) * dx
    for p in range(1, nSub + 1):
        dfp = df / 3**p
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if i == 0 and j == 0:
                    continue
                fxp, fyp = i * dfp, j * dfp
                psdp = 0.023 * r0 ** (-5 / 3) * (fxp**2 + fyp**2 + 1 / L0**2) ** (-11 / 6)
                cn = (rng.standard_normal() + 1j * rng.standard_normal()) * np.sqrt(psdp) * dfp
                scr += cn * np.outer(np.exp(2j * np.pi * fyp * x), np.exp(2j * np.pi * fxp * x))
    scr -= scr.mean()
    return scr.real, scr.imag


def pupil_points(nLenslet, d, pix):
    """Coordinates [m] of the ``(nLenslet*pix+1)^2`` subaperture-corner grid, centred."""
    m = nLenslet * pix + 1
    x = (np.arange(m) - (m - 1) / 2) * d / pix
    return np.meshgrid(x, x)  # X along columns, Y along rows


def gtilt_slopes(phase, nLenslet, pix, d, valid, *, wavelength, order="F"):
    """Exact mean-gradient slopes [rad] from phase on the corner grid.

    Parameters
    ----------
    phase : numpy.ndarray
        ``(..., m, m)`` phase [rad at ``wavelength``], ``m = nLenslet*pix + 1``.
    valid : numpy.ndarray
        Valid-subaperture map.
    order : {"F", "C"}
        Subaperture enumeration of the output.

    Returns
    -------
    numpy.ndarray
        ``(..., 2*nValid)`` slopes, X block then Y block.
    """
    w = np.ones(pix + 1)
    w[[0, -1]] = 0.5
    w /= w.sum()
    N = nLenslet
    m = N * pix + 1
    # Trapezoid weights of each lenslet along the grid, (N, m).
    Wl = np.zeros((N, m))
    for r in range(N):
        Wl[r, r * pix : (r + 1) * pix + 1] = w
    ex = phase[..., :, pix::pix] - phase[..., :, 0:-pix:pix]  # (..., m, N) right - left edge
    ey = phase[..., pix::pix, :] - phase[..., 0:-pix:pix, :]  # (..., N, m) top - bottom edge
    sx = np.einsum("rm,...mc->...rc", Wl, ex)
    sy = np.einsum("cm,...rm->...rc", Wl, ey)
    scale = wavelength / (2 * np.pi) / d
    if order == "F":
        cols, rows = np.nonzero(valid.T)
    else:
        rows, cols = np.nonzero(valid)
    return np.concatenate([sx[..., rows, cols], sy[..., rows, cols]], axis=-1) * scale


def simulate_slopes(
    valid,
    D,
    gsDirections,
    altitudes,
    fractions,
    *,
    r0,
    L0,
    nFrames,
    gsHeight=np.inf,
    wavelength=5e-7,
    pix=4,
    screenPix=1024,
    noiseRms=0.0,
    seed=0,
):
    """Slopes of random turbulence realisations seen by several SH-WFS.

    Each FFT draw gives two screens per layer; every screen is sampled at a grid of
    footprint positions separated by the full metapupil size, so frames from one screen are
    spatially distinct but (like real frozen-flow data) not fully independent at scales
    approaching ``L0``.

    Returns
    -------
    numpy.ndarray
        ``(nFrames, nGS*2*nValid)``, per WFS X block then Y block, subapertures column-major.
    """
    from scipy.ndimage import spline_filter

    rng = np.random.default_rng(seed)
    N = valid.shape[0]
    d = D / N
    dx = d / pix
    px, py = pupil_points(N, d, pix)
    gs = np.asarray(gsDirections, float)
    nGS = gs.shape[0]
    altitudes = np.asarray(altitudes, float)
    fractions = np.asarray(fractions, float) / np.sum(fractions)
    span = D + 2 * np.abs(gs).max() * altitudes.max() + 8 * dx
    step = int(np.ceil(span / dx))
    nPos = max(1, (screenPix - 8) // step)
    positions = [(a, b) for a in range(nPos) for b in range(nPos)]
    origin = -screenPix / 2 + 4 + span / dx / 2

    # Layer-plane sample coordinates [pix] of every guide star, relative to the footprint.
    rel = []
    for h in altitudes:
        s = 1 - h / gsHeight if np.isfinite(gsHeight) else 1.0
        rel.append([((s * py + g[1] * h) / dx, (s * px + g[0] * h) / dx) for g in gs])

    nV = int(valid.sum())
    out = np.zeros((nFrames, nGS, 2 * nV))
    t = 0
    while t < nFrames:
        screens = []
        for f in fractions:
            if f > 0:
                a, b = ft_phase_screens(screenPix, dx, r0 * f ** (-3 / 5), L0, rng)
                screens.append((spline_filter(a), spline_filter(b)))
            else:
                screens.append(None)
        for half in (0, 1):
            for pa, pb in positions:
                if t >= nFrames:
                    break
                cy = origin + pa * step + screenPix / 2
                cx = origin + pb * step + screenPix / 2
                phase = np.zeros((nGS, *px.shape))
                for k, scr in enumerate(screens):
                    if scr is None:
                        continue
                    for g in range(nGS):
                        Yr, Xr = rel[k][g]
                        phase[g] += map_coordinates(
                            scr[half],
                            [(Yr + cy).ravel(), (Xr + cx).ravel()],
                            order=3,
                            prefilter=False,
                            mode="grid-wrap",
                        ).reshape(px.shape)
                out[t] = gtilt_slopes(phase, N, pix, d, valid, wavelength=wavelength)
                t += 1
    if noiseRms > 0:
        out += noiseRms * rng.standard_normal(out.shape)
    return out.reshape(nFrames, -1)


def bin_profile(altitudes, fractions, bins):
    """Distribute a layered profile onto a bin grid by linear (triangle) interpolation.

    This is the profile SLODAR can at best recover with response functions on ``bins``: a
    layer between two bins excites the two neighbouring response functions roughly in
    proportion to its distance from each.
    """
    bins = np.asarray(bins, float)
    out = np.zeros(bins.size)
    for h, f in zip(altitudes, fractions):
        if h <= bins[0]:
            out[0] += f
        elif h >= bins[-1]:
            out[-1] += f
        else:
            k = np.searchsorted(bins, h) - 1
            t = (h - bins[k]) / (bins[k + 1] - bins[k])
            out[k] += f * (1 - t)
            out[k + 1] += f * t
    return out
