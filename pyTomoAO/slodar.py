"""
SLODAR (SLOpe Detection And Ranging) turbulence profiling.

This module estimates the vertical turbulence profile :math:`C_n^2(h)` from simultaneous
open-loop (or pseudo-open-loop) slope time series recorded by several Shack-Hartmann
wavefront sensors looking at different guide stars.

The method follows Wilson (2002, MNRAS 337, 103) and Butterley, Wilson & Sarazin (2006,
MNRAS 369, 835): the spatial cross-covariance of the slopes measured by two sensors whose
guide stars are separated by :math:`\\theta` peaks at a subaperture offset
:math:`\\delta` set by the altitude of each layer, :math:`h_\\delta = \\delta\\, d/\\theta`.
The profile is recovered by fitting the measured covariance maps with theoretical
impulse-response maps -- the covariance a thin layer of unit strength at altitude
:math:`h` would produce -- under a non-negativity constraint.

The theoretical maps are built from pyTomoAO's own von Karman phase covariance
(:func:`pyTomoAO.tomographyUtilsCPU._covariance_matrix`) and the same Fried-geometry
gradient stencil as the tomographic gradient operator
(:func:`pyTomoAO.tomographyUtilsCPU._sparseGradientMatrixAmplitudeWeighted`), so a profile
measured with this module is expressed in exactly the turbulence model the reconstructor
uses, and can be fed back to it through :meth:`slodarResult.to_atmosphere_config`.

Conventions
-----------
- Slopes are angles of arrival (wavefront tilt) in **radians**. Multiply by
  ``slopeScale`` to convert other units.
- Per sensor, slopes are ordered like the rest of pyTomoAO: ``"simu"`` = all X slopes then
  all Y slopes; valid subapertures are enumerated column-major (``subapOrder="F"``), which is
  the row order of the tomographic gradient operator. ``subapOrder="C"`` (row-major) is also
  accepted, e.g. for SPECULA slopes.
- X runs along the columns of ``validLLMap`` and Y along its rows, as in the gradient
  operator and :func:`~pyTomoAO.tomographyUtilsCPU._create_guide_star_grid`.
- Guide-star directions are Cartesian angular offsets ``[x, y]`` in radians.
- Altitudes are ranges along the line of sight unless stated otherwise;
  :class:`slodarResult` also reports zenith altitudes using ``zenithAngleInDeg``.
- Turbulence strengths are returned as :math:`r_0^{-5/3}` per layer at ``wavelength``
  (the quantity the covariance model is linear in), as :math:`C_n^2\\,dh` in
  :math:`m^{1/3}`, and as fractions summing to one (pyTomoAO's ``fractionnalR0``).

Cone effect
-----------
For a guide star at finite range :math:`H` (LGS), a pupil point :math:`\\mathbf{p}` seen by
guide star :math:`i` intersects a layer at range :math:`h` at
:math:`\\mathbf{x}_i = \\mathbf{p}(1-h/H) + \\boldsymbol{\\theta}_i h`; the subaperture footprint
shrinks by the same factor. Two subapertures of WFS :math:`i, j` separated by
:math:`\\boldsymbol{\\delta}` (in subapertures) therefore sample points separated by
:math:`\\boldsymbol{\\delta} d (1-h/H) + (\\boldsymbol{\\theta}_j-\\boldsymbol{\\theta}_i) h`, and
the native altitude of offset :math:`\\delta` along the baseline is compressed to

.. math:: h_\\delta = \\frac{\\delta d}{\\theta + \\delta d / H}.

Both the compressed offsets and the shrunken footprint are included in the response maps.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import lsq_linear, nnls
from scipy.special import gamma

from pyTomoAO._plotting import pyplot
from pyTomoAO.tomographyUtilsCPU import (
    _covariance_matrix,
    _sparseGradientMatrixAmplitudeWeighted,
)

# Module logger; see the note in pyTomoAO/reconstructor.py.
logger = logging.getLogger(__name__)

_RAD2ARCSEC = 180 * 3600 / np.pi

ARCSEC2RAD = np.pi / (180 * 3600)

#: Component names of the four covariance maps, in storage order. ``"xy"`` is the covariance
#: of X slopes on the first sensor with Y slopes on the second.
COMPONENTS = ("xx", "xy", "yx", "yy")

# r0^{-5/3} = _R0_CN2 * k^2 * Cn2 dh. The constant is the ratio of the Kolmogorov phase
# structure-function coefficients, 2.91438 / 6.88388 = 0.42336, written out so that it
# agrees exactly with the 6.88 used by the von Karman covariance kernel.
#: D_phi(r) = 8 pi^2 * 0.033 * I * k^2 Cn2dh r^{5/3} with 0.033 = Gamma(8/3) sin(pi/3)/(4 pi^2)
#: and I = int_0^inf (1 - J0(q)) q^{-8/3} dq = 2^{-5/3} Gamma(1/6) / ((5/3) Gamma(11/6)).
_KOLMOGOROV_CN2 = (
    8
    * np.pi**2
    * (gamma(8 / 3) * np.sin(np.pi / 3) / (4 * np.pi**2))
    * 2 ** (-5 / 3)
    * gamma(1 / 6)
    / ((5 / 3) * gamma(11 / 6))
)  # 2.9144
_KOLMOGOROV_R0 = 2 * (24 / 5 * gamma(6 / 5)) ** (5 / 6)  # 6.8839
_R0_CN2 = _KOLMOGOROV_CN2 / _KOLMOGOROV_R0  # 0.42336


def pseudo_open_loop_slopes(slopes, commands, interactionMatrix, delay=1, sign=1.0):
    """Pseudo-open-loop slopes from closed-loop telemetry.

    In closed loop the sensors measure the turbulence minus the correction the deformable
    mirror applied while the frame was integrated. Adding that correction back,

    .. math::
        s_{\\rm pol}(t) = s(t) + \\sigma\\, M_{\\rm int}\\, c(t - \\tau),

    recovers (to the accuracy of the interaction matrix and the sensor linearity) the slopes
    an open-loop sensor would have measured, which is what SLODAR and the MMSE
    reconstructor's covariance model describe.

    Parameters
    ----------
    slopes : array_like
        ``(nFrames, nSlopes)`` closed-loop slopes, one row per loop step.
    commands : array_like
        ``(nFrames, nActuators)`` commands, row ``t`` being the command *computed* at step
        ``t`` (the same time base as ``slopes``).
    interactionMatrix : array_like
        ``(nSlopes, nActuators)`` interaction matrix, in the units of ``slopes`` per unit
        command.
    delay : int, optional
        Frames between computing a command and the slopes it affects, ``tau`` (default 1:
        the command computed at ``t - 1`` was on the mirror during frame ``t``). Only whole
        frames are supported; real loops often have a fractional delay (e.g. about 1.5
        frames), which this approximates by the nearest integer.
    sign : float, optional
        How the recorded commands act on the slopes. Write the closed-loop measurement as
        ``s = s_ol + sigma_DM * M c`` with ``M`` the interaction matrix as measured and ``c``
        the recorded command; then ``sign = -sigma_DM``:

        - ``+1`` (default): the loop applies the *opposite* of the recorded command, so
          ``s = s_ol - M c``. This is the usual case when ``M`` is measured by poking
          ``+c`` but the DM is driven with ``-c``, as in SPECULA (``DM`` with its default
          ``sign: -1`` and an IM calibrated with ``sign: 1``).
        - ``-1``: the recorded command is applied as is and ``M`` was measured by poking
          that same command, so ``s = s_ol + M c``.

        Self-check: in a well-corrected loop the wrong sign gives ``s_pol`` close to
        ``-s_ol``, with almost the same variance (and spatial covariance) as the right
        one, so the variance does not tell them apart. The frame-to-frame increments do:
        the right sign recovers the temporally smooth open-loop slopes, while the wrong one
        keeps the servo-lag jitter. ``np.var(np.diff(s_pol, axis=0))`` is typically 2-4x
        smaller for the right sign.

    Returns
    -------
    numpy.ndarray
        ``(nFrames - delay, nSlopes)`` pseudo-open-loop slopes, aligned with
        ``slopes[delay:]``.
    """
    s = np.asarray(slopes, dtype=float)
    c = np.asarray(commands, dtype=float)
    M = np.asarray(interactionMatrix, dtype=float)
    if s.ndim != 2 or c.ndim != 2:
        raise ValueError("slopes and commands must be 2D (nFrames, n)")
    if s.shape[0] != c.shape[0]:
        raise ValueError("slopes and commands must have the same number of frames")
    if M.shape != (s.shape[1], c.shape[1]):
        raise ValueError(f"interactionMatrix must be ({s.shape[1]}, {c.shape[1]}), got {M.shape}")
    delay = int(delay)
    if not 0 <= delay < s.shape[0]:
        raise ValueError("delay must be in [0, nFrames)")
    logger.info(
        "\n-->> Pseudo-open-loop slopes: %d frames, %d slopes, %d actuators, delay %d, "
        "sign %+g <<--",
        s.shape[0],
        s.shape[1],
        c.shape[1],
        delay,
        sign,
    )
    return s[delay:] + sign * (c[: s.shape[0] - delay] @ M.T)


def slope_noise_covariance(
    noiseVariance, reconstructor=None, d=None, nValidSubap=None, wavelength=None
):
    """Per-slope noise variances for ``build_reconstructor(noiseCovariance=...)``.

    Converts per-WFS angle-of-arrival noise variances [rad^2] -- from
    :meth:`slodarProfiler.temporal_noise_variance` (recommended) or a SLODAR fit -- to the
    units of the reconstructor's slope covariance ``Gamma Cxx Gamma^T``: rad^2 of phase
    difference across a subaperture at the atmosphere wavelength,
    ``(2 pi d / wavelength)^2 * noiseVariance``.

    Parameters
    ----------
    noiseVariance : array_like
        ``(nGS,)`` per-sensor angle-of-arrival noise variances [rad^2].
    reconstructor : tomographicReconstructor, optional
        Takes ``d`` (the physical subaperture size ``D / nLenslet``, not the padded
        ``DSupport``), ``nValidSubap`` and ``wavelength`` (the atmosphere model's) from it.
        Explicit arguments override it.
    d : float, optional
        Subaperture size [m].
    nValidSubap : int, optional
        Valid subapertures per sensor (each sensor contributes ``2 * nValidSubap`` slopes).
    wavelength : float, optional
        Wavelength of the reconstructor's atmosphere model [m].

    Returns
    -------
    numpy.ndarray
        ``(nGS * 2 * nValidSubap,)`` variances.
    """
    v = np.atleast_1d(np.asarray(noiseVariance, dtype=float))
    if v.size == 0:
        raise ValueError("No noise estimate (empty noiseVariance)")
    if np.any(v < 0):
        raise ValueError("noise variances must be non-negative")
    if reconstructor is not None:
        w = reconstructor.lgsWfsParams
        d = w.D / w.nLenslet if d is None else d
        nValidSubap = w.nValidSubap if nValidSubap is None else nValidSubap
        wavelength = reconstructor.atmParams.wavelength if wavelength is None else wavelength
    if d is None or nValidSubap is None or wavelength is None:
        raise ValueError("Give a reconstructor, or d, nValidSubap and wavelength")
    phase = (2 * np.pi * float(d) / float(wavelength)) ** 2 * v
    logger.info(
        "\nSlope noise covariance: %d sensor(s), noise %s arcsec rms -> %s rad^2 "
        "(phase difference over d = %.3f m at %.0f nm), %d slopes per sensor",
        v.size,
        np.array2string(np.sqrt(v) * _RAD2ARCSEC, precision=4),
        np.array2string(phase, precision=3),
        float(d),
        float(wavelength) * 1e9,
        2 * int(nValidSubap),
    )
    return np.repeat(phase, 2 * int(nValidSubap))


def _nnls(A, b):
    """Non-negative least squares ``min ||A x - b||, x >= 0`` with an optimality check.

    ``scipy.optimize.nnls`` can stop at a non-optimal point on some SciPy releases (seen with
    SciPy 1.13, the newest available for Python 3.9: a residual 10x the minimum and a biased
    SLODAR profile). The result is therefore checked against the Karush-Kuhn-Tucker
    conditions -- the gradient ``A^T (A x - b)`` must vanish on the free variables and be
    non-negative on the bound ones -- and, if they fail, refined with a bounded-variable
    least-squares solve.

    Parameters
    ----------
    A : numpy.ndarray
        ``(m, n)`` design matrix.
    b : numpy.ndarray
        ``(m,)`` data.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` non-negative solution.
    """
    x, _ = nnls(A, b, maxiter=50 * A.shape[1])
    grad = A.T @ (A @ x - b)
    scale = np.max(np.abs(A.T @ b))
    tol = 1e-8 * scale
    free = x > 0
    if scale == 0 or (np.all(np.abs(grad[free]) <= tol) and np.all(grad[~free] >= -tol)):
        return x
    ref = lsq_linear(A, b, bounds=(0, np.inf), method="bvls", tol=1e-12).x
    ref = np.clip(ref, 0, None)
    if np.linalg.norm(A @ ref - b) < np.linalg.norm(A @ x - b):
        logger.debug("nnls returned a non-optimal point; using the BVLS solution instead")
        return ref
    return x


def _gradient_stencil(nEdgePoints=3):
    """Points and weights of the Fried-geometry Shack-Hartmann gradient.

    The X slope of a subaperture of size ``d`` is modelled as the trapezoid-weighted mean,
    along the subaperture edge, of the phase difference between its right and left edges,
    divided by ``d`` (and the transposed stencil for Y). With ``nEdgePoints=3`` this is
    exactly the 3x3 stencil of pyTomoAO's tomographic gradient operator.

    Parameters
    ----------
    nEdgePoints : int, optional
        Number of samples along each subaperture edge (>= 2, default 3).

    Returns
    -------
    points : numpy.ndarray
        ``(nPoints, 2)`` stencil positions ``[x, y]`` in units of the subaperture size,
        relative to the subaperture centre.
    wx, wy : numpy.ndarray
        ``(nPoints,)`` weights such that ``slope_x * d = sum(wx * phase(points))``.
    """
    if nEdgePoints < 2:
        raise ValueError("nEdgePoints must be at least 2")
    t = np.linspace(-0.5, 0.5, nEdgePoints)
    w = np.ones(nEdgePoints)
    w[[0, -1]] = 0.5
    w /= w.sum()

    pts = {}

    def add(x, y, ax, wt):
        key = (round(x, 12), round(y, 12))
        entry = pts.setdefault(key, [0.0, 0.0])
        entry[ax] += wt

    for tk, wk in zip(t, w):
        add(0.5, tk, 0, wk)
        add(-0.5, tk, 0, -wk)
        add(tk, 0.5, 1, wk)
        add(tk, -0.5, 1, -wk)
    keys = sorted(pts)
    points = np.array(keys, dtype=float)
    wx = np.array([pts[k][0] for k in keys])
    wy = np.array([pts[k][1] for k in keys])
    return points, wx, wy


def _pyTomoAO_stencil():
    """The 3x3 gradient stencil exactly as pyTomoAO's gradient operator builds it.

    Returns
    -------
    points, wx, wy : numpy.ndarray
        Same layout as :func:`_gradient_stencil`.
    """
    G, _ = _sparseGradientMatrixAmplitudeWeighted(np.ones((1, 1), dtype=bool), None, 2)
    G = G.toarray()
    r, c = np.divmod(np.arange(9), 3)
    points = np.stack([(c - 1) / 2.0, (r - 1) / 2.0], axis=1)
    return points, G[0], G[1]


@dataclass
class slodarResult:
    """Turbulence profile returned by :meth:`slodarProfiler.fit_profile`.

    Attributes
    ----------
    altitude : numpy.ndarray
        Layer ranges along the line of sight [m].
    altitude_zenith : numpy.ndarray
        Layer altitudes above the telescope, ``altitude * cos(zenith)`` [m].
    r0m53 : numpy.ndarray
        Layer strengths as :math:`r_0^{-5/3}` at ``wavelength`` [m^-5/3], line of sight.
    cn2dh : numpy.ndarray
        Integrated :math:`C_n^2\\,dh` per layer [m^1/3], line of sight.
    fractionalCn2 : numpy.ndarray
        Normalised layer weights (sum to one); pyTomoAO's ``fractionnalR0``.
    r0 : float
        Total Fried parameter along the line of sight at ``wavelength`` [m].
    r0_zenith : float
        Total Fried parameter scaled to zenith [m].
    L0 : float
        Outer scale used (or estimated) for the fit [m].
    wavelength : float
        Wavelength at which ``r0`` is given [m].
    zenithAngleInDeg : float
        Zenith angle used to convert ranges to altitudes.
    noiseVariance : numpy.ndarray
        Per-WFS slope noise variance estimated from the auto-covariance [rad^2], or an empty
        array when the auto-covariance was not fitted.
    residualRms : float
        Weighted RMS of the fit residual, relative to the weighted RMS of the data.
    L0Grid, L0Cost : numpy.ndarray
        Outer scales tried and their fit cost when ``fitL0`` was requested (else empty).
        See :attr:`L0AtGridEdge`.
    measured, model : numpy.ndarray
        The fitted data vector and the best-fit model, for diagnostics.
    """

    altitude: np.ndarray
    altitude_zenith: np.ndarray
    r0m53: np.ndarray
    cn2dh: np.ndarray
    fractionalCn2: np.ndarray
    r0: float
    r0_zenith: float
    L0: float
    wavelength: float
    zenithAngleInDeg: float = 0.0
    noiseVariance: np.ndarray = field(default_factory=lambda: np.zeros(0))
    residualRms: float = np.nan
    L0Grid: np.ndarray = field(default_factory=lambda: np.zeros(0))
    L0Cost: np.ndarray = field(default_factory=lambda: np.zeros(0))
    measured: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    model: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)

    @property
    def L0AtGridEdge(self):
        """True when ``L0`` was fitted and the best value is at an end of ``L0Grid``.

        The fit minimum then lies outside the grid, so L0 is not constrained by the data.
        This is typical of short telemetry records, where it can ruin the profile.
        """
        if self.L0Grid.size == 0:
            return False
        lo, hi = self.L0Grid.min(), self.L0Grid.max()
        return bool(np.isclose(self.L0, lo) or np.isclose(self.L0, hi))

    def to_atmosphere_config(self, windSpeed=None, windDirection=None, minFraction=0.0):
        """Express the profile as an ``atmosphere_parameters`` configuration section.

        The result can be dropped into a pyTomoAO YAML configuration (or passed to
        :class:`~pyTomoAO.atmosphereParametersClass.atmosphereParameters` inside a
        ``{"atmosphere_parameters": ...}`` dictionary) so that a reconstructor can be rebuilt
        on the measured profile.

        Parameters
        ----------
        windSpeed, windDirection : array_like, optional
            Per-layer wind; SLODAR from spatial covariances does not measure it, so zeros are
            used by default.
        minFraction : float, optional
            Layers with a fractional strength below this are dropped (default 0, keep all).

        Returns
        -------
        dict
            Keys as documented for ``atmosphere_parameters``. Altitudes are zenith
            altitudes in km and ``r0`` is the zenith value, as the configuration expects.

        Notes
        -----
        Dropping layers with ``minFraction`` renormalises the remaining fractions but keeps
        the total ``r0``, so the strength of the dropped layers is shared out over the kept
        ones in proportion to their own.
        """
        keep = self.fractionalCn2 > minFraction
        if not keep.any():
            raise ValueError("No layer above minFraction")
        frac = self.fractionalCn2[keep] / self.fractionalCn2[keep].sum()
        n = int(keep.sum())
        logger.info(
            "\nSLODAR profile -> atmosphere_parameters: %d of %d layers kept (minFraction "
            "%g, %.3f of the turbulence dropped and shared out), r0 %.4f m at zenith, "
            "L0 %.1f m",
            n,
            keep.size,
            minFraction,
            float(self.fractionalCn2[~keep].sum()),
            self.r0_zenith,
            self.L0,
        )
        ws = np.zeros(n) if windSpeed is None else np.asarray(windSpeed, float)[keep]
        wd = np.zeros(n) if windDirection is None else np.asarray(windDirection, float)[keep]
        return {
            "nLayer": n,
            "zenithAngleInDeg": float(self.zenithAngleInDeg),
            "altitude": (self.altitude_zenith[keep] / 1e3).tolist(),
            "L0": float(self.L0),
            "r0": float(self.r0_zenith),
            "fractionnalR0": frac.tolist(),
            "wavelength": float(self.wavelength),
            "windDirection": wd.tolist(),
            "windSpeed": ws.tolist(),
        }

    def noise_covariance(self, d=None, nValidSubap=None, wavelength=None, reconstructor=None):
        """Per-slope noise variances from this fit, for ``build_reconstructor``.

        See :func:`slope_noise_covariance`; ``wavelength`` defaults to the reconstructor's
        atmosphere wavelength when ``reconstructor`` is given, else to the profile's.

        .. caution::
           On real Shack-Hartmann data the noise fitted here, from the zero-offset excess
           of the auto-covariance, absorbs any mismatch between the sensor and the
           geometric slope model at zero offset. In SPECULA simulations with a physical
           Shack-Hartmann it overestimated the noise by up to 5x. For the reconstructor,
           prefer :meth:`slodarProfiler.temporal_noise_variance` on the (pseudo-)open-loop
           slopes. At low flux, also calibrate the centroid gain first (see the SLODAR user
           guide).

        Parameters
        ----------
        d, nValidSubap, wavelength, reconstructor
            As in :func:`slope_noise_covariance`.

        Returns
        -------
        numpy.ndarray
            ``(nGS * 2 * nValidSubap,)`` variances.
        """
        if self.noiseVariance.size == 0:
            raise ValueError("No noise estimate: fit the profile with useAuto=True, fitNoise=True")
        if wavelength is None and reconstructor is None:
            wavelength = self.wavelength
        return slope_noise_covariance(
            self.noiseVariance,
            reconstructor=reconstructor,
            d=d,
            nValidSubap=nValidSubap,
            wavelength=wavelength,
        )

    def __str__(self):
        lines = [
            "SLODAR profile:",
            f"  - r0 (line of sight) = {self.r0:.4f} m, r0 (zenith) = {self.r0_zenith:.4f} m "
            f"@ {self.wavelength * 1e9:.0f} nm",
            f"  - L0 = {self.L0:.1f} m, relative residual = {self.residualRms:.3f}",
        ]
        for h, f, c in zip(self.altitude_zenith, self.fractionalCn2, self.cn2dh):
            lines.append(f"    {h / 1e3:7.2f} km : {f:6.3f}  (Cn2dh = {c:.3e} m^1/3)")
        return "\n".join(lines)

    def plot(self, ax=None, reference=None, label="SLODAR"):
        """Plot the fractional profile against altitude.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into; a new figure is created by default.
        reference : tuple of array_like, optional
            ``(altitude_m, fraction)`` of a reference profile to overplot.
        label : str, optional
            Legend label of the SLODAR profile.

        Returns
        -------
        matplotlib.axes.Axes
        """
        plt = pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 4))
        ax.plot(self.fractionalCn2, self.altitude_zenith / 1e3, "o-", label=label)
        if reference is not None:
            ax.plot(reference[1], np.asarray(reference[0]) / 1e3, "s--", label="reference")
        ax.set_xlabel("fractional $C_n^2$")
        ax.set_ylabel("altitude [km]")
        ax.legend()
        return ax


class slodarProfiler:
    """
    SLODAR turbulence profiler for a multi-guide-star Shack-Hartmann system.

    Parameters
    ----------
    validLLMap : array_like of bool
        ``(nLenslet, nLenslet)`` valid-subaperture map, shared by all sensors.
    D : float
        Pupil diameter spanned by ``nLenslet`` subapertures [m]; the subaperture size is
        ``d = D / nLenslet``.
    gsDirections : array_like
        ``(nGS, 2)`` Cartesian angular positions ``[x, y]`` of the guide stars [rad].
    gsHeight : float, optional
        Guide-star range [m]; ``np.inf`` (default) for natural guide stars.
    L0 : float, optional
        Outer scale used by the response functions [m] (default 25).
    wavelength : float, optional
        Wavelength at which ``r0`` is reported [m] (default 500 nm).
    conjugationAltitude : float, optional
        Altitude the WFS lenslet arrays are optically conjugated to [m]. Zero (default) is
        the pupil-conjugated case; a negative value gives *generalised SLODAR*, in which the
        ground layer is displaced from zero offset and can be resolved. This only changes the
        geometric model -- it must match how the sensors were actually built.
    zenithAngleInDeg : float, optional
        Zenith angle of the observation, used to report altitudes and zenith ``r0``.
    subapOrder : {"F", "C"}, optional
        Enumeration of valid subapertures in the slope vector: column-major (``"F"``,
        default, as pyTomoAO's gradient operator) or row-major (``"C"``).
    slopesOrder : {"simu", "inverted", "keck"}, optional
        Per-sensor slope layout: ``"simu"`` = X block then Y block (default), ``"inverted"``
        = Y block then X block, ``"keck"`` = interleaved ``[x0, y0, x1, y1, ...]``.
    slopeScale : float, optional
        Factor converting the input slopes to radians (default 1).
    nEdgePoints : int, optional
        Samples per subaperture edge in the gradient model. 3 (default) reproduces
        pyTomoAO's tomographic gradient operator.

    Examples
    --------
    >>> from pyTomoAO import example_config
    >>> from pyTomoAO.slodar import slodarProfiler
    >>> prof = slodarProfiler.from_config(example_config("kapa"))  # doctest: +SKIP
    >>> prof.compute_covariance_maps(slopes, removeTT=True)  # doctest: +SKIP
    >>> result = prof.fit_profile()  # doctest: +SKIP
    """

    def __init__(
        self,
        validLLMap,
        D,
        gsDirections,
        gsHeight=np.inf,
        L0=25.0,
        wavelength=5e-7,
        conjugationAltitude=0.0,
        zenithAngleInDeg=0.0,
        subapOrder="F",
        slopesOrder="simu",
        slopeScale=1.0,
        nEdgePoints=3,
    ):
        self.validLLMap = np.asarray(validLLMap, dtype=bool)
        if self.validLLMap.ndim != 2 or self.validLLMap.shape[0] != self.validLLMap.shape[1]:
            raise ValueError("validLLMap must be a square 2D map")
        if D <= 0:
            raise ValueError("D must be positive")
        self.D = float(D)
        self.nLenslet = self.validLLMap.shape[0]
        self.d = self.D / self.nLenslet

        gs = np.atleast_2d(np.asarray(gsDirections, dtype=float))
        if gs.shape[1] != 2:
            raise ValueError("gsDirections must have shape (nGS, 2)")
        self.gsDirections = gs
        self.nGS = gs.shape[0]

        if gsHeight <= 0:
            raise ValueError("gsHeight must be positive")
        self.gsHeight = float(gsHeight)
        if L0 <= 0:
            raise ValueError("L0 must be positive")
        self.L0 = float(L0)
        self.wavelength = float(wavelength)
        self.conjugationAltitude = float(conjugationAltitude)
        if np.isfinite(self.gsHeight) and self.conjugationAltitude >= self.gsHeight:
            raise ValueError("conjugationAltitude must be below the guide stars")
        if not 0 <= zenithAngleInDeg < 90:
            raise ValueError("zenithAngleInDeg must be in [0, 90)")
        self.zenithAngleInDeg = float(zenithAngleInDeg)

        if subapOrder not in ("F", "C"):
            raise ValueError("subapOrder must be 'F' or 'C'")
        self.subapOrder = subapOrder
        if slopesOrder not in ("simu", "inverted", "keck"):
            raise ValueError("slopesOrder must be 'simu', 'inverted' or 'keck'")
        self.slopesOrder = slopesOrder
        self.slopeScale = float(slopeScale)

        if nEdgePoints == 3:
            self._stencil = _pyTomoAO_stencil()
        else:
            self._stencil = _gradient_stencil(nEdgePoints)
        self.nEdgePoints = nEdgePoints

        # Valid subaperture (row, col) positions in slope-vector order.
        if subapOrder == "F":
            cols, rows = np.nonzero(self.validLLMap.T)
        else:
            rows, cols = np.nonzero(self.validLLMap)
        self._rows, self._cols = rows, cols
        self.nValidSubap = rows.size

        N = self.nLenslet
        self._mapShape = (2 * N - 1, 2 * N - 1)
        dyy, dxx = np.mgrid[-(N - 1) : N, -(N - 1) : N]
        self._offsets = np.stack([dxx.ravel(), dyy.ravel()], axis=1).astype(float)

        # Offset index of every ordered subaperture pair (p, q), q - p, for the full-matrix
        # (tip-tilt removed) response.
        drow = rows[None, :] - rows[:, None]
        dcol = cols[None, :] - cols[:, None]
        self._pairOffsetIndex = ((drow + N - 1) * (2 * N - 1) + (dcol + N - 1)).ravel()
        self._counts = self._mask_counts()

        # Measured maps, filled by compute_covariance_maps.
        self.maps = None
        self.nFrames = 0
        self.removeTT = False
        self._responseCache = {}

        logger.info("\n-->> Initializing SLODAR profiler <<--")
        logger.info(
            "\n%d guide stars (%s), %dx%d lenslets, %d valid subapertures, D = %.2f m (d = %.3f m)",
            self.nGS,
            f"LGS at {self.gsHeight / 1e3:.1f} km range" if np.isfinite(self.gsHeight) else "NGS",
            self.nLenslet,
            self.nLenslet,
            self.nValidSubap,
            self.D,
            self.d,
        )
        logger.info(
            "\nGuide-star separations: %.1f-%.1f arcsec; L0 %.1f m, wavelength %.0f nm, "
            "zenith angle %.1f deg, conjugation %.0f m",
            *self._separation_range(),
            self.L0,
            self.wavelength * 1e9,
            self.zenithAngleInDeg,
            self.conjugationAltitude,
        )
        logger.info(
            "\nSlope layout: subapOrder %r, slopesOrder %r, slopeScale %.4g rad/unit, "
            "%d-point edge stencil",
            self.subapOrder,
            self.slopesOrder,
            self.slopeScale,
            self.nEdgePoints,
        )

    def _separation_range(self):
        """Smallest and largest guide-star separation [arcsec] (0, 0 for one star)."""
        if self.nGS < 2:
            return 0.0, 0.0
        sep = [self.separation(i, j) * _RAD2ARCSEC for i, j in self.pairs]
        return min(sep), max(sep)

    # ------------------------------------------------------------------
    # Construction from pyTomoAO configuration objects
    @classmethod
    def from_parameters(cls, lgsWfsParams, lgsAsterismParams, atmParams, **kwargs):
        """Build a profiler from pyTomoAO parameter objects.

        Parameters
        ----------
        lgsWfsParams : lgsWfsParameters
            Provides ``validLLMap`` and ``D``.
        lgsAsterismParams : lgsAsterismParameters
            Provides the guide-star directions and ``LGSheight`` (airmass-scaled range).
        atmParams : atmosphereParameters
            Provides ``L0``, ``wavelength`` and ``zenithAngleInDeg``.
        **kwargs
            Overrides for any other :class:`slodarProfiler` argument.

        Returns
        -------
        slodarProfiler
        """
        dv = lgsAsterismParams.directionVectorLGS
        args = {
            "validLLMap": lgsWfsParams.validLLMap,
            "D": lgsWfsParams.D,
            # directionVectorLGS holds tan(zenith) * (cos az, sin az); arctan turns each
            # component back into an angle. The reconstructor kernels use the tangents
            # directly; the two differ by theta^2 / 3 relative, 3e-8 at 60 arcsec.
            "gsDirections": np.stack([np.arctan(dv[0]), np.arctan(dv[1])], axis=1),
            "gsHeight": lgsAsterismParams.LGSheight,
            "L0": atmParams.L0,
            "wavelength": atmParams.wavelength,
            "zenithAngleInDeg": atmParams.zenithAngleInDeg,
        }
        args.update(kwargs)
        return cls(**args)

    @classmethod
    def from_config(cls, config_file, **kwargs):
        """Build a profiler from a pyTomoAO YAML configuration file.

        Only the ``atmosphere_parameters``, ``lgs_asterism`` and ``lgs_wfs_parameters``
        sections are read.

        Parameters
        ----------
        config_file : str
            Path to the YAML configuration.
        **kwargs
            Overrides passed to :meth:`from_parameters`.

        Returns
        -------
        slodarProfiler
        """
        import yaml

        from pyTomoAO.atmosphereParametersClass import atmosphereParameters
        from pyTomoAO.lgsAsterismParametersClass import lgsAsterismParameters
        from pyTomoAO.lgsWfsParametersClass import lgsWfsParameters

        logger.info("\nReading the SLODAR geometry from %s", config_file)
        with open(config_file) as f:
            config = yaml.safe_load(f)
        atm = atmosphereParameters(config)
        ast = lgsAsterismParameters(config, atm)
        wfs = lgsWfsParameters(config, ast)
        return cls.from_parameters(wfs, ast, atm, **kwargs)

    @classmethod
    def from_reconstructor(cls, rec, **kwargs):
        """Build a profiler sharing the geometry of a ``tomographicReconstructor``.

        Parameters
        ----------
        rec : tomographicReconstructor
        **kwargs
            Overrides passed to :meth:`from_parameters`.

        Returns
        -------
        slodarProfiler
        """
        logger.info("\nTaking the SLODAR geometry from a tomographicReconstructor")
        return cls.from_parameters(rec.lgsWfsParams, rec.lgsAsterismParams, rec.atmParams, **kwargs)

    # ------------------------------------------------------------------
    # Geometry
    @property
    def airmass(self):
        """Airmass ``1 / cos(zenith)``."""
        return 1.0 / np.cos(np.radians(self.zenithAngleInDeg))

    @property
    def pairs(self):
        """Cross pairs ``(i, j)`` with ``i < j``."""
        return [(i, j) for i in range(self.nGS) for j in range(i + 1, self.nGS)]

    def separation(self, i, j):
        """Angular separation of guide stars ``i`` and ``j`` [rad]."""
        return float(np.hypot(*(self.gsDirections[j] - self.gsDirections[i])))

    def _layer_geometry(self, h):
        """Footprint scale and the layer shift per unit angular offset at range ``h``.

        Returns ``(s, a)`` such that a pupil (conjugate-plane) point ``p`` of guide star
        ``theta`` lands at ``s * p + a * theta`` on the layer.
        """
        hc = self.conjugationAltitude
        H = self.gsHeight
        if np.isfinite(H):
            if h >= H:
                raise ValueError(f"Altitude {h} m is not below the guide stars ({H} m)")
            s = (H - h) / (H - hc)
            return s, H * (1 - s)
        return 1.0, h - hc

    def lattice_step(self, i, j):
        """Subaperture-lattice step along the baseline of pair ``(i, j)`` [m].

        Offsets along a baseline at angle :math:`\\phi` cross whole subapertures of the
        dominant axis every :math:`d / \\max(|\\cos\\phi|, |\\sin\\phi|)`; this is ``d`` for
        baselines along X or Y and :math:`\\sqrt{2} d` for diagonal ones.
        """
        b = self.gsDirections[j] - self.gsDirections[i]
        nb = np.hypot(*b)
        if nb == 0:
            raise ValueError("Guide stars i and j coincide")
        return self.d / max(abs(b[0]), abs(b[1])) * nb

    def native_altitudes(self, i, j, nMax=None):
        """Native SLODAR altitude bins of a guide-star pair.

        The range whose covariance peak falls ``delta`` lattice steps ``d'`` along the
        baseline (see :meth:`lattice_step`):

        - natural guide stars, ``h = hc + delta d' / theta``;
        - guide stars at range ``H`` (cone effect),
          ``h = H (delta d' + theta hc) / (delta d' + theta H)``, which for ``hc = 0`` is
          ``h = delta d' / (theta + delta d' / H)``: the NGS bins compressed by ``1 - h/H``.

        Parameters
        ----------
        i, j : int
            Guide-star indices (``i != j``).
        nMax : int, optional
            Largest offset in lattice steps (default ``nLenslet - 1``).

        Returns
        -------
        numpy.ndarray
            Ranges [m] for ``delta = 0 .. nMax``.
        """
        theta = self.separation(i, j)
        step = self.lattice_step(i, j)
        nMax = self.nLenslet - 1 if nMax is None else nMax
        delta = np.arange(nMax + 1) * step
        hc = self.conjugationAltitude
        if np.isfinite(self.gsHeight):
            H = self.gsHeight
            return H * (delta + theta * hc) / (delta + theta * H)
        return hc + delta / theta

    def default_altitudes(self, hMax=None, oversample=1):
        """Altitude grid combining the native bins of every baseline.

        The baseline with the finest native spacing sets the grid at low altitude; the
        others, which reach higher, extend it upwards with their own (coarser) spacing.

        Parameters
        ----------
        hMax : float, optional
            Truncate the grid at this range [m].
        oversample : int, optional
            Insert ``oversample - 1`` extra bins between native ones (default 1, none).
            Response functions closer than the native resolution are strongly correlated,
            so oversampling mostly helps when paired with ``regularization``.

        Returns
        -------
        numpy.ndarray
            Sorted ranges [m], starting at the conjugation altitude (or 0).
        """
        nativeSets = []
        seen = set()
        for i, j in self.pairs:
            key = (round(self.separation(i, j), 12), round(self.lattice_step(i, j), 9))
            if key not in seen:
                seen.add(key)
                nativeSets.append(self.native_altitudes(i, j))
        nativeSets.sort(key=lambda b: b[1] - b[0])
        grid = [max(self.conjugationAltitude, 0.0)]
        top = grid[0]
        for bins in nativeSets:
            spacing = bins[1] - bins[0]
            for b in bins:
                if b > top + 0.5 * spacing:
                    grid.append(b)
                    top = b
        grid = np.array(grid)
        if oversample > 1:
            fine = [
                np.linspace(a, b, oversample, endpoint=False) for a, b in zip(grid[:-1], grid[1:])
            ]
            grid = np.concatenate([*fine, grid[-1:]])
        if hMax is not None:
            grid = grid[grid <= hMax]
        return grid

    # ------------------------------------------------------------------
    # Data handling
    def _reshape_slopes(self, slopes):
        """Return slopes as ``(nFrames, nGS, 2, nValidSubap)`` in radians."""
        s = np.asarray(slopes, dtype=float)
        n = self.nValidSubap
        if s.ndim == 2:
            if s.shape[1] != self.nGS * 2 * n:
                raise ValueError(
                    f"Expected {self.nGS * 2 * n} slopes per frame "
                    f"({self.nGS} WFS x 2 x {n}), got {s.shape[1]}"
                )
            s = s.reshape(s.shape[0], self.nGS, 2 * n)
        if s.ndim != 3 or s.shape[1:] != (self.nGS, 2 * n):
            raise ValueError(
                "slopes must have shape (nFrames, nGS*2*nValid) or (nFrames, nGS, 2*nValid)"
            )
        if self.slopesOrder == "simu":
            out = s.reshape(s.shape[0], self.nGS, 2, n)
        elif self.slopesOrder == "inverted":
            out = s.reshape(s.shape[0], self.nGS, 2, n)[:, :, ::-1]
        else:
            out = s.reshape(s.shape[0], self.nGS, n, 2).transpose(0, 1, 3, 2)
        return out * self.slopeScale

    def _format_slopes(self, s):
        """Inverse of :meth:`_reshape_slopes` (without the unit scaling)."""
        T = s.shape[0]
        if self.slopesOrder == "simu":
            out = s
        elif self.slopesOrder == "inverted":
            out = s[:, :, ::-1]
        else:
            out = s.transpose(0, 1, 3, 2)
        return np.ascontiguousarray(out).reshape(T, -1)

    def _mask_counts(self):
        """Number of valid subaperture pairs at each offset, ``(2N-1, 2N-1)``."""
        counts = np.bincount(self._pairOffsetIndex, minlength=np.prod(self._mapShape))
        return counts.reshape(self._mapShape)

    @property
    def counts(self):
        """Valid subaperture pairs per offset, ``(2N-1, 2N-1)``; the map normalisation."""
        return self._counts

    def compute_covariance_maps(self, slopes, removeTT=False, chunkSize=None):
        """Compute the slope auto- and cross-covariance maps of all sensor pairs.

        For sensors ``i, j`` and slope components ``a, b`` the map is

        .. math::
            C_{ij}^{ab}(\\boldsymbol\\delta) = \\frac{1}{N_{\\boldsymbol\\delta}}
            \\sum_{\\mathbf p} \\langle s_i^a(\\mathbf p)\\, s_j^b(\\mathbf p+\\boldsymbol\\delta)
            \\rangle_t,

        where the sum runs over the :math:`N_{\\boldsymbol\\delta}` valid subaperture pairs
        at offset :math:`\\boldsymbol\\delta` (X along columns, Y along rows). The temporal
        mean of every slope is removed first.

        Parameters
        ----------
        slopes : numpy.ndarray
            ``(nFrames, nGS*2*nValid)`` or ``(nFrames, nGS, 2*nValid)`` open-loop or
            pseudo-open-loop slopes, ordered as configured.
        removeTT : bool, optional
            Remove the per-frame mean X and Y slope (tip-tilt) of every sensor before
            correlating, as needed for LGS sensors whose tip-tilt is unreliable. The
            response maps then include the same projection (default False).
        chunkSize : int, optional
            Frames processed per FFT batch. The batch's FFT buffer takes
            ``chunkSize * nGS * 2 * 2N * (N + 1) * 16`` bytes (about 0.3 GB for 2000 frames of
            six 20x20 sensors, but 3 GB for six 60x60 ones). By default it is sized to keep
            that buffer near 256 MB.

        Returns
        -------
        dict
            ``{(i, j): ndarray (4, 2N-1, 2N-1)}`` for all ``i <= j``, components in
            :data:`COMPONENTS` order. Also stored in ``self.maps``.
        """
        s = self._reshape_slopes(slopes)
        T = s.shape[0]
        if T < 2:
            raise ValueError("At least two frames are required")
        s = s - s.mean(axis=0, keepdims=True)
        if removeTT:
            s = s - s.mean(axis=3, keepdims=True)

        N = self.nLenslet
        P = 2 * N  # padding avoids circular wrap-around for offsets up to N-1
        if chunkSize is None:
            bytesPerFrame = self.nGS * 2 * P * (P // 2 + 1) * 16
            chunkSize = max(1, int(256e6 // bytesPerFrame))
        chunkSize = int(chunkSize)
        if chunkSize < 1:
            raise ValueError("chunkSize must be at least 1")
        nPairs = self.nGS * (self.nGS + 1) // 2
        logger.info(
            "\n-->> Computing SLODAR covariance maps: %d frames, %d sensors (%d maps), "
            "tip-tilt %s, %d FFT batch(es) of up to %d frames <<--",
            T,
            self.nGS,
            nPairs,
            "removed" if removeTT else "kept",
            -(-T // chunkSize),
            chunkSize,
        )
        t0 = time.perf_counter()
        sums = {}
        for start in range(0, T, chunkSize):
            block = s[start : start + chunkSize]
            grid = np.zeros((block.shape[0], self.nGS, 2, N, N))
            grid[:, :, :, self._rows, self._cols] = block
            F = np.fft.rfft2(grid, s=(P, P))
            for i in range(self.nGS):
                for j in range(i, self.nGS):
                    # sum_t conj(F_i^a) F_j^b  ->  sum_p s_i^a(p) s_j^b(p + delta)
                    prod = np.einsum("tayx,tbyx->abyx", np.conj(F[:, i]), F[:, j])
                    sums[(i, j)] = sums.get((i, j), 0) + prod

        valid = self._counts > 0
        maps = {}
        for key, prod in sums.items():
            corr = np.fft.irfft2(prod, s=(P, P))
            corr = np.fft.fftshift(corr, axes=(-2, -1))[..., 1:, 1:]  # offsets -(N-1)..(N-1)
            out = np.zeros((4, *self._mapShape))
            out[:, valid] = corr.reshape(4, *self._mapShape)[:, valid] / (T * self._counts[valid])
            maps[key] = out
        self.maps = maps
        self.nFrames = T
        self.removeTT = bool(removeTT)
        logger.info(
            "\n-->> SLODAR covariance maps computed in %.2f s <<--", time.perf_counter() - t0
        )
        return maps

    def temporal_noise_variance(self, slopes, lag=1, removeTT=False):
        """Per-WFS slope noise variance from the temporal covariance of the slopes.

        White measurement noise is uncorrelated from frame to frame, while turbulence
        evolves little over one frame at kHz-class rates. The noise variance is therefore
        ``C(0) - C(lag)``: the slope variance minus the covariance at ``lag`` frames,
        averaged over the subapertures and both axes of each sensor.

        This is the recommended noise estimate for
        ``build_reconstructor(noiseCovariance=...)``, through
        :func:`slope_noise_covariance`. The spatial estimate of :meth:`fit_profile`
        (zero-offset excess of the auto-covariance) also absorbs any mismatch between the
        sensor and the geometric slope model and can be several times too large on a
        physical Shack-Hartmann. In SPECULA tests this estimator was within 15% of the true
        noise.

        Parameters
        ----------
        slopes : numpy.ndarray
            Time-ordered open-loop or pseudo-open-loop slopes, as for
            :meth:`compute_covariance_maps`, at the loop frame rate (do not decimate).
        lag : int, optional
            Lag in frames (default 1). The estimate is biased high by the turbulence
            decorrelation over ``lag`` frames, which grows with wind speed and frame time.
        removeTT : bool, optional
            Remove the per-frame tip-tilt of each sensor first. The result is corrected for
            the ``1 - 1/nValid`` loss of white-noise variance this causes.

        Returns
        -------
        numpy.ndarray
            ``(nGS,)`` angle-of-arrival noise variances [rad^2], non-negative.

        Notes
        -----
        With an exact interaction matrix, pseudo-open-loop slopes ``s + M c(t - 1)`` cancel
        the applied correction exactly, so their noise stays white and the estimate is
        unbiased. Interaction-matrix or centroid-gain errors leave a loop-filtered
        residual, correlated over a few frames, that biases it. At low flux, calibrate the
        centroid gain first: a thresholded centroid loses gain, which scales the noise and
        the turbulence alike.
        """
        lag = int(lag)
        s = self._reshape_slopes(slopes)
        if lag < 1 or s.shape[0] <= lag + 1:
            raise ValueError("Need lag >= 1 and more than lag + 1 frames")
        s = s - s.mean(axis=0, keepdims=True)
        if removeTT:
            s = s - s.mean(axis=3, keepdims=True)
        c0 = np.mean(s * s, axis=0)
        c1 = np.mean(s[:-lag] * s[lag:], axis=0)
        var = (c0 - c1).mean(axis=(1, 2))
        if removeTT:
            var = var / (1 - 1 / self.nValidSubap)
        var = np.clip(var, 0, None)
        logger.info(
            "\nTemporal slope noise (C(0) - C(%d), %d frames, tip-tilt %s): %s arcsec rms "
            "per sensor",
            lag,
            s.shape[0],
            "removed" if removeTT else "kept",
            np.array2string(np.sqrt(var) * _RAD2ARCSEC, precision=4),
        )
        return var

    # ------------------------------------------------------------------
    # Theoretical response
    def _stationary_covariance(self, rvec, s, L0):
        """Slope covariance of two subapertures whose centres are ``rvec`` apart on a layer.

        Parameters
        ----------
        rvec : numpy.ndarray
            ``(M, 2)`` layer-plane separations [m].
        s : float
            Footprint scale of the subaperture on the layer (cone effect).
        L0 : float
            Outer scale [m].

        Returns
        -------
        numpy.ndarray
            ``(4, M)`` covariances (xx, xy, yx, yy) [rad^2] per unit :math:`r_0^{-5/3}`.
        """
        pts, wx, wy = self._stencil
        P = pts * self.d * s
        rho1 = P[:, 0] + 1j * P[:, 1]
        rho2 = (rvec[:, None, 0] + P[None, :, 0]) + 1j * (rvec[:, None, 1] + P[None, :, 1])
        nP = P.shape[0]
        # Phase covariance at `wavelength` for r0 = 1 m, i.e. per unit r0^{-5/3}.
        C = _covariance_matrix(rho1, rho2.ravel(), 1.0, L0, 1.0).reshape(nP, -1, nP)
        W = np.stack([wx, wy])
        out = np.einsum("au,umv,bv->abm", W, C, W).reshape(4, -1)
        return out * (self.wavelength / (2 * np.pi)) ** 2 / self.d**2

    def _offset_function(self, i, j, h, L0):
        """Stationary covariance of pair ``(i, j)`` at every offset, ``(4, nOffsets)``."""
        s, a = self._layer_geometry(h)
        base = (self.gsDirections[j] - self.gsDirections[i]) * a
        rvec = self._offsets * self.d * s + base
        return self._stationary_covariance(rvec, s, L0)

    def _pair_response(self, i, j, h, L0, removeTT):
        """Response map of pair ``(i, j)`` to a unit layer at range ``h``, ``(4, 2N-1, 2N-1)``."""
        f = self._offset_function(i, j, h, L0)
        if not removeTT:
            out = f.reshape(4, *self._mapShape)
            return np.where(self._counts > 0, out, 0.0)
        n = self.nValidSubap
        C = f[:, self._pairOffsetIndex].reshape(4, n, n)
        C = (
            C
            - C.mean(axis=1, keepdims=True)
            - C.mean(axis=2, keepdims=True)
            + C.mean(axis=(1, 2), keepdims=True)
        )
        flat = np.zeros((4, np.prod(self._mapShape)))
        for c in range(4):
            flat[c] = np.bincount(
                self._pairOffsetIndex, weights=C[c].ravel(), minlength=flat.shape[1]
            )
        cnt = self._counts.ravel()
        flat[:, cnt > 0] /= cnt[cnt > 0]
        return flat.reshape(4, *self._mapShape)

    def response_maps(self, h, L0=None, removeTT=None):
        """Theoretical covariance maps for a thin layer of unit :math:`r_0^{-5/3}` at range ``h``.

        Parameters
        ----------
        h : float
            Layer range [m].
        L0 : float, optional
            Outer scale [m] (default ``self.L0``).
        removeTT : bool, optional
            Include the per-frame tip-tilt removal projection (default: as used for the
            measured maps).

        Returns
        -------
        dict
            ``{(i, j): ndarray (4, 2N-1, 2N-1)}`` for all ``i <= j``.
        """
        L0 = self.L0 if L0 is None else float(L0)
        removeTT = self.removeTT if removeTT is None else bool(removeTT)
        # The key includes the geometry, so changing gsDirections, gsHeight or
        # conjugationAltitude on an existing profiler cannot return stale maps.
        key = (
            round(float(h), 6),
            L0,
            removeTT,
            self.gsDirections.tobytes(),
            self.gsHeight,
            self.conjugationAltitude,
        )
        if key in self._responseCache:
            return self._responseCache[key]
        logger.debug(
            "Computing SLODAR response maps: range %.0f m, L0 %.1f m, tip-tilt %s",
            h,
            L0,
            "removed" if removeTT else "kept",
        )
        out = {}
        byBaseline = {}
        for i in range(self.nGS):
            for j in range(i, self.nGS):
                b = tuple(np.round(self.gsDirections[j] - self.gsDirections[i], 15))
                if b not in byBaseline:
                    byBaseline[b] = self._pair_response(i, j, h, L0, removeTT)
                out[(i, j)] = byBaseline[b]
        self._responseCache[key] = out
        return out

    def _noise_response(self):
        """Auto-covariance map of unit white slope noise, ``(2N-1, 2N-1)`` (x and y alike)."""
        N = self.nLenslet
        out = np.zeros(self._mapShape)
        out[N - 1, N - 1] = 1.0
        if self.removeTT:
            # P I P = I - 11^T/n; averaged per offset this is delta(0) - 1/n.
            out = out - (self._counts > 0) / self.nValidSubap
        return out

    # ------------------------------------------------------------------
    # Fitting
    def _selection(self, pairs, components, useAuto, minOverlap):
        """Which map entries enter the fit, and their weights."""
        comp_idx = [COMPONENTS.index(c) for c in components if c in COMPONENTS]
        longTrans = [c for c in components if c in ("ll", "tt")]
        unknown = set(components) - set(COMPONENTS) - {"ll", "tt"}
        if unknown:
            raise ValueError(f"Unknown components {unknown}")
        minCount = max(1, int(np.ceil(minOverlap * self.nValidSubap)))
        mask = self._counts >= minCount
        keys = list(pairs)
        if useAuto:
            keys += [(g, g) for g in range(self.nGS)]
        return keys, comp_idx, longTrans, mask

    def _project(self, maps, keys, comp_idx, longTrans, mask, autoZero):
        """Flatten the selected parts of a map dictionary into one vector."""
        N = self.nLenslet
        parts = []
        # Auto-covariances have no baseline, so "ll"/"tt" do not apply to them; they always
        # contribute their xx and yy components (plus any other selected Cartesian ones),
        # which is what carries the total turbulence and the noise.
        autoIdx = sorted(set(comp_idx) | {0, 3})
        for key in keys:
            m = maps[key]
            sel = mask.copy()
            isAuto = key[0] == key[1]
            if isAuto and not autoZero:
                sel[N - 1, N - 1] = False
            for c in autoIdx if isAuto else comp_idx:
                parts.append(m[c][sel])
            if longTrans and key[0] != key[1]:
                b = self.gsDirections[key[1]] - self.gsDirections[key[0]]
                c_, s_ = b / np.hypot(*b)
                for lt in longTrans:
                    if lt == "ll":
                        v = c_ * c_ * m[0] + c_ * s_ * (m[1] + m[2]) + s_ * s_ * m[3]
                    else:
                        v = s_ * s_ * m[0] - c_ * s_ * (m[1] + m[2]) + c_ * c_ * m[3]
                    parts.append(v[sel])
        return np.concatenate(parts)

    def _weights(self, keys, comp_idx, longTrans, mask, autoZero):
        """Least-squares weights ``sqrt(N_delta)`` matching :meth:`_project`."""
        wmap = {k: np.broadcast_to(np.sqrt(self._counts), (4, *self._mapShape)) for k in keys}
        return self._project(wmap, keys, comp_idx, longTrans, mask, autoZero)

    def fit_profile(
        self,
        altitudes=None,
        pairs=None,
        components=("xx", "yy"),
        useAuto=True,
        fitNoise=True,
        L0=None,
        fitL0=False,
        L0Grid=None,
        minOverlap=0.1,
        regularization=0.0,
    ):
        """Fit the measured covariance maps with thin-layer response functions.

        Solves the weighted non-negative least-squares problem

        .. math::
            \\min_{w \\ge 0} \\left\\| W \\left(\\sum_k w_k\\, R(h_k) + \\sum_g n_g\\, R_n^{(g)}
            - C_{\\rm meas}\\right) \\right\\|^2,

        where :math:`R(h_k)` are the response maps of :meth:`response_maps`, :math:`w_k`
        the layer strengths (:math:`r_0^{-5/3}`), :math:`n_g` optional per-WFS noise
        variances (auto-covariance only) and :math:`W` weights each map entry by
        :math:`\\sqrt{N_\\delta}`.

        Parameters
        ----------
        altitudes : array_like, optional
            Layer ranges [m] (default :meth:`default_altitudes`).
        pairs : list of tuple, optional
            Cross pairs to use (default: all).
        components : sequence of str, optional
            Map components to fit: any of ``"xx", "xy", "yx", "yy"`` and the longitudinal /
            transverse combinations ``"ll", "tt"`` (cross maps only). Default
            ``("xx", "yy")``.
        useAuto : bool, optional
            Also fit the auto-covariance maps (default True). They carry no altitude
            information for NGS but constrain the total turbulence and the noise. Their xx
            and yy components are always used, whichever ``components`` are selected for
            the cross maps (``"ll"``/``"tt"`` are not defined without a baseline).
        fitNoise : bool, optional
            Fit per-WFS noise variances on the zero-offset point of the auto-covariance
            (requires ``useAuto``); otherwise that point, which noise biases, is excluded
            (default True). See :meth:`slodarResult.noise_covariance` for why
            :meth:`temporal_noise_variance` is the better noise estimate for the
            reconstructor.
        L0 : float, optional
            Outer scale to use (default ``self.L0``); ignored if ``fitL0``.
        fitL0 : bool, optional
            Estimate L0 by minimising the fit residual over ``L0Grid`` (default False).
        L0Grid : array_like, optional
            Outer scales tried when ``fitL0`` (default 12 values from 5 to 200 m).
        minOverlap : float, optional
            Offsets with fewer than ``minOverlap * nValidSubap`` subaperture pairs are not
            fitted (default 0.1).
        regularization : float, optional
            Ridge penalty on the normalised layer strengths (default 0).

        Returns
        -------
        slodarResult
        """
        if self.maps is None:
            raise RuntimeError("Call compute_covariance_maps first")
        altitudes = self.default_altitudes() if altitudes is None else np.asarray(altitudes, float)
        pairs = self.pairs if pairs is None else [tuple(p) for p in pairs]
        keys, comp_idx, longTrans, mask = self._selection(pairs, components, useAuto, minOverlap)
        autoZero = bool(useAuto and fitNoise)
        y = self._project(self.maps, keys, comp_idx, longTrans, mask, autoZero)
        wts = self._weights(keys, comp_idx, longTrans, mask, autoZero)
        logger.info(
            "\n-->> Fitting SLODAR profile: %d altitudes (%.2f-%.2f km), %d cross pair(s), "
            "components %s, auto-covariances %s, noise %s, L0 %s; %d data points from "
            "%d frames <<--",
            len(altitudes),
            altitudes.min() / 1e3 if len(altitudes) else 0.0,
            altitudes.max() / 1e3 if len(altitudes) else 0.0,
            len(pairs),
            ",".join(components),
            "used" if useAuto else "not used",
            "fitted" if autoZero else "not fitted",
            "fitted on a grid" if fitL0 else f"fixed at {self.L0 if L0 is None else L0:g} m",
            y.size,
            self.nFrames,
        )
        t0 = time.perf_counter()

        noiseCols = []
        if autoZero:
            nr = self._noise_response()
            zero = np.zeros((4, *self._mapShape))
            for g in range(self.nGS):
                m = dict.fromkeys(keys, zero)
                m[(g, g)] = np.stack([nr, 0 * nr, 0 * nr, nr])
                noiseCols.append(self._project(m, keys, comp_idx, longTrans, mask, autoZero))
            if any(not np.any(col) for col in noiseCols):
                # Cannot happen with the auto maps always contributing xx/yy, but a silent
                # all-zero noise column would report zero noise and build an unregularised
                # reconstructor downstream, so fail loudly instead.
                raise RuntimeError("Noise response is empty for the selected components")

        def solve(L0value):
            cols = []
            for h in altitudes:
                resp = self.response_maps(h, L0value)
                cols.append(self._project(resp, keys, comp_idx, longTrans, mask, autoZero))
            A = np.stack(cols + noiseCols, axis=1)
            Aw = A * wts[:, None]
            yw = y * wts
            norm = np.linalg.norm(Aw, axis=0)
            norm[norm == 0] = 1.0
            An = Aw / norm
            if regularization > 0:
                nL = len(altitudes)
                R = np.zeros((nL, An.shape[1]))
                R[:, :nL] = np.sqrt(regularization) * np.eye(nL)
                An = np.vstack([An, R])
                yw = np.concatenate([yw, np.zeros(nL)])
            x = _nnls(An, yw)
            x = x / norm
            model = A @ x
            cost = np.linalg.norm((model - y) * wts) / np.linalg.norm(y * wts)
            return x, model, cost

        L0g = np.zeros(0)
        costs = np.zeros(0)
        if fitL0:
            L0g = np.geomspace(5, 200, 12) if L0Grid is None else np.asarray(L0Grid, float)
            fits = [solve(v) for v in L0g]
            costs = np.array([f[2] for f in fits])
            for v, c_ in zip(L0g, costs):
                logger.debug("SLODAR L0 grid: L0 %7.1f m, relative residual %.5f", v, c_)
            best = int(np.argmin(costs))
            L0used = float(L0g[best])
            x, model, cost = fits[best]
            logger.info("\nSLODAR L0 grid search: best L0 = %.1f m", L0used)
            if np.isclose(L0used, L0g.min()) or np.isclose(L0used, L0g.max()):
                # The minimum is on the edge of the grid, so L0 is not constrained by these
                # data (typical of short records); the profile built on it can be poor.
                logger.warning(
                    "SLODAR L0 fit is at the edge of L0Grid (%.1f m, grid %.1f-%.1f m): "
                    "L0 is not constrained by these data; use longer telemetry or widen "
                    "L0Grid, and treat this profile with caution",
                    L0used,
                    L0g.min(),
                    L0g.max(),
                )
        else:
            L0used = self.L0 if L0 is None else float(L0)
            x, model, cost = solve(L0used)

        nL = len(altitudes)
        w = x[:nL]
        noise = x[nL:] if autoZero else np.zeros(0)
        res = self._make_result(altitudes, w, L0used, noise, cost, L0g, costs, y, model)
        logger.info(
            "\n-->> SLODAR profile fitted in %.2f s: r0 %.4f m (%.4f m at zenith), L0 %.1f m, "
            "%d non-zero layer(s), relative residual %.4f <<--",
            time.perf_counter() - t0,
            res.r0,
            res.r0_zenith,
            res.L0,
            int(np.count_nonzero(w)),
            res.residualRms,
        )
        if autoZero:
            logger.info(
                "\nSLODAR zero-offset noise term: %s arcsec rms per sensor (for the "
                "reconstructor, prefer temporal_noise_variance)",
                np.array2string(np.sqrt(noise) * _RAD2ARCSEC, precision=4),
            )
        return res

    def _make_result(self, altitudes, w, L0, noise, cost, L0g, costs, y, model):
        k = 2 * np.pi / self.wavelength
        total = w.sum()
        r0 = total ** (-3 / 5) if total > 0 else np.inf
        frac = w / total if total > 0 else np.zeros_like(w)
        return slodarResult(
            altitude=np.asarray(altitudes, float),
            altitude_zenith=np.asarray(altitudes, float) / self.airmass,
            r0m53=w,
            cn2dh=w / (_R0_CN2 * k**2),
            fractionalCn2=frac,
            r0=float(r0),
            r0_zenith=float(r0 * self.airmass ** (3 / 5)),
            L0=float(L0),
            wavelength=self.wavelength,
            zenithAngleInDeg=self.zenithAngleInDeg,
            noiseVariance=noise,
            residualRms=float(cost),
            L0Grid=L0g,
            L0Cost=costs,
            measured=y,
            model=model,
        )

    def profile(self, slopes, removeTT=False, **fitKwargs):
        """Compute the covariance maps and fit the profile in one call.

        Parameters
        ----------
        slopes : numpy.ndarray
            Slope time series, see :meth:`compute_covariance_maps`.
        removeTT : bool, optional
            Remove per-frame tip-tilt (default False).
        **fitKwargs
            Passed to :meth:`fit_profile`.

        Returns
        -------
        slodarResult
        """
        self.compute_covariance_maps(slopes, removeTT=removeTT)
        return self.fit_profile(**fitKwargs)

    # ------------------------------------------------------------------
    # Simulation helpers
    def slope_covariance(self, altitudes, r0m53, L0=None):
        """Full theoretical slope covariance of all sensors for a layered profile.

        Parameters
        ----------
        altitudes : array_like
            Layer ranges [m].
        r0m53 : array_like
            Layer strengths :math:`r_0^{-5/3}` at ``wavelength``.
        L0 : float, optional
            Outer scale (default ``self.L0``).

        Returns
        -------
        numpy.ndarray
            ``(nGS*2*nValid, nGS*2*nValid)`` covariance [rad^2], rows and columns in the
            ``(WFS, component, subaperture)`` order of the ``"simu"`` layout.
        """
        L0 = self.L0 if L0 is None else float(L0)
        n = self.nValidSubap
        G = self.nGS
        C = np.zeros((G, 2, n, G, 2, n))
        for h, w in zip(altitudes, r0m53):
            if w == 0:
                continue
            for i in range(G):
                for j in range(G):
                    f = self._offset_function(i, j, h, L0)
                    blk = f[:, self._pairOffsetIndex].reshape(2, 2, n, n)
                    C[i, :, :, j, :, :] += w * blk.transpose(0, 2, 1, 3)
        C = C.reshape(G * 2 * n, G * 2 * n)
        return 0.5 * (C + C.T)

    def draw_slopes(self, altitudes, r0m53, nFrames, noiseRms=0.0, L0=None, rng=None):
        """Draw independent Gaussian slope frames from the theoretical covariance.

        Useful to validate the estimator: the maps of these slopes converge to
        :meth:`response_maps` by construction, so any bias comes from the estimator and the
        finite number of frames, not from the turbulence model.

        Parameters
        ----------
        altitudes, r0m53 : array_like
            Layered profile (ranges [m] and :math:`r_0^{-5/3}`).
        nFrames : int
            Number of frames.
        noiseRms : float, optional
            White measurement noise per slope [rad] (default 0).
        L0 : float, optional
            Outer scale (default ``self.L0``).
        rng : numpy.random.Generator or int, optional
            Random generator or seed.

        Returns
        -------
        numpy.ndarray
            ``(nFrames, nGS*2*nValid)`` slopes in the configured layout and units (i.e.
            divided by ``slopeScale``).
        """
        rng = np.random.default_rng(rng)
        logger.info(
            "\nDrawing %d synthetic slope frames from the theoretical covariance (%d layers, "
            "noise %.4g rad rms)",
            nFrames,
            len(altitudes),
            noiseRms,
        )
        C = self.slope_covariance(altitudes, r0m53, L0)
        evals, evecs = np.linalg.eigh(C)
        evals = np.clip(evals, 0, None)
        Lmat = evecs * np.sqrt(evals)
        z = rng.standard_normal((nFrames, C.shape[0]))
        s = z @ Lmat.T
        if noiseRms > 0:
            s += noiseRms * rng.standard_normal(s.shape)
        s = s.reshape(nFrames, self.nGS, 2, self.nValidSubap)
        return self._format_slopes(s) / self.slopeScale

    @staticmethod
    def cn2dh_to_r0m53(cn2dh, wavelength=5e-7):
        """Convert :math:`C_n^2 dh` [m^1/3] to :math:`r_0^{-5/3}` at ``wavelength``."""
        k = 2 * np.pi / wavelength
        return _R0_CN2 * k**2 * np.asarray(cn2dh, float)

    @staticmethod
    def profile_to_r0m53(r0, fractions):
        """Split a total ``r0`` into per-layer :math:`r_0^{-5/3}` with the given fractions."""
        f = np.asarray(fractions, float)
        return r0 ** (-5 / 3) * f / f.sum()

    # ------------------------------------------------------------------
    # Visualisation
    def plot_covariance_maps(self, pair, components=("xx", "yy"), model=None):
        """Display measured (and optionally model) covariance maps of one pair.

        Parameters
        ----------
        pair : tuple of int
            ``(i, j)`` with ``i <= j``.
        components : sequence of str, optional
            Components to show (default ``("xx", "yy")``).
        model : dict, optional
            Model maps with the layout of ``self.maps`` to show below the data.

        Returns
        -------
        matplotlib.figure.Figure
        """
        plt = pyplot()
        rows = 1 if model is None else 2
        fig, axes = plt.subplots(rows, len(components), figsize=(3.2 * len(components), 3 * rows))
        axes = np.atleast_2d(axes).reshape(rows, len(components))
        N = self.nLenslet
        ext = [-(N - 1) - 0.5, N - 0.5, -(N - 1) - 0.5, N - 0.5]
        for c, comp in enumerate(components):
            k = COMPONENTS.index(comp)
            axes[0, c].imshow(self.maps[pair][k], origin="lower", extent=ext)
            axes[0, c].set_title(f"{pair} {comp} measured")
            if model is not None:
                axes[1, c].imshow(model[pair][k], origin="lower", extent=ext)
                axes[1, c].set_title(f"{pair} {comp} model")
        for ax in axes.ravel():
            ax.set_xlabel(r"$\delta_x$ [subap]")
            ax.set_ylabel(r"$\delta_y$ [subap]")
        fig.tight_layout()
        return fig
