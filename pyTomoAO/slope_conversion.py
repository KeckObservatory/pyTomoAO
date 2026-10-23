"""
Conversion of external slope vectors into the convention of pyTomoAO's gradient operator.

The model-based reconstructor (``RecStatSA``, see
:meth:`~pyTomoAO.reconstructor.tomographicReconstructor.build_reconstructor`) acts on slopes in
the units and order of the tomographic gradient operator ``Gamma``
(:func:`pyTomoAO.tomographyUtilsCPU._sparseGradientMatrixAmplitudeWeighted`):

- **Units.** ``Gamma`` applied to a phase map returns the trapezoid-weighted phase *difference*
  across each subaperture, i.e. ``d * slope`` with ``d`` the subaperture size. Feeding the
  reconstructor ``d * angle-of-arrival`` [m] therefore returns OPD in metres on the
  reconstruction grid. The MMSE estimator is homogeneous (its regularisation is scaled to the
  mean slope variance), so this holds whatever wavelength the turbulence model uses.
- **Order.** Per wavefront sensor, all X slopes then all Y slopes; valid subapertures are
  enumerated column-major (``"F"`` order over ``validLLMap``). X is the derivative along the
  columns of ``validLLMap`` and Y along its rows. Sensors follow the LGS order of
  :attr:`~pyTomoAO.lgsAsterismParametersClass.lgsAsterismParameters.directionVectorLGS`.

:func:`slope_conversion_matrix` builds the sparse matrix ``P`` taking slopes in another
convention (order, units, signs) to that one, so that ``RecStatSA @ P`` reconstructs OPD [m]
directly from the external slopes. :func:`specula_slope_scale` gives the unit conversion for
SPECULA's Shack-Hartmann slopes.
"""

import numpy as np
from scipy.sparse import csr_matrix

RAD2ARCSEC = 180 * 3600 / np.pi

#: Accepted values of ``slopesOrder``; same meaning as in :class:`pyTomoAO.slodar.slodarProfiler`.
SLOPES_ORDERS = ("simu", "inverted", "keck")


def _subap_permutation(validLLMap, subapOrder):
    """Index, in the input subaperture order, of each subaperture in pyTomoAO's ``"F"`` order."""
    valid = np.asarray(validLLMap, dtype=bool)
    if subapOrder == "F":
        return np.arange(int(valid.sum()))
    if subapOrder != "C":
        raise ValueError("subapOrder must be 'F' or 'C'")
    rank = np.full(valid.shape, -1, dtype=int)
    rank[valid] = np.arange(int(valid.sum()))  # boolean indexing enumerates in C order
    cols, rows = np.nonzero(valid.T)  # F order: column-major
    return rank[rows, cols]


def slope_conversion_matrix(
    validLLMap,
    nWFS,
    D,
    subapOrder="F",
    slopesOrder="simu",
    slopeScale=1.0,
    xSign=1.0,
    ySign=1.0,
):
    """Sparse matrix taking external slopes to pyTomoAO's gradient-operator convention.

    ``P @ s`` returns, for an input slope vector ``s`` covering ``nWFS`` sensors (sensor blocks
    concatenated in pyTomoAO's LGS order), the slopes in the order of the tomographic gradient
    operator and in units of ``d * angle`` [m], ready for ``RecStatSA``.

    Parameters
    ----------
    validLLMap : array_like of bool
        Valid-subaperture map, ``(nLenslet, nLenslet)``, the same for every sensor.
    nWFS : int
        Number of sensors in the slope vector.
    D : float
        Telescope diameter [m]; the subaperture size is ``d = D / nLenslet``.
    subapOrder : {"F", "C"}, optional
        Enumeration of the valid subapertures in the input: ``"F"`` column-major (pyTomoAO's
        own), ``"C"`` row-major (e.g. SPECULA). Default ``"F"``.
    slopesOrder : {"simu", "inverted", "keck"}, optional
        Layout of each sensor's block in the input: ``"simu"`` all X then all Y,
        ``"inverted"`` all Y then all X, ``"keck"`` interleaved ``[x0, y0, x1, y1, ...]``.
        Default ``"simu"``.
    slopeScale : float, optional
        Factor converting the input slopes to angles of arrival in radians (e.g.
        :func:`specula_slope_scale`). Default 1 (input already in radians).
    xSign, ySign : float, optional
        Sign (or extra factor) applied to the X and Y slopes, for inputs whose axes point the
        other way. Default +1.

    Returns
    -------
    scipy.sparse.csr_matrix
        ``(nWFS * 2 * nValid, nWFS * 2 * nValid)``, one non-zero per row.

    Notes
    -----
    pyTomoAO's X slope is the derivative along the columns of ``validLLMap`` and Y along its
    rows. An input whose X runs along the rows corresponds to ``slopesOrder="inverted"`` (with
    the same subaperture enumeration), not to a sign change.

    Examples
    --------
    SPECULA ``IdealDerivativeSensor`` or Shack-Hartmann slopes of six LGS, concatenated in
    the order of the asterism:

    >>> P = slope_conversion_matrix(
    ...     validLLMap, 6, 10.0, subapOrder="C", slopeScale=specula_slope_scale(4.0, 8)
    ... )  # doctest: +SKIP
    >>> opd = rec.RecStatSA @ (P @ slopes)  # OPD [m] on rec.gridMask  # doctest: +SKIP
    """
    valid = np.asarray(validLLMap, dtype=bool)
    if valid.ndim != 2 or valid.shape[0] != valid.shape[1]:
        raise ValueError("validLLMap must be a square 2D map")
    if int(nWFS) < 1:
        raise ValueError("nWFS must be at least 1")
    if D <= 0:
        raise ValueError("D must be positive")
    if slopesOrder not in SLOPES_ORDERS:
        raise ValueError(f"slopesOrder must be one of {SLOPES_ORDERS}")
    nWFS = int(nWFS)
    n = int(valid.sum())
    d = D / valid.shape[0]
    perm = _subap_permutation(valid, subapOrder)

    # Column (input position) of the X and Y slope of each subaperture, in F order.
    if slopesOrder == "simu":
        colX, colY = perm, n + perm
    elif slopesOrder == "inverted":
        colX, colY = n + perm, perm
    else:  # keck: interleaved
        colX, colY = 2 * perm, 2 * perm + 1

    rows, cols, vals = [], [], []
    for k in range(nWFS):
        off = 2 * n * k
        rows.append(off + np.arange(2 * n))
        cols.append(off + np.concatenate([colX, colY]))
        vals.append(np.concatenate([np.full(n, xSign), np.full(n, ySign)]))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    vals = np.concatenate(vals) * (d * slopeScale)
    size = 2 * n * nWFS
    return csr_matrix((vals, (rows, cols)), shape=(size, size))


def specula_slope_scale(fovArcsec, nPxPerSubap):
    """Radians per SPECULA slope unit for its Shack-Hartmann-like sensors.

    SPECULA normalises slopes to the subaperture field of view: its ``IdealDerivativeSensor``
    returns ``grad[m/m] * RAD2ARCSEC / (fov / 2) * (n - 1) / n`` (the last factor matches the
    pixel spacing of its SH slope computer).

    Parameters
    ----------
    fovArcsec : float
        Sensor field of view [arcsec], the ``fov`` parameter of the SPECULA sensor.
    nPxPerSubap : int
        Pupil pixels per subaperture (``pixel_pupil / nLenslet``).

    Returns
    -------
    float
        Multiply SPECULA slopes by this to get angles of arrival in radians.
    """
    if fovArcsec <= 0:
        raise ValueError("fovArcsec must be positive")
    if nPxPerSubap < 2:
        raise ValueError("nPxPerSubap must be at least 2")
    return (fovArcsec / 2) / RAD2ARCSEC * nPxPerSubap / (nPxPerSubap - 1)
