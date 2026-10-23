"""Tests for pyTomoAO.slope_conversion and the reconstructor's slope/grid helpers."""

import numpy as np
import pytest

from pyTomoAO import slope_conversion_matrix, specula_slope_scale, tomographicReconstructor
from pyTomoAO.tomographyUtilsCPU import _sparseGradientMatrixAmplitudeWeighted

# Small, irregular map: a hole and clipped corners exercise the C/F permutation.
VALID = np.array(
    [
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 0, 0],
    ],
    dtype=bool,
)
D = 2.5  # d = 0.5 m


def _gamma():
    G, gm = _sparseGradientMatrixAmplitudeWeighted(VALID, None, 2)
    return G.toarray(), gm


def _external(sF, subapOrder, slopesOrder):
    """Lay one sensor's F-ordered [X..., Y...] slopes out in an external convention."""
    n = VALID.sum()
    x, y = sF[:n], sF[n:]
    if subapOrder == "C":
        cols, rows = np.nonzero(VALID.T)  # F-order positions
        rank = np.full(VALID.shape, -1)
        rank[VALID] = np.arange(n)  # C-order rank of each position
        idx = rank[rows, cols]  # C-order index of each F-ordered subaperture
        xc, yc = np.empty(n), np.empty(n)
        xc[idx], yc[idx] = x, y
        x, y = xc, yc
    if slopesOrder == "simu":
        return np.concatenate([x, y])
    if slopesOrder == "inverted":
        return np.concatenate([y, x])
    out = np.empty(2 * n)
    out[0::2], out[1::2] = x, y
    return out


def test_identity_for_pytomoao_convention():
    P = slope_conversion_matrix(VALID, 3, D)
    n = 2 * VALID.sum() * 3
    np.testing.assert_allclose(P.toarray(), 0.5 * np.eye(n))


@pytest.mark.parametrize("subapOrder", ["F", "C"])
@pytest.mark.parametrize("slopesOrder", ["simu", "inverted", "keck"])
def test_round_trip_through_gradient_operator(subapOrder, slopesOrder):
    """Angles laid out in any supported convention map back to Gamma @ phase."""
    G, gm = _gamma()
    rng = np.random.default_rng(0)
    nWFS, d, scale = 2, D / VALID.shape[0], 3.7
    target, ext = [], []
    for _ in range(nWFS):
        sG = G @ rng.standard_normal(gm.sum())  # pyTomoAO units: d * angle
        target.append(sG)
        ext.append(_external(sG / d / scale, subapOrder, slopesOrder))  # angle / scale
    P = slope_conversion_matrix(
        VALID, nWFS, D, subapOrder=subapOrder, slopesOrder=slopesOrder, slopeScale=scale
    )
    np.testing.assert_allclose(P @ np.concatenate(ext), np.concatenate(target), atol=1e-12)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_x_runs_along_columns(axis):
    """A phase ramp along the map columns is a pure X slope of d * gradient, and vice versa.

    Pins the documented convention against the gradient operator itself: an external sensor
    differentiating along columns (e.g. ``np.gradient(phase, axis=1)``) is X.
    """
    G, gm = _gamma()
    nMap = gm.shape[0]
    h = D / VALID.shape[0] / 2  # grid spacing
    rr, cc = np.mgrid[0:nMap, 0:nMap] * h
    a = 1.3e-6  # rad of tilt
    phase = a * (cc if axis == "x" else rr)
    sG = G @ phase[gm]
    n = VALID.sum()
    d = 2 * h
    expected_x = np.full(n, a * d if axis == "x" else 0.0)
    expected_y = np.full(n, a * d if axis == "y" else 0.0)
    np.testing.assert_allclose(sG, np.concatenate([expected_x, expected_y]), atol=1e-18)
    # And the converter maps an angle of `a` on that axis to the same thing.
    ang = np.concatenate([expected_x, expected_y]) / d
    P = slope_conversion_matrix(VALID, 1, D, subapOrder="C")
    np.testing.assert_allclose(P @ ang, sG, atol=1e-18)


def test_signs_and_scale():
    n = VALID.sum()
    P = slope_conversion_matrix(VALID, 1, D, slopeScale=2.0, xSign=-1, ySign=1).toarray()
    np.testing.assert_allclose(np.diag(P)[:n], -1.0)
    np.testing.assert_allclose(np.diag(P)[n:], 1.0)


def test_one_nonzero_per_row():
    P = slope_conversion_matrix(VALID, 4, D, subapOrder="C", slopesOrder="keck")
    assert (np.diff(P.indptr) == 1).all()
    assert sorted(P.indices) == list(range(P.shape[1]))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"subapOrder": "A"}, "subapOrder"),
        ({"slopesOrder": "xy"}, "slopesOrder"),
    ],
)
def test_rejects_bad_conventions(kwargs, match):
    with pytest.raises(ValueError, match=match):
        slope_conversion_matrix(VALID, 1, D, **kwargs)


def test_rejects_bad_geometry():
    with pytest.raises(ValueError, match="square"):
        slope_conversion_matrix(np.ones((3, 4), bool), 1, D)
    with pytest.raises(ValueError, match="nWFS"):
        slope_conversion_matrix(VALID, 0, D)
    with pytest.raises(ValueError, match="D must"):
        slope_conversion_matrix(VALID, 1, 0.0)


def test_specula_slope_scale():
    # fov = 4", 8 px per subaperture: 2" in radians, times 8/7
    assert specula_slope_scale(4.0, 8) == pytest.approx(2 / 206264.80624709636 * 8 / 7)
    with pytest.raises(ValueError):
        specula_slope_scale(0.0, 8)
    with pytest.raises(ValueError):
        specula_slope_scale(4.0, 1)


# --- reconstructor helpers --------------------------------------------------------------
@pytest.fixture(scope="module")
def revolt_rec(revolt_config):
    rec = tomographicReconstructor(revolt_config, force_cpu=True)
    rec.build_reconstructor()
    return rec


def test_reconstructor_slope_conversion_matrix_shape(revolt_rec):
    P = revolt_rec.slope_conversion_matrix(subapOrder="C")
    n = 2 * revolt_rec.lgsWfsParams.nValidSubap * revolt_rec.lgsAsterismParams.nLGS
    assert P.shape == (n, n)


def test_grid_coordinates_span_the_padded_support(revolt_rec):
    x, y = revolt_rec.grid_coordinates()
    gm = np.asarray(revolt_rec.gridMask, bool)
    assert x.shape == y.shape == (gm.sum(),)
    step = revolt_rec.lgsWfsParams.DSupport / (gm.shape[0] - 1)
    np.testing.assert_allclose(np.diff(np.unique(np.round(x / step, 9))), 1.0)
    # Row-major over the mask: x varies fastest along a row of the mask.
    r, c = np.nonzero(gm)
    np.testing.assert_allclose(x, -revolt_rec.lgsWfsParams.DSupport / 2 + c * step)
    np.testing.assert_allclose(y, -revolt_rec.lgsWfsParams.DSupport / 2 + r * step)
