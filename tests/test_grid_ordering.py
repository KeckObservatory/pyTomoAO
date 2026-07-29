"""Tests that grid points are indexed consistently end to end (#104).

The gradient operator used to index the reconstruction grid in Fortran order (a MATLAB
port artefact) while masking it with a C-order boolean, and the covariance kernels matched
that. `reconstruct_wavefront` then scattered the result in C order, so the wavefront it
returned was **transposed**; only `visualize_reconstruction` compensated, by displaying
`reconstructed_wavefront.T`.

Nothing in the old suite could catch this: the mean over a symmetric pupil is invariant
under transposition, and a round trip through the same operator is self-consistent whichever
convention it uses. The tests here are deliberately orientation-sensitive.
"""

import logging

import numpy as np
import pytest

from pyTomoAO import reconstructor as reconstructor_module
from pyTomoAO import tomographyUtilsCPU as cpu

logger = logging.getLogger(__name__)


def _asymmetric_lenslet_map(n=10):
    """A valid-lenslet map that is deliberately not symmetric under transpose.

    Every configuration shipped with the package has a symmetric pupil, for which the two
    index conventions coincide -- so a symmetric mask cannot detect the bug.
    """
    yy, xx = np.mgrid[0:n, 0:n]
    cy, cx = (n - 1) / 2 + 1.0, (n - 1) / 2 - 0.5
    valid = ((xx - cx) / 4.5) ** 2 + ((yy - cy) / 3.0) ** 2 < 1.0
    assert not np.array_equal(valid, valid.T), "test map must be asymmetric"
    return valid


class TestGradientOperatorOrdering:
    """Gamma's columns must index the reconstruction grid in C order."""

    def test_constant_phase_has_zero_gradient(self):
        valid = _asymmetric_lenslet_map()
        Gamma, grid_mask = cpu._sparseGradientMatrixAmplitudeWeighted(valid, None, 2)
        slopes = Gamma @ np.ones(int(grid_mask.sum()))
        assert np.allclose(slopes, 0.0, atol=1e-12), "a flat wavefront produced slopes"

    @pytest.mark.parametrize("axis", ["x", "y"])
    def test_a_ramp_produces_gradient_on_the_matching_axis(self, axis):
        """The orientation check.

        Gamma's rows are [all x-slopes, all y-slopes]. Feeding it a phase that ramps along
        one axis must put the signal in the matching half of the slope vector; under the
        old Fortran/C mix-up an x-ramp came out as a y-gradient.
        """
        valid = _asymmetric_lenslet_map()
        Gamma, grid_mask = cpu._sparseGradientMatrixAmplitudeWeighted(valid, None, 2)

        n = grid_mask.shape[0]
        yy, xx = np.mgrid[0:n, 0:n]
        ramp = (xx if axis == "x" else yy).astype(float)

        slopes = Gamma @ ramp[grid_mask]
        half = slopes.size // 2
        sx, sy = slopes[:half], slopes[half:]

        if axis == "x":
            assert np.abs(sx).max() > 1e-6, "x-ramp produced no x-gradient"
            assert np.abs(sy).max() < 1e-9 * max(np.abs(sx).max(), 1.0), (
                "x-ramp leaked into the y-gradient: grid indexing is transposed"
            )
        else:
            assert np.abs(sy).max() > 1e-6, "y-ramp produced no y-gradient"
            assert np.abs(sx).max() < 1e-9 * max(np.abs(sy).max(), 1.0), (
                "y-ramp leaked into the x-gradient: grid indexing is transposed"
            )


class TestReconstructionOrientation:
    """reconstruct_wavefront must return the wavefront, not its transpose."""

    @staticmethod
    def _ramp_axes(wavefront):
        """Spread of the row- and column-averaged profiles.

        Rows and columns lying entirely outside the pupil are dropped rather than averaged,
        which would be a mean of an all-NaN slice.
        """
        valid = ~np.isnan(wavefront)
        columns = wavefront[:, valid.any(axis=0)]
        rows = wavefront[valid.any(axis=1), :]
        along_x = np.nanstd(np.nanmean(columns, axis=0))
        along_y = np.nanstd(np.nanmean(rows, axis=1))
        return along_x, along_y

    @pytest.fixture(scope="class")
    def rec(self, kapa_config):
        r = reconstructor_module.tomographicReconstructor(kapa_config)
        r.build_reconstructor()
        return r

    def test_pure_x_gradient_reconstructs_a_ramp_along_x(self, rec):
        n = int(rec.lgsWfsParams.nValidSubap)
        slopes = np.zeros(2 * n)
        slopes[:n] = 1.0
        wavefront = rec.reconstruct_wavefront(np.tile(slopes, rec.nLGS))

        along_x, along_y = self._ramp_axes(wavefront)
        assert along_x > 100 * along_y, (
            f"x-gradient gave a ramp along y (x {along_x:.3e}, y {along_y:.3e}): "
            "the reconstructed wavefront is transposed"
        )

    def test_pure_y_gradient_reconstructs_a_ramp_along_y(self, rec):
        n = int(rec.lgsWfsParams.nValidSubap)
        slopes = np.zeros(2 * n)
        slopes[n:] = 1.0
        wavefront = rec.reconstruct_wavefront(np.tile(slopes, rec.nLGS))

        along_x, along_y = self._ramp_axes(wavefront)
        assert along_y > 100 * along_x, (
            f"y-gradient gave a ramp along x (x {along_x:.3e}, y {along_y:.3e}): "
            "the reconstructed wavefront is transposed"
        )
