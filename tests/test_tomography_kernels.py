"""Tests for the covariance kernels shared by the CPU and GPU backends.

Covers the WFS rotation units (#92) and the zero-separation variance term (#90). Both were
defects that no configuration in the repository exercised: every shipped config sets
`wfsLensletsRotation` to zero, and the zero-separation case only misbehaves when floating
point puts a mathematically-zero distance a few ULPs off zero.
"""

import importlib
import logging

import numpy as np
import pytest
from scipy.special import gamma as gamma_fn

from pyTomoAO import tomographyUtilsCPU as cpu

logger = logging.getLogger(__name__)

reconstructor_module = importlib.import_module("pyTomoAO.tomographicReconstructor")

R0 = 0.15
L0 = 25.0
SAMPLING = 12
DIAMETER = 8.0


def analytic_var_term(r0, outer_scale, fractional_r0=1.0):
    """Von Karman phase variance at zero separation, i.e. the rho -> 0 limit."""
    base = (24 * gamma_fn(6 / 5) / 5) ** (5 / 6)
    return (
        (base * gamma_fn(11 / 6) * gamma_fn(5 / 6) / (2 * np.pi ** (8 / 3)))
        * (outer_scale / r0) ** (5 / 3)
        * fractional_r0
    )


class TestRotationUnits:
    """`wfsLensletsRotation` is documented and stored in radians (#92)."""

    def test_quarter_turn_maps_x_to_minus_y(self):
        x0, y0 = cpu._create_guide_star_grid(SAMPLING, DIAMETER, 0.0, 0.0, 0.0)
        xr, yr = cpu._create_guide_star_grid(SAMPLING, DIAMETER, np.pi / 2, 0.0, 0.0)

        # _rotateWFS: x' = x cos - y sin, y' = y cos + x sin
        assert np.allclose(xr, -y0, atol=1e-12), "pi/2 rotation did not map x to -y"
        assert np.allclose(yr, x0, atol=1e-12), "pi/2 rotation did not map y to x"

    def test_full_turn_is_identity(self):
        """The regression guard: under the old degree/radian mix-up this was a 360 rad turn."""
        x0, y0 = cpu._create_guide_star_grid(SAMPLING, DIAMETER, 0.0, 0.0, 0.0)
        xr, yr = cpu._create_guide_star_grid(SAMPLING, DIAMETER, 2 * np.pi, 0.0, 0.0)

        assert np.allclose(xr, x0, atol=1e-12), "2*pi rotation is not the identity"
        assert np.allclose(yr, y0, atol=1e-12), "2*pi rotation is not the identity"

    def test_rotation_preserves_pairwise_distances(self):
        """A rigid rotation cannot change the covariance of a grid with itself."""

        def positions(angle):
            x, y = cpu._create_guide_star_grid(SAMPLING, DIAMETER, angle, 0.0, 0.0)
            return (x + 1j * y).ravel()

        c0 = cpu._covariance_matrix(positions(0.0), positions(0.0), R0, L0, 1.0)
        cr = cpu._covariance_matrix(positions(0.37), positions(0.37), R0, L0, 1.0)
        assert np.allclose(c0, cr, rtol=1e-10)


class TestZeroSeparation:
    """Coincident points take the variance term, exactly or to within rounding (#90)."""

    @staticmethod
    def _grid():
        x, y = np.meshgrid(
            np.linspace(-1, 1, SAMPLING) * DIAMETER / 2,
            np.linspace(-1, 1, SAMPLING) * DIAMETER / 2,
        )
        return (x + 1j * y).ravel()

    def test_diagonal_is_the_variance_term(self):
        z = self._grid()
        cov = cpu._covariance_matrix(z, z, R0, L0, 1.0)
        expected = analytic_var_term(R0, L0)
        assert np.allclose(np.diag(cov), expected, rtol=1e-9), (
            f"diagonal {np.diag(cov)[0]} != var_term {expected}"
        )

    def test_diagonal_survives_rounding_noise(self):
        """The real-world trigger: the two grids are built by different arithmetic, so a
        mathematically-zero separation can come out as a few ULPs instead."""
        z = self._grid()
        z_noisy = z * (1 + 3e-16)

        clean = np.diag(cpu._covariance_matrix(z, z, R0, L0, 1.0))
        noisy = np.diag(cpu._covariance_matrix(z, z_noisy, R0, L0, 1.0))

        assert np.allclose(clean, analytic_var_term(R0, L0), rtol=1e-9)
        assert np.allclose(clean, noisy, rtol=1e-6), (
            "a few ULPs of coordinate noise changed the variance term"
        )


@pytest.mark.skipif(not reconstructor_module.CUDA, reason="requires CuPy and a CUDA device")
class TestBackendAgreement:
    """The CPU and GPU covariance kernels must agree (#90)."""

    def test_covariance_matrices_agree(self):
        import cupy as cp

        from pyTomoAO import tomographyUtilsGPU as gpu

        x, y = np.meshgrid(
            np.linspace(-1, 1, SAMPLING) * DIAMETER / 2,
            np.linspace(-1, 1, SAMPLING) * DIAMETER / 2,
        )
        z1 = (x + 1j * y).ravel()
        # A second grid offset and scaled like a cone-effect-corrected LGS grid, so the
        # zero-separation entry arises from different arithmetic on each side.
        z2 = z1 * 0.98 + (0.3 + 0.2j)

        ref = cpu._covariance_matrix(z1, z2, R0, L0, 1.0)
        got = cp.asnumpy(
            gpu._covariance_matrix(cp.asarray(z1), cp.asarray(z2), R0, L0, 1.0, use_float32=False)
        )

        assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-9

    def test_near_coincident_grids_agree(self):
        """The exact case that exposed #90.

        A separation that is mathematically zero evaluates to ~1e-18 on one backend and to
        exactly 0.0 on the other. Before the fix the GPU sent those into its tiny-z Bessel
        shortcut and returned 1.887x the variance term.
        """
        import cupy as cp

        from pyTomoAO import tomographyUtilsGPU as gpu

        z = self._z()
        z_noisy = z * (1 + 3e-16)

        ref = cpu._covariance_matrix(z, z, R0, L0, 1.0)
        got = cp.asnumpy(
            gpu._covariance_matrix(
                cp.asarray(z), cp.asarray(z_noisy), R0, L0, 1.0, use_float32=False
            )
        )
        assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-6

    def test_tiny_argument_bessel_matches_analytic_limit(self):
        """Guards the CUDA tiny-z shortcut constant directly.

        The shortcut used 1.89718990814 where the small-argument limit
        K_v(z) -> (1/2)*Gamma(v)*(2/z)^v gives 2^(5/6)*Gamma(5/6)/2 = 1.005634918 -- a
        factor of 1.8866 too large.
        """
        import cupy as cp

        from pyTomoAO import tomographyUtilsGPU as gpu

        z = np.array([1e-13, 1e-14, 1e-15])
        expected = 2 ** (5 / 6) * gamma_fn(5 / 6) / 2 * z ** (-5 / 6)
        got = cp.asnumpy(gpu._kv56(cp.asarray(z), use_float32=False))

        assert np.allclose(got, expected, rtol=1e-6), f"ratio {got / expected}"

    @staticmethod
    def _z():
        x, y = np.meshgrid(
            np.linspace(-1, 1, SAMPLING) * DIAMETER / 2,
            np.linspace(-1, 1, SAMPLING) * DIAMETER / 2,
        )
        return (x + 1j * y).ravel()
