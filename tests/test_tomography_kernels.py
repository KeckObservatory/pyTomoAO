"""Tests for the covariance kernels shared by the CPU and GPU backends.

Covers the WFS rotation units (#92) and the zero-separation variance term (#90). Both were
defects that no configuration in the repository exercised: every shipped config sets
`wfsLensletsRotation` to zero, and the zero-separation case only misbehaves when floating
point puts a mathematically-zero distance a few ULPs off zero.
"""

import logging

import numpy as np
import pytest
from scipy.special import gamma as gamma_fn
from scipy.special import kv

from pyTomoAO import reconstructor as reconstructor_module
from pyTomoAO import tomographyUtilsCPU as cpu

logger = logging.getLogger(__name__)


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


class TestBesselAccuracy:
    """`_kv56` must track scipy across the whole range it is evaluated on (#97).

    The hand-rolled kernel exists for speed -- it is ~20x faster than
    `scipy.special.kv` -- but it switches from a series to an asymptotic expansion, and the
    crossover is easy to get wrong. It previously sat at z = 2, where the asymptotic series
    is nowhere near converged, costing seven digits over the range that carries most of the
    pupil's baselines.
    """

    Z = np.logspace(-4, 1.7, 600)

    def test_matches_scipy_across_the_range(self):
        ref = kv(5 / 6, self.Z)
        got = np.real(cpu._kv56(self.Z.astype(np.complex128)))
        rel = np.abs(got - ref) / np.abs(ref)
        assert rel.max() < 1e-6, f"max rel err {rel.max():.3e} at z={self.Z[rel.argmax()]:.4f}"

    def test_no_discontinuity_at_the_crossover(self):
        """A step at the crossover would show up as a kink in the covariance function."""
        z = np.linspace(8.9, 9.1, 401)
        ref = kv(5 / 6, z)
        got = np.real(cpu._kv56(z.astype(np.complex128)))
        rel = np.abs(got - ref) / np.abs(ref)
        assert rel.max() < 1e-6, f"max rel err {rel.max():.3e} across the crossover"

    def test_accurate_in_the_asymptotic_branch(self):
        """Well past the crossover, where the a_5 coefficient carries real weight.

        a_5 read 5005/177147 instead of 40040/177147 -- exactly 8x too small.
        """
        z = np.linspace(9.0, 40.0, 200)
        ref = kv(5 / 6, z)
        got = np.real(cpu._kv56(z.astype(np.complex128)))
        rel = np.abs(got - ref) / np.abs(ref)
        assert rel.max() < 1e-7, f"max rel err {rel.max():.3e} at z={z[rel.argmax()]:.4f}"

    def test_real_kernel_matches_the_complex_one(self):
        """The real kernel is the hot path; the complex one stays for compatibility (#102).

        Both carry their own copy of the expansion, so this guards against them drifting
        apart. A coefficient disagreeing between copies of this expansion is exactly what
        #97 turned out to be.
        """
        real = cpu._kv56_real(self.Z)
        complex_ = np.real(cpu._kv56(self.Z.astype(np.complex128)))
        assert np.allclose(real, complex_, rtol=1e-13, atol=0), (
            f"max rel diff {np.abs(real / complex_ - 1).max():.3e}"
        )

    def test_real_kernel_matches_scipy(self):
        ref = kv(5 / 6, self.Z)
        rel = np.abs(cpu._kv56_real(self.Z) - ref) / np.abs(ref)
        assert rel.max() < 1e-6, f"max rel err {rel.max():.3e} at z={self.Z[rel.argmax()]:.4f}"


class TestMaskedCovarianceEquivalence:
    """Masking the coordinates must equal masking the finished covariance matrix (#101).

    The kernels used to evaluate the covariance over the full grid and then drop ~71% of
    it. Restricting the coordinates first has to select exactly the same pairs -- the rows
    and columns of the full matrix are indexed by `z.T.flatten()`, so the mask has to be
    applied to that same ordering.
    """

    @staticmethod
    def _grids():
        x1, y1 = cpu._create_guide_star_grid(SAMPLING, DIAMETER, 0.21, 0.05, -0.03)
        x2, y2 = np.meshgrid(
            np.linspace(-1, 1, SAMPLING) * DIAMETER / 2,
            np.linspace(-1, 1, SAMPLING) * DIAMETER / 2,
        )
        # An off-centre elliptical pupil: deliberately not symmetric under transpose, so a
        # mask applied to the wrong axis ordering would show up.
        yy, xx = np.mgrid[0:SAMPLING, 0:SAMPLING]
        cy, cx = (SAMPLING - 1) / 2 + 1.0, (SAMPLING - 1) / 2 - 0.5
        mask = ((xx - cx) / 5.5) ** 2 + ((yy - cy) / 4.0) ** 2 < 1.0
        return (x1 + 1j * y1), (x2 + 1j * y2), mask

    def test_matches_masking_after_the_fact(self):
        z1, z2, mask = self._grids()
        assert 0 < mask.sum() < mask.size, "test mask must be a strict subset"
        mask_flat = mask.flatten()

        full = cpu._covariance_matrix(z1.T, z2.T, R0, L0, 0.7)
        after = full[mask_flat, :][:, mask_flat]

        before = cpu._covariance_matrix(
            z1.T.flatten()[mask_flat], z2.T.flatten()[mask_flat], R0, L0, 0.7
        )

        assert before.shape == after.shape == (mask.sum(), mask.sum())
        assert np.array_equal(before, after), (
            f"max diff {np.abs(before - after).max():.3e} -- coordinate masking is not "
            "equivalent to masking the finished matrix"
        )


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


@pytest.mark.skipif(not reconstructor_module.CUDA, reason="requires CuPy and a CUDA device")
class TestFieldOptimisationRank:
    """Both backends must return the optimisation directions on their own axis (#119).

    The GPU kernel used to concatenate them into a 2-D array while the CPU kernel stacked
    them into a 3-D one. `_build_reconstructor_model` then weighted the result with
    `fitSrcWeight[:, None, None]`, which is only correct when there is a single direction.
    With the bundled `keck` configuration (nFitSrc = 7, so 49 directions) that broadcast
    asked for a 49x larger array and the build died allocating 66 GB on the device.
    """

    @staticmethod
    def _params(config, n_fit_src):
        rec = reconstructor_module.tomographicReconstructor(config, force_cpu=True)
        rec.tomoParams.nFitSrc = n_fit_src
        if n_fit_src > 1:
            rec.tomoParams.fovOptimization = 20.0
        _, grid_mask = cpu._sparseGradientMatrixAmplitudeWeighted(
            rec.lgsWfsParams.validLLMapSupport, None, 2
        )
        rec.tomoParams.sampling = grid_mask.shape[0]
        return rec, grid_mask

    @pytest.mark.parametrize("n_fit_src", [1, 2])
    def test_backends_agree_on_shape_and_values(self, revolt_config, n_fit_src):
        import cupy as cp

        from pyTomoAO import tomographyUtilsGPU as gpu

        rec, grid_mask = self._params(revolt_config, n_fit_src)
        args = (rec.tomoParams, rec.lgsWfsParams, rec.atmParams, rec.lgsAsterismParams, grid_mask)

        on_cpu = cpu._cross_correlation(*args)
        on_gpu = cp.asnumpy(gpu._cross_correlation(*args, use_float32=False))

        assert on_cpu.ndim == 3, "the CPU kernel should keep directions on their own axis"
        assert on_gpu.shape == on_cpu.shape, (
            f"nFitSrc={n_fit_src}: GPU returned {on_gpu.shape}, CPU {on_cpu.shape}"
        )
        assert on_cpu.shape[0] == n_fit_src**2
        assert np.abs(on_gpu - on_cpu).max() / np.abs(on_cpu).max() < 1e-9
