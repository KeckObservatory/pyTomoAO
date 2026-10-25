"""Tests for the SLODAR turbulence profiler (pyTomoAO.slodar).

The covariance-map estimator is checked against a brute-force sum, the geometry against the
closed-form altitude bins, the response model against pyTomoAO's own gradient operator and
phase covariance, and the full chain on Gaussian slopes drawn from the model.
"""

import numpy as np
import pytest

from pyTomoAO import tomographyUtilsCPU as cpu
from pyTomoAO.slodar import (
    _KOLMOGOROV_CN2,
    _R0_CN2,
    ARCSEC2RAD,
    COMPONENTS,
    _gradient_stencil,
    _pyTomoAO_stencil,
    slodarProfiler,
)


def _annulus(n=8, inner=1.0):
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2)
    return (r < n / 2) & (r > inner)


def _asymmetric_map(n=8):
    """Valid map that is not symmetric under transpose, to catch ordering mix-ups."""
    v = _annulus(n)
    v[0, :] = False
    v[:, -1] = False
    assert not np.array_equal(v, v.T)
    return v


def _brute_force_map(si, sj, rows, cols, n, a, b):
    """sum_p <si^a(p) sj^b(p + delta)> / N_delta, by explicit loops."""
    T = si.shape[0]
    out = np.zeros((2 * n - 1, 2 * n - 1))
    cnt = np.zeros_like(out)
    for p in range(rows.size):
        for q in range(rows.size):
            dy = rows[q] - rows[p] + n - 1
            dx = cols[q] - cols[p] + n - 1
            out[dy, dx] += (si[:, a, p] * sj[:, b, q]).sum() / T
            cnt[dy, dx] += 1
    out[cnt > 0] /= cnt[cnt > 0]
    return out, cnt


# ---------------------------------------------------------------------------
class TestConstants:
    def test_r0_cn2_constant(self):
        """r0^{-5/3} = 0.423 k^2 Cn2 dh and D_phi = 2.914 k^2 Cn2 dh r^{5/3}."""
        assert pytest.approx(0.42336, rel=1e-4) == _R0_CN2
        assert pytest.approx(2.91438, rel=1e-4) == _KOLMOGOROV_CN2

    def test_cn2_conversion_round_trip(self):
        cn2 = 1e-13
        w = slodarProfiler.cn2dh_to_r0m53(cn2, 5e-7)
        r0 = w ** (-3 / 5)
        # r0 = (0.423 k^2 Cn2dh)^(-3/5): Cn2dh = 1e-13 m^1/3 gives r0 = 0.320 m at 500 nm.
        k = 2 * np.pi / 5e-7
        assert r0 == pytest.approx((0.42336 * k**2 * cn2) ** (-3 / 5), rel=1e-4)
        assert r0 == pytest.approx(0.3198, rel=1e-3)


class TestStencil:
    def test_three_point_stencil_matches_gradient_operator(self):
        """nEdgePoints=3 is exactly the 3x3 stencil of the tomographic gradient operator."""

        def as_dict(pts, wx, wy):
            return {
                (round(x, 9), round(y, 9)): (a, b)
                for (x, y), a, b in zip(pts, wx, wy)
                if a != 0 or b != 0
            }

        a = as_dict(*_gradient_stencil(3))
        b = as_dict(*_pyTomoAO_stencil())
        assert a.keys() == b.keys()
        for k in a:
            np.testing.assert_allclose(a[k], b[k], atol=1e-15)

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_stencil_differentiates_ramps(self, n):
        pts, wx, wy = _gradient_stencil(n)
        assert np.dot(wx, pts[:, 0]) == pytest.approx(1.0)
        assert np.dot(wx, pts[:, 1]) == pytest.approx(0.0, abs=1e-14)
        assert np.dot(wy, pts[:, 1]) == pytest.approx(1.0)
        assert np.dot(wy, pts[:, 0]) == pytest.approx(0.0, abs=1e-14)
        assert wx.sum() == pytest.approx(0.0, abs=1e-14)


# ---------------------------------------------------------------------------
class TestCovarianceMaps:
    @pytest.mark.parametrize("subapOrder", ["F", "C"])
    @pytest.mark.parametrize("removeTT", [False, True])
    def test_fft_maps_match_brute_force(self, subapOrder, removeTT):
        valid = _asymmetric_map(6)
        gs = np.array([[0, 0], [10, 0], [0, 10]]) * ARCSEC2RAD
        p = slodarProfiler(valid, 3.0, gs, subapOrder=subapOrder)
        rng = np.random.default_rng(0)
        T, n = 37, p.nValidSubap
        s = rng.standard_normal((T, 3, 2, n))
        maps = p.compute_covariance_maps(s.reshape(T, -1), removeTT=removeTT, chunkSize=10)

        s0 = s - s.mean(0)
        if removeTT:
            s0 = s0 - s0.mean(3, keepdims=True)
        for i, j in [(0, 0), (0, 2), (1, 2)]:
            for c, comp in enumerate(COMPONENTS):
                a, b = "xy".index(comp[0]), "xy".index(comp[1])
                ref, cnt = _brute_force_map(s0[:, i], s0[:, j], p._rows, p._cols, p.nLenslet, a, b)
                np.testing.assert_allclose(maps[(i, j)][c], ref, atol=1e-12)
        np.testing.assert_array_equal(p.counts, cnt.astype(int))

    def test_counts_normalisation(self):
        valid = _annulus(8)
        p = slodarProfiler(valid, 4.0, [[0, 0], [1e-5, 0]])
        N = 8
        assert p.counts[N - 1, N - 1] == valid.sum()
        assert p.counts.sum() == valid.sum() ** 2
        np.testing.assert_array_equal(p.counts, p.counts[::-1, ::-1])

    @pytest.mark.parametrize("order", ["simu", "inverted", "keck"])
    def test_slope_layouts_are_equivalent(self, order):
        valid = _annulus(6)
        gs = np.array([[0, 0], [10, 0]]) * ARCSEC2RAD
        ref = slodarProfiler(valid, 3.0, gs, slopesOrder="simu")
        alt = slodarProfiler(valid, 3.0, gs, slopesOrder=order)
        rng = np.random.default_rng(1)
        s = rng.standard_normal((20, 2, 2, ref.nValidSubap))
        ref.compute_covariance_maps(s.reshape(20, -1))
        alt.compute_covariance_maps(alt._format_slopes(s))
        for k in ref.maps:
            np.testing.assert_allclose(alt.maps[k], ref.maps[k])

    def test_shift_is_recovered(self):
        """A pure translation between two sensors puts the cross-covariance peak at the shift."""
        N = 10
        valid = np.ones((N, N), dtype=bool)
        p = slodarProfiler(valid, 5.0, np.array([[0, 0], [20, 0]]) * ARCSEC2RAD, subapOrder="C")
        rng = np.random.default_rng(3)
        big = rng.standard_normal((200, 2, N, N + 3))
        s0 = big[..., :N]
        s1 = big[..., 3 : N + 3]  # s1(col) = s0(col + 3) -> peak at delta_x = -3
        s = np.stack([s0, s1], axis=1).reshape(200, 2, 2, N * N)
        p.compute_covariance_maps(s.reshape(200, -1))
        xx = p.maps[(0, 1)][0]
        iy, ix = np.unravel_index(np.argmax(xx), xx.shape)
        assert (iy - (N - 1), ix - (N - 1)) == (0, -3)

    def test_wrong_slope_count_raises(self):
        p = slodarProfiler(_annulus(6), 3.0, [[0, 0], [1e-5, 0]])
        with pytest.raises(ValueError, match="slopes per frame"):
            p.compute_covariance_maps(np.zeros((10, 7)))


# ---------------------------------------------------------------------------
class TestGeometry:
    def test_ngs_bins(self):
        theta = 30 * ARCSEC2RAD
        p = slodarProfiler(_annulus(8), 4.0, [[0, 0], [theta, 0]])
        np.testing.assert_allclose(p.native_altitudes(0, 1), np.arange(8) * 0.5 / theta)

    def test_lgs_bins_are_cone_compressed(self):
        """h = delta d / (theta + delta d / H), i.e. delta d (1 - h/H) / theta."""
        theta, H = 30 * ARCSEC2RAD, 90e3
        p = slodarProfiler(_annulus(8), 4.0, [[0, 0], [theta, 0]], gsHeight=H)
        h = p.native_altitudes(0, 1)
        delta = np.arange(8) * 0.5
        np.testing.assert_allclose(h, delta * (1 - h / H) / theta)
        ngs = slodarProfiler(_annulus(8), 4.0, [[0, 0], [theta, 0]]).native_altitudes(0, 1)
        assert np.all(h[1:] < ngs[1:])

    def test_lgs_response_peaks_on_native_bin(self):
        """With the cone effect the cross-covariance of a layer on bin k peaks at offset k.

        Maps are sum_p <s_i(p) s_j(p + delta)>, so the peak lies at delta = -(theta_j -
        theta_i) h / (d (1 - h/H)): opposite to the baseline direction.
        """
        theta, H = 40 * ARCSEC2RAD, 90e3
        N = 12
        p = slodarProfiler(np.ones((N, N), bool), 6.0, [[0, 0], [theta, 0]], gsHeight=H)
        for k in (2, 5, 8):
            h = p.native_altitudes(0, 1)[k]
            xx = p.response_maps(h)[(0, 1)][0]
            iy, ix = np.unravel_index(np.argmax(xx), xx.shape)
            assert (iy - (N - 1), ix - (N - 1)) == (0, -k)

    def test_diagonal_baseline_lattice_step(self):
        theta = 30 * ARCSEC2RAD
        p = slodarProfiler(_annulus(8), 4.0, [[0, 0], [theta, theta]])
        assert p.lattice_step(0, 1) == pytest.approx(0.5 * np.sqrt(2))
        assert p.native_altitudes(0, 1)[1] == pytest.approx(0.5 * np.sqrt(2) / (theta * np.sqrt(2)))

    def test_generalised_slodar_moves_ground_layer(self):
        """With negative conjugation the ground layer no longer sits at zero offset."""
        theta = 30 * ARCSEC2RAD
        N = 10
        p = slodarProfiler(
            np.ones((N, N), bool), 5.0, [[0, 0], [theta, 0]], conjugationAltitude=-3000.0
        )
        h = p.native_altitudes(0, 1)
        assert h[0] == pytest.approx(-3000.0)
        k0 = int(np.argmin(np.abs(h)))
        xx = p.response_maps(0.0)[(0, 1)][0]
        _, ix = np.unravel_index(np.argmax(xx), xx.shape)
        assert k0 > 0
        assert ix - (N - 1) == -k0
        assert 0.0 in p.default_altitudes()

    def test_default_altitudes_combine_baselines(self):
        gs = np.array([[0, 0], [30, 0], [60, 0]]) * ARCSEC2RAD
        p = slodarProfiler(_annulus(8), 4.0, gs)
        grid = p.default_altitudes()
        assert np.all(np.diff(grid) > 0)
        fine = p.native_altitudes(0, 2)
        np.testing.assert_allclose(grid[: len(fine)], fine)
        assert grid[-1] == pytest.approx(p.native_altitudes(0, 1)[-1])

    def test_from_config_uses_kapa_geometry(self, kapa_config):
        p = slodarProfiler.from_config(kapa_config)
        assert p.nGS == 4
        assert p.nValidSubap == 304
        assert p.d == pytest.approx(0.5)
        assert p.gsHeight == pytest.approx(90e3 / np.cos(np.radians(30)))
        np.testing.assert_allclose(np.hypot(*p.gsDirections.T), 7.6 * ARCSEC2RAD, rtol=1e-9)


# ---------------------------------------------------------------------------
class TestResponseModel:
    def test_auto_covariance_matches_gradient_operator(self):
        """For a ground layer the model equals Gamma Cphi Gamma^T built with pyTomoAO's kernels."""
        valid = _asymmetric_map(6)
        D, N, L0, r0 = 3.0, 6, 20.0, 0.2
        p = slodarProfiler(valid, D, [[0, 0], [1e-5, 0]], L0=L0, wavelength=5e-7)
        C = p.slope_covariance([0.0], [r0 ** (-5 / 3)])
        n = p.nValidSubap
        Css_model = C[: 2 * n, : 2 * n]

        Gamma, gridMask = cpu._sparseGradientMatrixAmplitudeWeighted(valid, None, 2)
        m = gridMask.shape[0]
        x = (np.arange(m) - (m - 1) / 2) * (D / N) / 2
        xx, yy = np.meshgrid(x, x)
        z = (xx + 1j * yy)[gridMask]
        Cphi = cpu._covariance_matrix(z, z, r0, L0, 1.0)
        Css_ref = Gamma @ Cphi @ Gamma.T
        Css_ref *= (5e-7 / (2 * np.pi)) ** 2 / (D / N) ** 2
        np.testing.assert_allclose(Css_model, Css_ref, rtol=1e-6, atol=1e-6 * Css_ref.max())

    def test_response_is_linear_in_strength(self):
        p = slodarProfiler(_annulus(6), 3.0, np.array([[0, 0], [20, 0]]) * ARCSEC2RAD)
        C1 = p.slope_covariance([1000.0], [1.0])
        C2 = p.slope_covariance([1000.0], [2.5])
        np.testing.assert_allclose(C2, 2.5 * C1)

    def test_response_maps_match_slope_covariance(self):
        """Maps of the full model covariance equal the response maps (removeTT both ways)."""
        gs = np.array([[0, 0], [20, 0], [0, 20]]) * ARCSEC2RAD
        p = slodarProfiler(_asymmetric_map(7), 3.5, gs, gsHeight=90e3)
        n = p.nValidSubap
        h = 2500.0
        C = p.slope_covariance([h], [1.0]).reshape(3, 2, n, 3, 2, n)
        for removeTT in (False, True):
            if removeTT:
                P = np.eye(n) - 1.0 / n
                C = np.einsum("pq,iaqjbr,rs->iapjbs", P, C, P)
            R = p.response_maps(h, removeTT=removeTT)
            for i, j in [(0, 1), (1, 2), (0, 0)]:
                for c, comp in enumerate(COMPONENTS):
                    a, b = "xy".index(comp[0]), "xy".index(comp[1])
                    blk = C[i, a, :, j, b, :]
                    ref = np.zeros(p._mapShape).ravel()
                    np.add.at(ref, p._pairOffsetIndex, blk.ravel())
                    ref = ref.reshape(p._mapShape)
                    ref[p.counts > 0] /= p.counts[p.counts > 0]
                    np.testing.assert_allclose(R[(i, j)][c], ref, rtol=1e-9, atol=1e-20)


# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def setup():
    """Three LGS, a 10x10 annular pupil and the native altitude grid up to 15 km."""
    valid = _annulus(10, inner=1.2)
    gs = np.array([[-25, 0], [25, 0], [0, 40]]) * ARCSEC2RAD
    p = slodarProfiler(valid, 5.0, gs, gsHeight=90e3, L0=25.0)
    return p, p.default_altitudes(hMax=15e3)


class TestProfileRetrieval:
    @pytest.mark.parametrize("removeTT", [False, True])
    def test_two_layer_profile_recovered(self, setup, removeTT):
        p, grid = setup
        alts = [grid[0], grid[3]]
        fr = [0.6, 0.4]
        w = p.profile_to_r0m53(0.15, fr)
        s = p.draw_slopes(alts, w, 6000, rng=42)
        res = p.profile(s, removeTT=removeTT, altitudes=grid)
        assert res.r0 == pytest.approx(0.15, rel=0.04)
        assert res.fractionalCn2[0] == pytest.approx(0.6, abs=0.05)
        assert res.fractionalCn2[3] == pytest.approx(0.4, abs=0.05)
        assert res.fractionalCn2.sum() == pytest.approx(1.0)
        assert np.all(res.r0m53 >= 0)

    def test_noise_is_estimated_and_does_not_bias_r0(self, setup):
        p, grid = setup
        alts, fr = [grid[0], grid[4]], [0.5, 0.5]
        w = p.profile_to_r0m53(0.15, fr)
        sig = np.sqrt(np.mean(np.diag(p.slope_covariance(alts, w))))
        s = p.draw_slopes(alts, w, 6000, noiseRms=0.7 * sig, rng=7)
        res = p.profile(s, removeTT=True, altitudes=grid, fitNoise=True)
        assert res.r0 == pytest.approx(0.15, rel=0.05)
        np.testing.assert_allclose(res.noiseVariance, (0.7 * sig) ** 2, rtol=0.1)

    def test_fit_L0(self, setup):
        p, grid = setup
        alts, fr = [grid[0], grid[2]], [0.7, 0.3]
        w = p.profile_to_r0m53(0.15, fr)
        s = p.draw_slopes(alts, w, 8000, L0=10.0, rng=5)
        p.compute_covariance_maps(s)
        res = p.fit_profile(altitudes=grid, fitL0=True, L0Grid=[5, 10, 20, 40, 80])
        assert res.L0 == 10
        assert res.r0 == pytest.approx(0.15, rel=0.05)

    def test_longitudinal_transverse_components(self, setup):
        p, grid = setup
        alts, fr = [grid[0], grid[3]], [0.6, 0.4]
        s = p.draw_slopes(alts, p.profile_to_r0m53(0.15, fr), 4000, rng=9)
        p.compute_covariance_maps(s)
        res = p.fit_profile(altitudes=grid, components=("ll", "tt"), useAuto=True)
        assert res.fractionalCn2[3] == pytest.approx(0.4, abs=0.07)

    def test_to_atmosphere_config_round_trip(self, setup):
        from pyTomoAO.atmosphereParametersClass import atmosphereParameters

        p, grid = setup
        s = p.draw_slopes([grid[0], grid[2]], p.profile_to_r0m53(0.15, [0.5, 0.5]), 2000, rng=1)
        res = p.profile(s, altitudes=grid)
        cfg = res.to_atmosphere_config(minFraction=0.01)
        atm = atmosphereParameters({"atmosphere_parameters": cfg})
        assert atm.r0 == pytest.approx(res.r0)
        assert atm.nLayer == cfg["nLayer"]
        assert str(res).startswith("SLODAR profile")

    def test_fit_requires_maps(self):
        p = slodarProfiler(_annulus(6), 3.0, [[0, 0], [1e-5, 0]])
        with pytest.raises(RuntimeError):
            p.fit_profile()

    def test_unknown_component_raises(self, setup):
        p, grid = setup
        p.compute_covariance_maps(
            np.random.default_rng(0).standard_normal((10, 3 * 2 * p.nValidSubap))
        )
        with pytest.raises(ValueError, match="Unknown components"):
            p.fit_profile(altitudes=grid, components=("zz",))


class TestValidation:
    def test_invalid_arguments(self):
        v = _annulus(6)
        with pytest.raises(ValueError):
            slodarProfiler(v[:, :5], 3.0, [[0, 0]])
        with pytest.raises(ValueError):
            slodarProfiler(v, -1.0, [[0, 0]])
        with pytest.raises(ValueError):
            slodarProfiler(v, 3.0, [[0, 0, 0]])
        with pytest.raises(ValueError):
            slodarProfiler(v, 3.0, [[0, 0]], subapOrder="X")
        with pytest.raises(ValueError):
            slodarProfiler(v, 3.0, [[0, 0]], slopesOrder="bad")
        with pytest.raises(ValueError):
            slodarProfiler(v, 3.0, [[0, 0]], gsHeight=90e3, conjugationAltitude=1e5)
        p = slodarProfiler(v, 3.0, [[0, 0], [0, 0]])
        with pytest.raises(ValueError, match="coincide"):
            p.native_altitudes(0, 1)


class TestPseudoOpenLoop:
    def test_recovers_open_loop_slopes(self):
        from pyTomoAO.slodar import pseudo_open_loop_slopes

        rng = np.random.default_rng(3)
        T, nS, nA = 50, 12, 5
        M = rng.normal(size=(nS, nA))
        s_ol = rng.normal(size=(T, nS))
        c = rng.normal(size=(T, nA))
        s = s_ol.copy()
        s[1:] -= c[:-1] @ M.T  # frame t sees the command computed at t-1
        pol = pseudo_open_loop_slopes(s, c, M, delay=1)
        assert pol.shape == (T - 1, nS)
        assert np.allclose(pol, s_ol[1:])
        # The wrong delay does not recover them.
        assert not np.allclose(pseudo_open_loop_slopes(s, c, M, delay=2), s_ol[2:])
        # Opposite IM sign convention.
        assert np.allclose(pseudo_open_loop_slopes(s, c, -M, delay=1, sign=-1), s_ol[1:])

    def test_shape_checks(self):
        from pyTomoAO.slodar import pseudo_open_loop_slopes

        with pytest.raises(ValueError):
            pseudo_open_loop_slopes(np.zeros((4, 3)), np.zeros((4, 2)), np.zeros((2, 3)))
        with pytest.raises(ValueError):
            pseudo_open_loop_slopes(np.zeros((4, 3)), np.zeros((5, 2)), np.zeros((3, 2)))
        with pytest.raises(ValueError):
            pseudo_open_loop_slopes(np.zeros((4, 3)), np.zeros((4, 2)), np.zeros((3, 2)), delay=4)


# ---------------------------------------------------------------------------
class TestNNLSSafeguard:
    """``_nnls`` must return the optimum even when scipy's nnls does not (SciPy 1.13)."""

    @staticmethod
    def _problem():
        rng = np.random.default_rng(3)
        A = rng.random((300, 12))
        x0 = rng.random(12)
        x0[::3] = 0.0
        b = A @ x0 + 1e-3 * rng.standard_normal(300)
        return A, b

    def test_optimal_result_is_kept(self):
        from pyTomoAO.slodar import _nnls

        A, b = self._problem()
        x = _nnls(A, b)
        grad = A.T @ (A @ x - b)
        assert np.all(x >= 0)
        assert np.all(grad[x <= 0] >= -1e-8 * np.max(np.abs(A.T @ b)))

    def test_non_optimal_nnls_is_refined(self, monkeypatch):
        import scipy.optimize

        from pyTomoAO import slodar

        A, b = self._problem()
        ref = scipy.optimize.lsq_linear(A, b, bounds=(0, np.inf), method="bvls").x

        def bad_nnls(A_, b_, **kwargs):
            x, _ = scipy.optimize.nnls(A_, b_)
            x = x.copy()
            x[np.argmax(x)] = 0.0  # a feasible but clearly non-optimal point
            return x, np.linalg.norm(A_ @ x - b_)

        monkeypatch.setattr(slodar, "nnls", bad_nnls)
        x = slodar._nnls(A, b)
        np.testing.assert_allclose(x, ref, atol=1e-8)

    def test_free_variable_gradient_is_checked(self, monkeypatch):
        """A point on the right support but off the optimum (grad != 0 on free vars) is refined."""
        import scipy.optimize

        from pyTomoAO import slodar

        A, b = self._problem()
        ref = scipy.optimize.lsq_linear(A, b, bounds=(0, np.inf), method="bvls").x

        def early_stop(A_, b_, **kwargs):
            x, _ = scipy.optimize.nnls(A_, b_)
            return 1.2 * x, 0.0  # same active set, every free variable 20% off

        monkeypatch.setattr(slodar, "nnls", early_stop)
        np.testing.assert_allclose(slodar._nnls(A, b), ref, atol=1e-8)


# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def noisy():
    """Own profiler (not the shared ``setup`` one, whose maps other tests overwrite)."""
    valid = _annulus(10, inner=1.2)
    gs = np.array([[-25, 0], [25, 0], [0, 40]]) * ARCSEC2RAD
    p = slodarProfiler(valid, 5.0, gs, gsHeight=90e3, L0=25.0)
    grid = p.default_altitudes(hMax=15e3)
    alts, w = [grid[0], grid[4]], p.profile_to_r0m53(0.15, [0.5, 0.5])
    sig = np.sqrt(np.mean(np.diag(p.slope_covariance(alts, w))))
    s = p.draw_slopes(alts, w, 6000, noiseRms=0.7 * sig, rng=7)
    p.compute_covariance_maps(s, removeTT=True)
    return p, grid, (0.7 * sig) ** 2


class TestNoiseEstimates:
    """Review of #126: no silent-zero noise, and the temporal estimator."""

    @pytest.mark.parametrize("components", [("ll", "tt"), ("xy",), ("xx", "yy"), ("xx",)])
    def test_noise_fitted_whatever_the_components(self, noisy, components):
        p, grid, var = noisy
        res = p.fit_profile(altitudes=grid, components=components, fitNoise=True)
        assert res.noiseVariance.shape == (p.nGS,)
        np.testing.assert_allclose(res.noiseVariance, var, rtol=0.1)
        assert res.r0 == pytest.approx(0.15, rel=0.05)

    def test_no_noise_estimate_without_autos(self, noisy):
        p, grid, _ = noisy
        res = p.fit_profile(altitudes=grid, useAuto=False, fitNoise=True)
        assert res.noiseVariance.size == 0
        with pytest.raises(ValueError, match="No noise estimate"):
            res.noise_covariance(d=0.5, nValidSubap=p.nValidSubap)

    @staticmethod
    def _correlated_slopes(p, alts, w, nFrames, rho, noiseRms, rng):
        """Turbulence slopes that evolve slowly (AR(1), correlation rho per frame) + white noise."""
        rng = np.random.default_rng(rng)
        z = p.draw_slopes(alts, w, nFrames, rng=rng)
        turb = np.empty_like(z)
        turb[0] = z[0]
        for t in range(1, nFrames):
            turb[t] = rho * turb[t - 1] + np.sqrt(1 - rho**2) * z[t]
        return turb + noiseRms * rng.standard_normal(z.shape)

    @pytest.mark.parametrize("removeTT", [False, True])
    def test_temporal_noise_variance(self, setup, removeTT):
        p, grid = setup
        alts, w = [grid[0], grid[3]], p.profile_to_r0m53(0.15, [0.6, 0.4])
        sig = np.sqrt(np.mean(np.diag(p.slope_covariance(alts, w))))
        s = self._correlated_slopes(p, alts, w, 4000, rho=0.9995, noiseRms=0.3 * sig, rng=11)
        v = p.temporal_noise_variance(s, removeTT=removeTT)
        assert v.shape == (p.nGS,)
        np.testing.assert_allclose(v, (0.3 * sig) ** 2, rtol=0.1)

    def test_temporal_noise_variance_rejects_short_series(self, setup):
        p, _ = setup
        with pytest.raises(ValueError):
            p.temporal_noise_variance(np.zeros((2, p.nGS * 2 * p.nValidSubap)), lag=1)


class TestSlopeNoiseCovariance:
    def test_from_reconstructor(self, revolt_config):
        from pyTomoAO.reconstructor import tomographicReconstructor
        from pyTomoAO.slodar import slope_noise_covariance

        rec = tomographicReconstructor(revolt_config, force_cpu=True)
        w = rec.lgsWfsParams
        var = np.full(rec.lgsAsterismParams.nLGS, 1e-12)
        got = slope_noise_covariance(var, reconstructor=rec)
        d = w.D / w.nLenslet  # the physical subaperture, not DSupport / nLenslet
        expected = (2 * np.pi * d / rec.atmParams.wavelength) ** 2 * 1e-12
        assert got.shape == (2 * w.nValidSubap * rec.lgsAsterismParams.nLGS,)
        np.testing.assert_allclose(got, expected)

    def test_requires_geometry(self):
        from pyTomoAO.slodar import slope_noise_covariance

        with pytest.raises(ValueError, match="reconstructor"):
            slope_noise_covariance([1.0], d=0.5)
        with pytest.raises(ValueError, match="empty"):
            slope_noise_covariance([], d=0.5, nValidSubap=3, wavelength=5e-7)


class TestCachingAndChunks:
    def test_response_cache_follows_geometry(self, setup):
        p0, grid = setup
        p = slodarProfiler(p0.validLLMap, p0.D, p0.gsDirections.copy(), gsHeight=90e3, L0=25.0)
        before = p.response_maps(grid[3])[(0, 1)].copy()
        p.gsDirections = p.gsDirections * 2  # wider asterism, same altitude
        after = p.response_maps(grid[3])[(0, 1)]
        assert np.max(np.abs(after - before)) > 0.1 * np.max(np.abs(before))

    def test_auto_chunk_size_matches_explicit(self, setup):
        p, grid = setup
        s = p.draw_slopes([grid[0]], p.profile_to_r0m53(0.15, [1.0]), 300, rng=2)
        a = {k: v.copy() for k, v in p.compute_covariance_maps(s).items()}
        b = p.compute_covariance_maps(s, chunkSize=7)
        for k, v in a.items():
            np.testing.assert_allclose(b[k], v, rtol=0, atol=1e-9 * np.max(np.abs(v)))


def test_pol_sign_self_check_documented_statistic():
    """The docstring's sign check: the right sign has much smaller frame-to-frame increments."""
    from pyTomoAO.slodar import pseudo_open_loop_slopes

    rng = np.random.default_rng(0)
    T, n, rho, g = 4000, 30, 0.99, 0.5
    s_ol = np.empty((T, n))
    x = rng.standard_normal(n)
    for t in range(T):
        x = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(n)
        s_ol[t] = x
    s, c, ci = np.empty((T, n)), np.empty((T, n)), np.zeros(n)
    for t in range(T):  # the loop applies -c (sign=+1), M = I, one-frame delay, POLC
        s[t] = s_ol[t] - ci
        ci = ci + g * (s[t] + ci - ci)
        c[t] = ci
    M = np.eye(n)
    right = pseudo_open_loop_slopes(s, c, M, sign=1)
    wrong = pseudo_open_loop_slopes(s, c, M, sign=-1)
    np.testing.assert_allclose(right, s_ol[1:])
    assert right.var() == pytest.approx(wrong.var(), rel=0.1)  # variance cannot tell
    assert np.var(np.diff(wrong, axis=0)) > 2 * np.var(np.diff(right, axis=0))
