"""Tests for the explicit slope-noise covariance of the model-based reconstructor."""

import numpy as np
import pytest

from pyTomoAO import example_config
from pyTomoAO import reconstructor as reconstructor_module
from pyTomoAO.slodar import slodarResult
from pyTomoAO.tomographyUtilsCPU import _noise_covariance_matrix

CONFIG = example_config("revolt")


class TestNoiseCovarianceMatrix:
    def test_scalar(self):
        c = _noise_covariance_matrix(2.5, 6, 2)
        assert np.array_equal(c, 2.5 * np.eye(6))

    def test_per_sensor(self):
        c = _noise_covariance_matrix([1.0, 3.0], 6, 2)
        assert np.array_equal(np.diag(c), [1, 1, 1, 3, 3, 3])
        assert np.count_nonzero(c - np.diag(np.diag(c))) == 0

    def test_per_slope(self):
        v = np.arange(6.0)
        assert np.array_equal(np.diag(_noise_covariance_matrix(v, 6, 2)), v)

    def test_full_matrix_is_symmetrised(self):
        a = np.arange(16.0).reshape(4, 4)
        c = _noise_covariance_matrix(a, 4, 1)
        assert np.allclose(c, 0.5 * (a + a.T))

    @pytest.mark.parametrize(
        "bad", [np.ones(5), -np.ones(6), np.ones((3, 3)), [1.0, 2.0, 3.0, 4.0]]
    )
    def test_rejects_bad_input(self, bad):
        with pytest.raises(ValueError):
            _noise_covariance_matrix(bad, 6, 2)


@pytest.fixture(scope="module")
def alpha_reference():
    """Single-LGS reconstructor built with the default alpha regularisation."""
    rec = reconstructor_module.tomographicReconstructor(CONFIG)
    rec.build_reconstructor(alpha=0.1)
    css = rec.Gamma @ rec.Cxx @ rec.Gamma.T
    return rec, float(np.mean(np.diag(css)))


class TestBuildWithNoiseCovariance:
    def test_scalar_matches_alpha(self, alpha_reference):
        """sigma^2 = alpha * mean(diag Css) must give the alpha reconstructor exactly."""
        ref, mean_css = alpha_reference
        rec = reconstructor_module.tomographicReconstructor(CONFIG)
        rec.build_reconstructor(noiseCovariance=0.1 * mean_css)
        assert np.allclose(rec.CnZ, ref.CnZ)
        assert np.allclose(rec.RecStatSA, ref.RecStatSA, rtol=1e-8, atol=1e-12)

    def test_noisier_slopes_are_down_weighted(self, alpha_reference):
        """A per-slope variance increase must shrink the reconstructor columns of those slopes."""
        ref, mean_css = alpha_reference
        n = ref.Gamma.shape[0]
        var = np.full(n, 0.1 * mean_css)
        var[: n // 2] *= 100.0  # X slopes much noisier
        rec = reconstructor_module.tomographicReconstructor(CONFIG)
        rec.build_reconstructor(noiseCovariance=var)
        x = np.linalg.norm(rec.RecStatSA[:, : n // 2]) / np.linalg.norm(ref.RecStatSA[:, : n // 2])
        y = np.linalg.norm(rec.RecStatSA[:, n // 2 :]) / np.linalg.norm(ref.RecStatSA[:, n // 2 :])
        assert x < 0.5 < y

    def test_im_based_rejects_noise_covariance(self):
        rec = reconstructor_module.tomographicReconstructor(CONFIG)
        with pytest.raises(ValueError, match="model-based"):
            rec.build_reconstructor(IM=np.eye(4), noiseCovariance=1.0)


def _result(noise):
    return slodarResult(
        altitude=np.array([0.0]),
        altitude_zenith=np.array([0.0]),
        r0m53=np.array([1.0]),
        cn2dh=np.array([1.0]),
        fractionalCn2=np.array([1.0]),
        r0=1.0,
        r0_zenith=1.0,
        L0=25.0,
        wavelength=5e-7,
        noiseVariance=np.asarray(noise, float),
    )


class TestSlodarNoiseCovariance:
    def test_conversion(self):
        sig = 0.05 / 206264.806  # 0.05" rms angle-of-arrival noise
        res = _result([sig**2, 4 * sig**2])
        v = res.noise_covariance(d=0.5, nValidSubap=3)
        expected = (2 * np.pi * 0.5 * sig / 5e-7) ** 2
        assert v.shape == (12,)
        assert np.allclose(v[:6], expected)
        assert np.allclose(v[6:], 4 * expected)
        # Same noise, reported at another wavelength, scales as lambda^-2.
        assert np.allclose(res.noise_covariance(0.5, 3, wavelength=1e-6)[:6], expected / 4)

    def test_requires_noise_fit(self):
        with pytest.raises(ValueError, match="No noise estimate"):
            _result([]).noise_covariance(0.5, 3)
