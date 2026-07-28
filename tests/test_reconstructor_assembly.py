"""Tests for reconstructor assembly and pupil masking.

Covers the non-idempotent `assemble_reconstructor_and_fitting` (#93) and the zero-valued
grid points that were being turned into NaN (#94).
"""

import importlib
import logging
from pathlib import Path

import numpy as np
import pytest

logger = logging.getLogger(__name__)

reconstructor_module = importlib.import_module("pyTomoAO.tomographicReconstructor")

CONFIG = Path(__file__).resolve().parents[1] / "examples" / "benchmark"
CONFIG /= "reconstructor_config_revolt.yaml"


@pytest.fixture(scope="module")
def built():
    """A built single-LGS reconstructor, shared across the module (the build is the slow part)."""
    rec = reconstructor_module.tomographicReconstructor(str(CONFIG))
    rec.build_reconstructor()
    return rec


class TestAssembleIdempotence:
    """Repeat calls must return the same FR (#93)."""

    @pytest.mark.parametrize("slopes_order", ["simu", "keck", "inverted"])
    def test_repeat_calls_agree(self, built, slopes_order):
        first = built.assemble_reconstructor_and_fitting(
            nChannels=1, slopesOrder=slopes_order
        ).copy()
        second = built.assemble_reconstructor_and_fitting(
            nChannels=1, slopesOrder=slopes_order
        ).copy()

        assert np.allclose(first, second), (
            f"{slopes_order}: second call changed FR by "
            f"{np.abs(first - second).max():.3g} (max |FR| {np.abs(first).max():.3g})"
        )

    def test_reconstructor_is_not_mutated(self, built):
        """assemble must not rewrite the matrix build_reconstructor produced."""
        before = built.reconstructor.copy()
        built.assemble_reconstructor_and_fitting(nChannels=1, slopesOrder="simu")
        assert np.array_equal(built.reconstructor, before), (
            "assemble_reconstructor_and_fitting mutated the tomographic reconstructor"
        )

    def test_scaling_factor_is_linear(self, built):
        """A knob users are told to calibrate; tuning it must not depend on call history."""
        one = built.assemble_reconstructor_and_fitting(
            nChannels=1, slopesOrder="simu", scalingFactor=1.0
        ).copy()
        ten = built.assemble_reconstructor_and_fitting(
            nChannels=1, slopesOrder="simu", scalingFactor=10.0
        ).copy()
        assert np.allclose(ten, 10.0 * one, rtol=1e-9)

    def test_invalid_slopes_order_rejected(self, built):
        with pytest.raises(ValueError, match="Invalid slopes order"):
            built.assemble_reconstructor_and_fitting(nChannels=1, slopesOrder="sideways")


class TestPupilMasking:
    """Zero is a legal reconstructed value, not a mask sentinel (#94)."""

    def test_zero_valued_points_are_not_masked(self, built):
        n_valid = int(built.gridMask.sum())
        rec = reconstructor_module.tomographicReconstructor(str(CONFIG))
        rec._gridMask = built.gridMask
        rec.method = "Model"
        # A reconstructor that maps every slope vector to exactly zero.
        rec._reconstructor = np.zeros((n_valid, 40))

        wavefront = rec.reconstruct_wavefront(np.ones(40))

        assert np.count_nonzero(~np.isnan(wavefront)) == n_valid, (
            "valid grid points reconstructing to zero were turned into NaN"
        )
        assert np.all(wavefront[built.gridMask] == 0.0)
        assert np.all(np.isnan(wavefront[~built.gridMask]))

    def test_masked_points_are_nan(self, built):
        n_valid = int(built.gridMask.sum())
        rec = reconstructor_module.tomographicReconstructor(str(CONFIG))
        rec._gridMask = built.gridMask
        rec.method = "Model"
        rng = np.random.default_rng(0)
        rec._reconstructor = rng.standard_normal((n_valid, 40))

        wavefront = rec.reconstruct_wavefront(np.ones(40))

        assert wavefront.shape == built.gridMask.shape
        assert np.all(np.isnan(wavefront[~built.gridMask]))
        assert np.all(np.isfinite(wavefront[built.gridMask]))
