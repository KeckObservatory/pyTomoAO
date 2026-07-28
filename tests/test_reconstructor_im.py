"""Tests for the interaction-matrix-based reconstructor.

This path had no coverage on either backend, which is how the ``cp.sqeeze`` typo and the
missing host-to-device copy of ``IM`` in the GPU kernel both reached a release (#89).

Run with::

    pytest tests/test_reconstructor_im.py -v
"""

import importlib
import logging

import numpy as np
import pytest

from pyTomoAO import example_config

logger = logging.getLogger(__name__)

# See the note in tests/test_tomographicReconstructor.py: the dotted path
# "pyTomoAO.tomographicReconstructor" resolves to the class, not the module.
reconstructor_module = importlib.import_module("pyTomoAO.tomographicReconstructor")

# The single-LGS REVOLT configuration keeps Cxx at 277x277, so the whole build runs in
# well under a second on either backend.
CONFIG = example_config("revolt")


@pytest.fixture
def reconstructor():
    return reconstructor_module.tomographicReconstructor(CONFIG)


@pytest.fixture
def interaction_matrix(reconstructor):
    """A dense stand-in for a measured interaction matrix, shaped like the real thing."""
    n_act = int(reconstructor.dmParams.validActuators.sum())
    n_slopes = int(reconstructor.lgsWfsParams.nValidSubap) * 2
    rng = np.random.default_rng(0)
    return rng.standard_normal((n_slopes, n_act * reconstructor.nLGS))


def test_build_reconstructor_im_runs(reconstructor, interaction_matrix):
    """The IM path builds and returns a usable reconstructor on the active backend."""
    logger.info("Building IM-based reconstructor on the active backend")
    R = reconstructor.build_reconstructor(IM=interaction_matrix)

    n_act = int(reconstructor.dmParams.validActuators.sum())
    n_slopes = interaction_matrix.shape[0]

    assert reconstructor.method == "IM"
    assert R.shape == (n_act, n_slopes), f"Expected {(n_act, n_slopes)}, got {R.shape}"
    assert np.all(np.isfinite(R)), "Reconstructor contains NaN or inf"
    assert np.any(R != 0), "Reconstructor is entirely zero"


def test_build_reconstructor_im_sets_matrices(reconstructor, interaction_matrix):
    """The intermediate covariance matrices are stored and self-consistent."""
    reconstructor.build_reconstructor(IM=interaction_matrix)

    n_act = int(reconstructor.dmParams.validActuators.sum())
    n_gs = reconstructor.nLGS

    assert reconstructor.Cxx.shape == (n_act * n_gs, n_act * n_gs)
    assert reconstructor.CnZ.shape == (interaction_matrix.shape[0],) * 2
    # gridMask comes from the DM geometry on this path, not from the gradient operator.
    assert reconstructor.gridMask.shape == reconstructor.dmParams.validActuators.shape


def test_build_reconstructor_im_accepts_alpha(reconstructor, interaction_matrix):
    """Stronger regularization damps the reconstructor rather than failing."""
    weak = reconstructor.build_reconstructor(IM=interaction_matrix, alpha=1).copy()

    fresh = reconstructor_module.tomographicReconstructor(CONFIG)
    strong = fresh.build_reconstructor(IM=interaction_matrix, alpha=1000).copy()

    assert np.all(np.isfinite(strong))
    assert np.linalg.norm(strong) < np.linalg.norm(weak), (
        "Increasing alpha should shrink the reconstructor"
    )


# A CPU/GPU equivalence test on this path belongs here, but the two backends currently
# disagree on the zero-separation diagonal of the covariance matrices (issue #90) by a factor
# of ~1.887, which contaminates a whole row of the reconstructor. Add the test with the fix
# for #90 rather than pinning the wrong behaviour here.
