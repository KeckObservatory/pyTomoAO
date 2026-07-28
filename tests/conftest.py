"""Shared fixtures.

Keeps tests independent of the working directory: paths are resolved from this file rather
than assumed relative to the repository root, so `pytest tests/...` works from anywhere.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples" / "benchmark"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def example_configs() -> Path:
    """Directory holding the example YAML configurations."""
    return EXAMPLES


@pytest.fixture(scope="session")
def kapa_config() -> str:
    """Four-LGS KAPA configuration (20x20 lenslets, 21x21 actuators, 7 layers)."""
    return str(EXAMPLES / "tomography_config_kapa.yaml")


@pytest.fixture(scope="session")
def revolt_config() -> str:
    """Single-LGS REVOLT configuration. Small enough for fast round trips."""
    return str(EXAMPLES / "reconstructor_config_revolt.yaml")
