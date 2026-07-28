"""Tests for the bundled reference configurations (#106).

The published wheel used to contain no data at all, so the configuration path in the
README raised FileNotFoundError for anyone who installed with pip.
"""

import importlib
from pathlib import Path

import pytest
import yaml

from pyTomoAO import example_config, list_example_configs

reconstructor_module = importlib.import_module("pyTomoAO.tomographicReconstructor")

REQUIRED_SECTIONS = (
    "atmosphere_parameters",
    "lgs_asterism",
    "lgs_wfs_parameters",
    "tomography_parameters",
    "dm_parameters",
)


def test_lists_the_bundled_configurations():
    assert list_example_configs() == ["kapa", "kapa-single-channel", "keck", "revolt"]


@pytest.mark.parametrize("name", list_example_configs())
def test_config_is_installed_alongside_the_package(name):
    """The path must resolve inside the package, not relative to the repository root."""
    path = Path(example_config(name))
    assert path.is_file(), f"{name} -> {path} does not exist"
    assert path.suffix == ".yaml"
    package_dir = Path(reconstructor_module.__file__).resolve().parent
    assert package_dir in path.resolve().parents, f"{path} is not inside {package_dir}"


@pytest.mark.parametrize("name", list_example_configs())
def test_config_parses_and_has_every_section(name):
    with open(example_config(name)) as handle:
        config = yaml.safe_load(handle)
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    assert not missing, f"{name} is missing {missing}"


def test_unknown_name_names_the_alternatives():
    with pytest.raises(KeyError, match="Unknown example configuration"):
        example_config("no-such-system")


def test_default_config_builds_a_reconstructor():
    """The documented one-liner works with no repository checkout."""
    rec = reconstructor_module.tomographicReconstructor(example_config("revolt"))
    assert rec.reconstructor.ndim == 2
