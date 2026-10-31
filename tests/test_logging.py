"""pyTomoAO's logging: module loggers, and the deprecated ``logger`` arguments (issue #127)."""

import logging
import warnings

import pytest

from pyTomoAO import example_config
from pyTomoAO.dm_fitting import fitting
from pyTomoAO.reconstructor import tomographicReconstructor


@pytest.fixture
def config():
    return example_config("kapa")


def test_no_warning_by_default(config):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        rec = tomographicReconstructor(config, force_cpu=True)
        fitting(rec.dmParams)


def test_module_loggers_receive_the_messages(config, caplog):
    with caplog.at_level(logging.INFO, logger="pyTomoAO"):
        rec = tomographicReconstructor(config, force_cpu=True)
        fitting(rec.dmParams)
    names = {r.name for r in caplog.records}
    assert "pyTomoAO.reconstructor" in names
    assert "pyTomoAO.dm_fitting" in names


@pytest.mark.parametrize("owner", ["reconstructor", "fitting"])
def test_logger_argument_is_deprecated_but_still_honoured(config, owner, caplog):
    custom = logging.getLogger("my.app")
    rec = tomographicReconstructor(config, force_cpu=True)
    with (
        caplog.at_level(logging.INFO, logger="my.app"),
        pytest.warns(DeprecationWarning, match="'logger' argument"),
    ):
        if owner == "reconstructor":
            tomographicReconstructor(config, logger=custom, force_cpu=True)
        else:
            fitting(rec.dmParams, logger=custom)
    # Backward compatible: the constructor's messages still go to the logger passed in.
    assert any(r.name == "my.app" for r in caplog.records)


def test_assembly_does_not_warn(config):
    rec = tomographicReconstructor(config, force_cpu=True)
    rec.build_reconstructor()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        rec.assemble_reconstructor_and_fitting()
