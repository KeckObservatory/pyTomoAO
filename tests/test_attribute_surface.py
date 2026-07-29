"""Tests for the reconstructor's attribute surface (#117).

`tomographicReconstructor` used to forward any unknown attribute across five parameter
objects by linear search, and `__setattr__` fell through to `object.__setattr__` when
nothing matched -- so a misspelled parameter silently became a new attribute and the
reconstructor went on to build with the old value.
"""

import logging

import numpy as np
import pytest

from pyTomoAO import reconstructor as reconstructor_module

logger = logging.getLogger(__name__)


@pytest.fixture
def rec(revolt_config):
    return reconstructor_module.tomographicReconstructor(revolt_config, force_cpu=True)


class TestForwardedParameters:
    """The four names reachable directly on the reconstructor."""

    def test_reads_delegate_to_the_owning_parameter_object(self, rec):
        assert rec.nLGS == rec.lgsAsterismParams.nLGS
        assert rec.r0 == rec.atmParams.r0
        assert rec.r0_zenith == rec.atmParams.r0_zenith
        assert rec.L0 == rec.atmParams.L0

    def test_r0_zenith_is_writable_and_r0_follows(self, rec):
        rec.r0_zenith = 0.1
        assert rec.atmParams.r0_zenith == 0.1
        # r0 is derived from r0_zenith and the zenith angle.
        assert rec.r0 == pytest.approx(rec.atmParams.r0)

    def test_L0_is_writable(self, rec):
        rec.L0 = 42.0
        assert rec.atmParams.L0 == 42.0

    def test_r0_is_read_only(self, rec):
        """It is derived; writing it would silently do nothing useful."""
        with pytest.raises(AttributeError):
            rec.r0 = 0.2

    def test_setting_nLGS_updates_both_parameter_objects_that_track_it(self, rec):
        """Only the asterism and WFS parameters carry nLGS.

        The suite used to assert that `tomoParams.nLGS` was updated too, but that ran
        against a `MagicMock`, which answers to any attribute. `tomographyParameters` has
        no `nLGS` and never did.
        """
        assert not hasattr(rec.tomoParams, "nLGS")

        rec.nLGS = 3
        assert rec.nLGS == 3
        assert rec.lgsAsterismParams.nLGS == 3
        assert rec.lgsWfsParams.nLGS == 3

    def test_setting_nLGS_resizes_the_per_sensor_arrays(self, rec):
        rec.nLGS = 4
        assert rec.lgsWfsParams.wfsLensletsRotation.shape == (4,)
        assert rec.lgsWfsParams.wfsLensletsOffset.shape == (2, 4)

    def test_negative_nLGS_is_rejected(self, rec):
        with pytest.raises(ValueError, match="non-negative"):
            rec.nLGS = -1


class TestUnknownAttributes:
    """The point of the change: mistakes are reported rather than absorbed."""

    def test_assigning_an_unknown_name_raises(self, rec):
        with pytest.raises(AttributeError):
            rec.r0_zenit = 0.1  # note the typo

    def test_a_swallowed_typo_no_longer_leaves_the_value_stale(self, rec):
        before = rec.r0_zenith
        with pytest.raises(AttributeError):
            rec.r0_zenth = 0.05
        assert rec.r0_zenith == before

    def test_reading_an_unknown_name_raises(self, rec):
        with pytest.raises(AttributeError):
            _ = rec.no_such_parameter

    def test_parameters_not_forwarded_live_on_their_owner(self, rec):
        """The explicit path replaces the search, and is where the value actually lives."""
        assert not hasattr(rec, "altitude")
        assert rec.atmParams.altitude is not None
        assert not hasattr(rec, "nValidSubap")
        assert rec.lgsWfsParams.nValidSubap > 0


class TestIntrospection:
    def test_public_attributes_are_discoverable(self, rec):
        names = set(dir(rec))
        for expected in ("nLGS", "r0", "r0_zenith", "L0", "reconstructor", "backend", "FR"):
            assert expected in names, f"{expected} missing from dir()"

    def test_documented_state_is_declared(self, rec):
        for expected in ("atmParams", "lgsWfsParams", "Gamma", "Cxx", "method"):
            assert hasattr(rec, expected), f"{expected} is not an attribute"

    def test_no_instance_dict(self, rec):
        """__slots__ is what makes an unknown assignment raise."""
        assert not hasattr(rec, "__dict__")


class TestStateStillWorks:
    """Guard against the declaration missing something build_reconstructor assigns."""

    def test_build_populates_the_intermediate_matrices(self, rec):
        rec.build_reconstructor()
        assert rec.method == "Model"
        for name in ("Gamma", "Cxx", "Cox", "CnZ", "RecStatSA"):
            assert getattr(rec, name) is not None, f"{name} was not populated"
        assert isinstance(rec.reconstructor, np.ndarray)
