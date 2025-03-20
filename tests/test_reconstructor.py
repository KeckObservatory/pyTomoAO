"""
To run the tests in this file using pytest, navigate to the repository:

    cd /path/to/pyTomoAO

Execute the following command in your terminal:

    pytest tests/test_reconstructor.py

Ensure that you have pytest installed in your environment. You can install it via pip if necessary:

    pip install pytest
"""

import pytest
import numpy as np
import logging
from pyTomoAO.TomographicReconstructor import TomographicReconstructor

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def reconstructor():
    """
    Fixture to create a TomographicReconstructor instance with mock parameters.
    """
    logger.debug("Setting up the TomographicReconstructor fixture.")
    # Create mock parameter objects with necessary attributes
    tomoParams = MockTomoParams()
    lgsWfsParams = MockLgsWfsParams()
    atmParams = MockAtmParams()
    lgsAsterismParams = MockLgsAsterismParams()

    # Initialize the TomographicReconstructor
    reconstructor = TomographicReconstructor(
        tomoParams, 
        atmParams, 
        lgsWfsParams, 
        lgsAsterismParams
    )
    logger.debug("TomographicReconstructor fixture created.")
    return reconstructor

def test_initialization(reconstructor):
    """
    Test the initialization of the TomographicReconstructor.
    """
    logger.info("Testing initialization of TomographicReconstructor.")
    assert isinstance(reconstructor, TomographicReconstructor)
    assert reconstructor.reconstructor.shape == (0,)  # Initially empty
    logger.info("Initialization test passed.")

def test_build_reconstructor(reconstructor):
    """
    Test the build_reconstructor method.
    """
    logger.info("Testing build_reconstructor method.")
    reconstructor.build_reconstructor()
    assert reconstructor.reconstructor.shape == (10, 10)  # Example shape
    logger.info("build_reconstructor test passed.")

def test_reconstructor_property(reconstructor):
    """
    Test the reconstructor property getter and setter.
    """
    logger.info("Testing reconstructor property getter and setter.")
    new_reconstructor = np.ones((10, 10), dtype=np.float32)
    reconstructor.reconstructor = new_reconstructor
    np.testing.assert_array_equal(reconstructor.reconstructor, new_reconstructor)
    logger.info("reconstructor property test passed.")

def test_invalid_reconstructor_assignment(reconstructor):
    """
    Test setting an invalid reconstructor value.
    """
    logger.info("Testing invalid reconstructor assignment.")
    with pytest.raises(ValueError):
        reconstructor.reconstructor = np.array([1, 2, 3])  # Not a 2D array
    logger.info("Invalid reconstructor assignment test passed.")

def test_R_property(reconstructor):
    """
    Test the R property getter and setter.
    """
    logger.info("Testing R property getter and setter.")
    new_R = np.ones((10, 10), dtype=np.float32)
    reconstructor.R = new_R
    np.testing.assert_array_equal(reconstructor.R, new_R)
    logger.info("R property test passed.")

# Mock classes for parameter objects
class MockTomoParams:
    pass

class MockLgsWfsParams:
    pass

class MockAtmParams:
    pass

class MockLgsAsterismParams:
    pass