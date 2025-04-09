import pytest
import numpy as np
from io import StringIO
import sys
from contextlib import redirect_stdout
import logging

# Import the class - assuming it's in a file called deformable_mirror.py
# Modify this import if your file has a different name
from pyTomoAO.dmParametersClass import dmParameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to get INFO messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("")

logger.info("#### Starting tests for dmParametersClass ####")

# Valid configuration for reuse in tests
@pytest.fixture
def valid_config():
    return {
        "dm_parameters": {
            "dmHeights": [0.0, 0.001, 0.002],
            "dmPitch": [0.5, 0.5, 0.5],
            "dmCrossCoupling": 0.15,
            "nActuators": [20, 20, 20],
            "validActuators": [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1]
            ]
        }
    }

class TestDmParameters:
    
    def test_initialization_with_valid_config(self, valid_config):
        """Test that the class initializes correctly with valid config"""
        logger.info("Starting initialization test")
        params = dmParameters(valid_config)
        assert isinstance(params, dmParameters)
        assert np.array_equal(params.dmHeights, np.array([0.0, 0.001, 0.002]))
        assert np.array_equal(params.dmPitch, np.array([0.5, 0.5, 0.5]))
        assert params.dmCrossCoupling == 0.15
        assert np.array_equal(params.nActuators, np.array([20, 20, 20]))
        
        expected_valid = np.array([
            [True, False, True, False],
            [False, True, False, True],
            [True, False, True, False],
            [False, True, False, True]
        ])
        assert np.array_equal(params.validActuators, expected_valid)
        logger.info("✅ Initialization test passed")

    def test_dmHeights_validation(self, valid_config):
        """Test that dmHeights validates inputs correctly"""
        logger.info("Starting dmHeights validation test")
        # Test negative values
        config = valid_config.copy()
        config["dm_parameters"] = valid_config["dm_parameters"].copy()
        config["dm_parameters"]["dmHeights"] = [-1.0, 0.001, 0.002]
        
        with pytest.raises(ValueError, match="All DM heights must be non-negative"):
            dmParameters(config)
            
        # Test conversion from list to array
        config["dm_parameters"]["dmHeights"] = [0.5, 1.0]
        params = dmParameters(config)
        assert isinstance(params.dmHeights, np.ndarray)
        assert np.array_equal(params.dmHeights, np.array([0.5, 1.0]))
        
        # Test empty array
        config["dm_parameters"]["dmHeights"] = []
        with pytest.raises(ValueError, match="dmHeights cannot be empty"):
            dmParameters(config)
        logger.info("✅ dmHeights validation test passed")

    def test_dmPitch_validation(self, valid_config):
        """Test that dmPitch validates inputs correctly"""
        logger.info("Starting dmPitch validation test")
        # Test zero values
        config = valid_config.copy()
        config["dm_parameters"] = valid_config["dm_parameters"].copy()
        config["dm_parameters"]["dmPitch"] = [0.0, 0.5, 0.5]
        
        with pytest.raises(ValueError, match="All pitch values must be positive"):
            dmParameters(config)
            
        # Test negative values
        config["dm_parameters"]["dmPitch"] = [-0.5, 0.5, 0.5]
        
        with pytest.raises(ValueError, match="All pitch values must be positive"):
            dmParameters(config)
            
        # Test conversion to float
        config["dm_parameters"]["dmPitch"] = [1, 2, 3]  # Integers
        params = dmParameters(config)
        assert np.array_equal(params.dmPitch, np.array([1.0, 2.0, 3.0]))
        logger.info("✅ dmPitch validation test passed")

    def test_dmCrossCoupling_validation(self, valid_config):
        """Test that dmCrossCoupling validates inputs correctly"""
        logger.info("Starting dmCrossCoupling validation test")
        # Test values outside 0-1 range
        config = valid_config.copy()
        config["dm_parameters"] = valid_config["dm_parameters"].copy()
        
        # Test negative value
        config["dm_parameters"]["dmCrossCoupling"] = -0.1
        with pytest.raises(ValueError, match="Cross-coupling must be between 0 and 1"):
            dmParameters(config)
            
        # Test value > 1
        config["dm_parameters"]["dmCrossCoupling"] = 1.5
        with pytest.raises(ValueError, match="Cross-coupling must be between 0 and 1"):
            dmParameters(config)
            
        # Test non-numeric value
        config["dm_parameters"]["dmCrossCoupling"] = "invalid"
        with pytest.raises(TypeError, match="Cross-coupling must be numeric"):
            dmParameters(config)
            
        # Test boundary values
        config["dm_parameters"]["dmCrossCoupling"] = 0
        params = dmParameters(config)
        assert params.dmCrossCoupling == 0.0
        
        config["dm_parameters"]["dmCrossCoupling"] = 1
        params = dmParameters(config)
        assert params.dmCrossCoupling == 1.0
        logger.info("✅ dmCrossCoupling validation test passed")

    def test_nActuators_validation(self, valid_config):
        """Test that nActuators validates inputs correctly"""
        logger.info("Starting nActuators validation test")
        # Test negative values
        config = valid_config.copy()
        config["dm_parameters"] = valid_config["dm_parameters"].copy()
        config["dm_parameters"]["nActuators"] = [-1, 20, 20]
        
        with pytest.raises(ValueError, match="Actuator counts cannot be negative"):
            dmParameters(config)
            
        # Test conversion to integer
        config["dm_parameters"]["nActuators"] = [10.5, 20.7, 30.2]
        params = dmParameters(config)
        assert np.array_equal(params.nActuators, np.array([10, 20, 30]))
        logger.info("✅ nActuators validation test passed")
    
    def test_validActuators_list_validation(self, valid_config):
        """Test that validActuators_list validates inputs correctly"""
        logger.info("Starting validActuators_list validation test")
        config = valid_config.copy()
        config["dm_parameters"] = valid_config["dm_parameters"].copy()
        
        # Test non-2D array
        config["dm_parameters"]["validActuators"] = [1, 0, 1, 0]
        with pytest.raises(ValueError, match="Actuator map must be 2D"):
            dmParameters(config)
            
        # Test 3D array which will fail the ndim check
        # This will create a 2x2x2 array which is 3D and should fail
        config["dm_parameters"]["validActuators"] = [[[1, 0], [0, 1]], [[0, 1], [1, 0]]]
        with pytest.raises(ValueError, match="Actuator map must be 2D"):
            dmParameters(config)
        logger.info("✅ validActuators_list validation test passed")

    def test_nActuatorsSupport(self, valid_config):
        """Test calculation of nActuatorsSupport"""
        logger.info("Starting nActuatorsSupport calculation test")
        params = dmParameters(valid_config)
        assert np.array_equal(params.nActuatorsSupport, np.array([24, 24, 24]))
        
        # Test with different nActuators
        config = valid_config.copy()
        config["dm_parameters"] = valid_config["dm_parameters"].copy()
        config["dm_parameters"]["nActuators"] = [10, 15, 20]
        params = dmParameters(config)
        assert np.array_equal(params.nActuatorsSupport, np.array([14, 19, 24]))
        logger.info("✅ nActuatorsSupport calculation test passed")

    def test_validActuatorsSupport(self, valid_config):
        """Test calculation of validActuatorsSupport (padded array)"""
        logger.info("Starting validActuatorsSupport calculation test")
        params = dmParameters(valid_config)
        
        # The original valid actuators
        original = np.array([
            [True, False, True, False],
            [False, True, False, True],
            [True, False, True, False],
            [False, True, False, True]
        ])
        
        # Expected padded array (zeros/False around the original)
        expected = np.zeros((8, 8), dtype=bool)
        expected[2:6, 2:6] = original
        
        assert np.array_equal(params.validActuatorsSupport, expected)
        logger.info("✅ validActuatorsSupport calculation test passed")

    def test_convert_to_array_helper(self, valid_config):
        """Test the _convert_to_array helper method"""
        logger.info("Starting _convert_to_array helper method test")
        params = dmParameters(valid_config)
        
        # Test with list
        result = params._convert_to_array([1, 2, 3], "test", float)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))
        
        # Test with numpy array
        result = params._convert_to_array(np.array([1, 2, 3]), "test", float)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))
        
        # Test with invalid type
        with pytest.raises(TypeError, match="test must be list/array"):
            params._convert_to_array("invalid", "test", float)
            
        # Test with empty list
        with pytest.raises(ValueError, match="test cannot be empty"):
            params._convert_to_array([], "test", float)
        logger.info("✅ _convert_to_array helper method test passed")

    def test_str_representation(self, valid_config):
        """Test the string representation of the class"""
        logger.info("Starting string representation test")
        params = dmParameters(valid_config)
        
        # Capture stdout
        f = StringIO()
        with redirect_stdout(f):
            print(params)
            
        output = f.getvalue()
        
        # Check that the string contains key information
        assert "Deformable Mirror Parameters:" in output
        assert "Actuator Grid: [20 20 20]" in output
        assert "Support Grid: [24 24 24]" in output
        assert "Cross-Coupling: 15.0%" in output
        assert "Valid Actuators: 8/16" in output
        logger.info("✅ String representation test passed")

    def test_format_array_stats(self, valid_config):
        """Test the _format_array_stats helper method"""
        logger.info("Starting _format_array_stats helper method test")
        params = dmParameters(valid_config)
        
        # Test uniform array
        uniform = np.array([1.0, 1.0, 1.0])
        result = params._format_array_stats(uniform, unit="m")
        assert result == "1.000 m (uniform)"
        
        # Test non-uniform array
        non_uniform = np.array([1.0, 2.0, 3.0])
        result = params._format_array_stats(non_uniform, unit="m")
        assert result == "1.000-3.000 m (mean: 2.000)"
        logger.info("✅ _format_array_stats helper method test passed")