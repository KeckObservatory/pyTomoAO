import pytest
import numpy as np
from unittest.mock import MagicMock
from pyTomoAO.lgsWfsParametersClass import lgsWfsParameters
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to get INFO messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("")


logger.info("#### Starting tests for lgsWfsParametersClass ####")
# Mock the dependencies
class MockLgsAsterismParameters:
    def __init__(self, nLGS=4):
        self.nLGS = nLGS

@pytest.fixture
def default_config():
    """Fixture providing a default valid configuration"""
    return {
        "lgs_wfs_parameters": {
            "D": 8.2,
            "nLenslet": 40,
            "nPx": 16,
            "fieldStopSize": 2.5,
            "validLLMap": [
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1]
            ],
            "validActuatorMap": [
                [1, 1, 0],
                [0, 1, 1],
                [1, 0, 1]
            ],
            "wfsLensletsRotation": [0, 1, 0, 0],
            "wfsLensletsOffset": [[0.1, -0.1, -0.1, 0.1], [0.1, 0.1, -0.1, -0.1]]
        }
    }

@pytest.fixture
def lgs_asterism_params():
    """Fixture providing mock lgsAsterismParameters"""
    return MockLgsAsterismParameters(nLGS=4)

class TestLgsWfsParametersClass:
    
    def test_initialization(self, default_config, lgs_asterism_params):
        """Test successful initialization with valid parameters"""
        logger.info("Testing initialization with valid parameters")
        params = lgsWfsParameters(default_config, lgs_asterism_params)
        
        # Check core properties
        assert params.D == 8.2
        assert params.nLenslet == 40
        assert params.nPx == 16
        assert params.fieldStopSize == 2.5
        assert params.nLGS == 4
        
        # Check array properties
        assert len(params.validLLMap_list) == 3
        assert len(params.validActuatorMap_list) == 3
        np.testing.assert_array_equal(params.wfsLensletsRotation, np.array([0, 1, 0, 0]))
        np.testing.assert_array_equal(params.wfsLensletsOffset, 
                                        np.array([[0.1, -0.1, -0.1, 0.1], [0.1, 0.1, -0.1, -0.1]]))
        logger.info("✅ Initialization test passed")
    
    def test_d_validation(self, default_config, lgs_asterism_params):
        """Test validation of telescope diameter"""
        logger.info("Testing validation of telescope diameter")
        # Test invalid types
        config = default_config.copy()
        config["lgs_wfs_parameters"]["D"] = "invalid"
        with pytest.raises(TypeError, match="Telescope diameter must be numeric"):
            lgsWfsParameters(config, lgs_asterism_params)
        
        # Test invalid values
        config["lgs_wfs_parameters"]["D"] = -5.0
        with pytest.raises(ValueError, match="Telescope diameter must be positive"):
            lgsWfsParameters(config, lgs_asterism_params)
        
        config["lgs_wfs_parameters"]["D"] = 0
        with pytest.raises(ValueError, match="Telescope diameter must be positive"):
            lgsWfsParameters(config, lgs_asterism_params)
        
        logger.info("✅ Telescope diameter validation test passed")
    
    def test_nlenslet_validation(self, default_config, lgs_asterism_params):
        """Test validation of number of lenslets"""
        logger.info("Testing validation of number of lenslets")
        # Test invalid types
        config = default_config.copy()
        config["lgs_wfs_parameters"]["nLenslet"] = 40.5
        with pytest.raises(TypeError, match="Lenslet count must be integer"):
            lgsWfsParameters(config, lgs_asterism_params)
        
        # Test invalid values
        config["lgs_wfs_parameters"]["nLenslet"] = 0
        with pytest.raises(ValueError, match="Lenslet count must be positive"):
            lgsWfsParameters(config, lgs_asterism_params)
        
        config["lgs_wfs_parameters"]["nLenslet"] = -10
        with pytest.raises(ValueError, match="Lenslet count must be positive"):
            lgsWfsParameters(config, lgs_asterism_params)
        
        logger.info("✅ Number of lenslets validation test passed")
    
    def test_npx_validation(self, default_config, lgs_asterism_params):
        """Test validation of pixels per lenslet"""
        logger.info("Testing validation of pixels per lenslet")
        # Test invalid types
        config = default_config.copy()
        config["lgs_wfs_parameters"]["nPx"] = 16.5
        with pytest.raises(TypeError, match="Pixel count must be integer"):
            lgsWfsParameters(config, lgs_asterism_params)
        
        # Test invalid values
        config["lgs_wfs_parameters"]["nPx"] = 0
        with pytest.raises(ValueError, match="Pixel count must be positive"):
            lgsWfsParameters(config, lgs_asterism_params)
        
        config["lgs_wfs_parameters"]["nPx"] = -10
        with pytest.raises(ValueError, match="Pixel count must be positive"):
            lgsWfsParameters(config, lgs_asterism_params)
        
        logger.info("✅ Pixels per lenslet validation test passed")
    
    def test_field_stop_validation(self, default_config, lgs_asterism_params):
        """Test validation of field stop size"""
        logger.info("Testing validation of field stop size")
        # Test invalid types
        config = default_config.copy()
        config["lgs_wfs_parameters"]["fieldStopSize"] = "invalid"
        with pytest.raises(TypeError, match="Field stop size must be numeric"):
            lgsWfsParameters(config, lgs_asterism_params)
        
        # Test invalid values
        config["lgs_wfs_parameters"]["fieldStopSize"] = 0
        with pytest.raises(ValueError, match="Field stop size must be positive"):
            lgsWfsParameters(config, lgs_asterism_params)
        
        config["lgs_wfs_parameters"]["fieldStopSize"] = -2.5
        with pytest.raises(ValueError, match="Field stop size must be positive"):
            lgsWfsParameters(config, lgs_asterism_params)
        
        logger.info("✅ Field stop size validation test passed")
    
    def test_nlgs_validation(self, default_config, lgs_asterism_params):
        """Test validation of number of laser guide stars"""
        logger.info("Testing validation of number of laser guide stars")
        # Create the instance first
        params = lgsWfsParameters(default_config, lgs_asterism_params)
        
        # Test setting invalid type
        with pytest.raises(TypeError, match="LGS count must be integer"):
            params.nLGS = 4.5
        
        # Test setting invalid value
        with pytest.raises(ValueError, match="LGS count cannot be negative"):
            params.nLGS = -1
        
        # Test setting to 0 (should be allowed)
        params.nLGS = 0
        assert params.nLGS == 0
        logger.info("✅ Number of laser guide stars validation test passed")
    
    def test_map_validation(self, default_config, lgs_asterism_params):
        """Test validation of valid lenslet and actuator maps"""
        logger.info("Testing validation of valid lenslet and actuator maps")
        # Test invalid lenslet map (not 2D)
        config = default_config.copy()
        config["lgs_wfs_parameters"] = default_config["lgs_wfs_parameters"].copy()
        config["lgs_wfs_parameters"]["validLLMap"] = [1, 0, 1]
        with pytest.raises(ValueError) as excinfo:
            lgsWfsParameters(config, lgs_asterism_params)
        assert "Invalid lenslet map: Lenslet map must be 2D" in str(excinfo.value)
        
        # Test invalid actuator map (not 2D) - need to test separately with valid lenslet map
        config = default_config.copy()
        config["lgs_wfs_parameters"] = default_config["lgs_wfs_parameters"].copy()
        # Keep a valid lenslet map and only make the actuator map invalid
        config["lgs_wfs_parameters"]["validActuatorMap"] = [1, 0, 1]
        with pytest.raises(ValueError) as excinfo:
            lgsWfsParameters(config, lgs_asterism_params)
        assert "Invalid actuator map" in str(excinfo.value)
        logger.info("✅ Valid lenslet and actuator maps validation test passed")
    
    def test_wfs_lenslets_rotation_validation(self, default_config, lgs_asterism_params):
        """Test validation of WFS lenslets rotation"""
        logger.info("Testing validation of WFS lenslets rotation")
        # Test with incorrect length
        config = default_config.copy()
        config["lgs_wfs_parameters"]["wfsLensletsRotation"] = [0, 1, 0]  # Only 3 elements for 4 LGS
        with pytest.raises(ValueError, match="wfsLensletsRotation length .* must match nLGS"):
            lgsWfsParameters(config, lgs_asterism_params)
        
        # Test with incorrect dimensionality
        config["lgs_wfs_parameters"]["wfsLensletsRotation"] = [[0, 1], [0, 0]]
        with pytest.raises(ValueError, match="wfsLensletsRotation must be 1D array"):
            lgsWfsParameters(config, lgs_asterism_params)
        
        # Test with None (should default to zeros)
        config["lgs_wfs_parameters"]["wfsLensletsRotation"] = None
        params = lgsWfsParameters(config, lgs_asterism_params)
        np.testing.assert_array_equal(params.wfsLensletsRotation, np.zeros(4))
        logger.info("✅ WFS lenslets rotation validation test passed")
    
    def test_wfs_lenslets_offset_validation(self, default_config, lgs_asterism_params):
        """Test validation of WFS lenslets offset"""
        logger.info("Testing validation of WFS lenslets offset")
        # Test with incorrect shape
        config = default_config.copy()
        config["lgs_wfs_parameters"]["wfsLensletsOffset"] = [[0.1, -0.1, -0.1]]  # Only 3 elements for 4 LGS
        with pytest.raises(ValueError, match="wfsLensletsOffset length .* must match nLGS"):
            lgsWfsParameters(config, lgs_asterism_params)
        
        # Test with incorrect dimensionality
        config["lgs_wfs_parameters"]["wfsLensletsOffset"] = [0.1, 0.1, -0.1, -0.1]
        with pytest.raises(ValueError, match="wfsLensletsOffset must be 2D array"):
            lgsWfsParameters(config, lgs_asterism_params)
        
        # Test with None (should default to zeros)
        config["lgs_wfs_parameters"]["wfsLensletsOffset"] = None
        params = lgsWfsParameters(config, lgs_asterism_params)
        np.testing.assert_array_equal(params.wfsLensletsOffset, np.zeros((2, 4)))
        logger.info("✅ WFS lenslets offset validation test passed")
    
    def test_computed_properties(self, default_config, lgs_asterism_params):
        """Test computed properties like nValidSubap, validLLMapSupport, DSupport"""
        logger.info("Testing computed properties")
        params = lgsWfsParameters(default_config, lgs_asterism_params)
        
        # Test nValidSubap
        # Count the number of True values in the validLLMap
        # Default map is [[1, 0, 1], [0, 1, 0], [1, 0, 1]] which has 5 True values
        assert params.nValidSubap == 5  # Based on the actual count in validLLMap
        
        # Test validLLMapSupport (padded map)
        padded_map = params.validLLMapSupport
        # The original map is 3x3, and validLLMapSupport adds padding of 2 on each side
        # So the padded shape should be 3+2+2=7 in each dimension
        assert padded_map.shape == (7, 7)
        
        # Test DSupport
        # DSupport = D * validLLMapSupport.shape[0] / nLenslet
        expected_dsupport = 8.2 * 7 / 40  # D * padded_shape / nLenslet
        assert params.DSupport == pytest.approx(expected_dsupport)
        logger.info("✅ Computed properties test passed")
    
    def test_nlgs_auto_adjustment(self, default_config, lgs_asterism_params):
        """Test auto-adjustment of arrays when nLGS changes"""
        logger.info("Testing auto-adjustment of arrays when nLGS changes")
        params = lgsWfsParameters(default_config, lgs_asterism_params)
        
        # Initial state
        assert params.nLGS == 4
        assert params.wfsLensletsRotation.shape == (4,)
        assert params.wfsLensletsOffset.shape == (2, 4)
        
        # Increase nLGS
        params.nLGS = 6
        assert params.nLGS == 6
        assert params.wfsLensletsRotation.shape == (6,)
        assert params.wfsLensletsOffset.shape == (2, 6)
        
        # Check that original values are preserved and new ones are zeroed
        np.testing.assert_array_equal(params.wfsLensletsRotation[:4], np.array([0, 1, 0, 0]))
        np.testing.assert_array_equal(params.wfsLensletsRotation[4:], np.array([0, 0]))
        
        # Decrease nLGS
        params.nLGS = 2
        assert params.nLGS == 2
        assert params.wfsLensletsRotation.shape == (2,)
        assert params.wfsLensletsOffset.shape == (2, 2)
        
        # Check that truncation preserved the right values
        np.testing.assert_array_equal(params.wfsLensletsRotation, np.array([0, 1]))
        logger.info("✅ Auto-adjustment of arrays when nLGS changes test passed")
    
    def test_map_setters(self, default_config, lgs_asterism_params):
        """Test the numpy array setters for maps"""
        logger.info("Testing numpy array setters for maps")
        params = lgsWfsParameters(default_config, lgs_asterism_params)
        
        # Test setting validLLMap with numpy array
        new_map = np.array([[1, 1], [1, 0]])
        params.validLLMap = new_map
        np.testing.assert_array_equal(np.array(params.validLLMap_list), new_map)
        np.testing.assert_array_equal(params.validLLMap, new_map)
        
        # Test validation of numpy array input
        with pytest.raises(TypeError, match="validLLMap must be a numpy array"):
            params.validLLMap = "invalid"
        
        with pytest.raises(ValueError, match="validLLMap must be 2D"):
            params.validLLMap = np.array([1, 0, 1])
        
        logger.info("✅ Numpy array setters for maps test passed")
    
    def test_str_representation(self, default_config, lgs_asterism_params):
        """Test string representation of the class"""
        logger.info("Testing string representation of the class")
        params = lgsWfsParameters(default_config, lgs_asterism_params)
        str_repr = str(params)
        
        # Check that key info is in the string representation
        assert "Telescope Diameter: 8.20 m" in str_repr
        assert "Lenslet Array: 40x40" in str_repr
        assert "Pixels per Lenslet: 16" in str_repr
        assert "Field Stop: 2.50 arcsec" in str_repr
        assert "Number of LGS: 4" in str_repr
        assert "Valid Elements: 5/9" in str_repr  # 5 valid out of 3x3=9
        logger.info("✅ String representation test passed")
    
    def test_default_parameters(self, lgs_asterism_params):
        """Test initialization with minimal parameters and defaults"""
        logger.info("Testing initialization with minimal parameters and defaults")
        config = {
            "lgs_wfs_parameters": {
                "D": 8.2,
                "nLenslet": 40,
                "nPx": 16,
                "fieldStopSize": 2.5,
                "validLLMap": [
                    [1, 0, 1],
                    [0, 1, 0],
                    [1, 0, 1]
                ],
                "validActuatorMap": [
                    [1, 1, 0],
                    [0, 1, 1],
                    [1, 0, 1]
                ]
                # No wfsLensletsRotation or wfsLensletsOffset
            }
        }
        
        params = lgsWfsParameters(config, lgs_asterism_params)
        
        # Check that defaults were applied
        np.testing.assert_array_equal(params.wfsLensletsRotation, np.zeros(4))
        np.testing.assert_array_equal(params.wfsLensletsOffset, np.zeros((2, 4)))
        logger.info("✅ Initialization with minimal parameters and defaults test passed")