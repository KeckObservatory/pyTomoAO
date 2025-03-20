import pytest
import numpy as np
import math
from unittest.mock import Mock, patch
import logging

# Import the class to test
from pyTomoAO.lgsAsterismParametersClass import lgsAsterismParameters
from pyTomoAO.atmosphereParametersClass import atmosphereParameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to get INFO messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("")

logger.info("#### Starting tests for lgsAsterismParametersClass ####")


class TestLgsAsterismParametersClass:
    
    @pytest.fixture
    def valid_config(self):
        """Returns a valid configuration dictionary for testing."""
        return {
            "lgs_asterism": {
                "radiusAst": 30.0,       # arcseconds
                "LGSwavelength": 589e-9, # meters (sodium wavelength)
                "baseLGSHeight": 90000,  # meters (90km)
                "nLGS": 4
            }
        }
    
    @pytest.fixture
    def mock_atm_params(self):
        """Creates a mock atmosphereParameters object."""
        mock_atm = Mock(spec=atmosphereParameters)
        mock_atm.airmass = 1.2
        return mock_atm
    
    @pytest.fixture
    def lgs_params(self, valid_config, mock_atm_params):
        """Creates a valid lgsAsterismParameters instance for testing."""
        return lgsAsterismParameters(valid_config, mock_atm_params)
    
    def test_initialization(self, valid_config, mock_atm_params):
        """Test successful initialization with valid parameters."""
        logger.info("Starting initialization test")
        lgs_params = lgsAsterismParameters(valid_config, mock_atm_params)
        
        assert lgs_params.radiusAst == 30.0
        assert lgs_params.LGSwavelength == 589e-9
        assert lgs_params.baseLGSHeight == 90000
        assert lgs_params.nLGS == 4
        assert lgs_params.atmospheric_parameters == mock_atm_params
        logger.info("✅ Initialization test passed")
    
    def test_property_validations(self, valid_config, mock_atm_params):
        """Test property validations for input parameters."""
        logger.info("Starting property validation test")
        lgs_params = lgsAsterismParameters(valid_config, mock_atm_params)
        
        # Test radiusAst validation
        with pytest.raises(TypeError):
            lgs_params.radiusAst = "invalid"
        with pytest.raises(ValueError):
            lgs_params.radiusAst = -10
            
        # Test LGSwavelength validation
        with pytest.raises(TypeError):
            lgs_params.LGSwavelength = "invalid"
        with pytest.raises(ValueError):
            lgs_params.LGSwavelength = 0
        with pytest.raises(ValueError):
            lgs_params.LGSwavelength = -500e-9
            
        # Test baseLGSHeight validation
        with pytest.raises(TypeError):
            lgs_params.baseLGSHeight = "invalid"
        with pytest.raises(ValueError):
            lgs_params.baseLGSHeight = 0
        with pytest.raises(ValueError):
            lgs_params.baseLGSHeight = -1000
            
        # Test nLGS validation
        with pytest.raises(TypeError):
            lgs_params.nLGS = 3.5
        with pytest.raises(ValueError):
            lgs_params.nLGS = -1
        
        logger.info("✅ Property validation test passed")
    
    def test_lgs_height_calculation(self, valid_config, mock_atm_params):
        """Test LGSheight calculation using airmass."""
        logger.info("Starting LGS height calculation test")
        mock_atm_params.airmass = 1.5
        lgs_params = lgsAsterismParameters(valid_config, mock_atm_params)
        
        expected_height = valid_config["lgs_asterism"]["baseLGSHeight"] * 1.5
        assert lgs_params.LGSheight == expected_height
        
        # Verify recalculation with changed airmass
        mock_atm_params.airmass = 2.0
        expected_height = valid_config["lgs_asterism"]["baseLGSHeight"] * 2.0
        assert lgs_params.LGSheight == expected_height
        logger.info("✅ LGS height calculation test passed")
    
    def test_lgs_directions(self, lgs_params):
        """Test LGSdirections calculation."""
        logger.info("Starting LGS directions calculation test")
        arcsec_to_rad = math.pi / (180 * 3600)
        expected_radius = lgs_params.radiusAst * arcsec_to_rad
        
        directions = lgs_params.LGSdirections
        
        # Check array shape
        assert directions.shape == (4, 2)
        
        # Check radius values (all should be the same)
        for i in range(lgs_params.nLGS):
            assert np.isclose(directions[i, 0], expected_radius)
        
        # Check angular spacing (should be evenly spaced)
        azimuth_angles = [math.degrees(directions[i, 1]) for i in range(lgs_params.nLGS)]
        for i in range(lgs_params.nLGS):
            expected_angle = i * 360 / lgs_params.nLGS
            assert np.isclose(azimuth_angles[i], expected_angle)
        
        logger.info("✅ LGS directions calculation test passed")
    
    def test_direction_vector_lgs(self, lgs_params):
        """Test directionVectorLGS calculation."""
        logger.info("Starting direction vector LGS calculation test")
        vectors = lgs_params.directionVectorLGS
        
        # Check array shape
        assert vectors.shape == (3, 4)
        
        # All z-components should be 1.0
        assert np.all(vectors[2, :] == 1.0)
        
        # Test x and y components calculation from zenith and azimuth
        for i in range(lgs_params.nLGS):
            zenith = lgs_params.LGSdirections[i, 0]
            azimuth = lgs_params.LGSdirections[i, 1]
            
            expected_x = math.tan(zenith) * math.cos(azimuth)
            expected_y = math.tan(zenith) * math.sin(azimuth)
            
            assert np.isclose(vectors[0, i], expected_x)
            assert np.isclose(vectors[1, i], expected_y)
        
        logger.info("✅ Direction vector LGS calculation test passed")
    
    def test_different_number_of_lgs(self, valid_config, mock_atm_params):
        """Test with different numbers of LGS."""
        logger.info("Starting different number of LGS test")
        for n_lgs in [1, 3, 6]:
            modified_config = valid_config.copy()
            modified_config["lgs_asterism"] = valid_config["lgs_asterism"].copy()
            modified_config["lgs_asterism"]["nLGS"] = n_lgs
            
            lgs_params = lgsAsterismParameters(modified_config, mock_atm_params)
            
            # Check LGSdirections shape
            assert lgs_params.LGSdirections.shape == (n_lgs, 2)
            
            # Check directionVectorLGS shape
            assert lgs_params.directionVectorLGS.shape == (3, n_lgs)
        
        logger.info("✅ Different number of LGS test passed")
    
    def test_string_representation(self, valid_config, mock_atm_params):
        """Test the string representation of the object."""
        logger.info("Starting string representation test")
        lgs_params = lgsAsterismParameters(valid_config, mock_atm_params)
        
        string_repr = str(lgs_params)
        
        # Check that important information is included in string representation
        assert "LGS Asterism Parameters" in string_repr
        assert f"Number of LGS: {lgs_params.nLGS}" in string_repr
        assert f"Base Radius: {lgs_params.radiusAst:.2f}" in string_repr
        assert f"Wavelength:" in string_repr
        assert f"{lgs_params.LGSwavelength * 1e9:.1f} nm" in string_repr
        assert f"Base Height: {lgs_params.baseLGSHeight/1000:.1f} km" in string_repr
        assert f"Current Airmass: {mock_atm_params.airmass:.2f}" in string_repr
        assert f"Effective Height: {lgs_params.LGSheight/1000:.1f} km" in string_repr
        assert "Direction Vectors" in string_repr
        logger.info("✅ String representation test passed")
    
    def test_wavelength_formatting(self, lgs_params):
        """Test the wavelength formatting helper method."""
        logger.info("Starting wavelength formatting test")
        formatted = lgs_params._format_wavelength()
        
        wavelength_nm = lgs_params.LGSwavelength * 1e9
        expected = f"{wavelength_nm:.1f} nm ({lgs_params.LGSwavelength:.2e} m)"
        
        assert formatted == expected
        logger.info("✅ Wavelength formatting test passed")