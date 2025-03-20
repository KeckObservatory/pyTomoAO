import pytest
import numpy as np
import math
from numbers import Number
from pyTomoAO.atmosphereParametersClass import atmosphereParameters
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to get INFO messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("")

logger.info("#### Starting tests for atmosphereParametersClass ####")

class TestAtmosphereParametersClass:
    @pytest.fixture
    def valid_config(self):
        """Create a valid configuration dictionary for testing"""
        return {
            "atmosphere_parameters": {
                "nLayer": 3,
                "zenithAngleInDeg": 30.0,
                "altitude": [5, 10, 15],  # In kilometers
                "L0": 30.0,
                "r0": 0.15,
                "fractionnalR0": [0.5, 0.3, 0.2],
                "wavelength": 500e-9,
                "windDirection": [90, 45, 180],
                "windSpeed": [10, 20, 15]
            }
        }
    
    def test_initialization(self, valid_config):
        """Test successful initialization with valid parameters"""
        logger.info("Starting initialization test")
        atm = atmosphereParameters(valid_config)
        assert atm.nLayer == 3
        assert atm.zenithAngleInDeg == 30.0
        assert np.allclose(atm.altitude_km, np.array([5, 10, 15]))
        assert atm.L0 == 30.0
        assert atm.r0_zenith == 0.15
        assert np.allclose(atm.fractionnalR0, np.array([0.5, 0.3, 0.2]))
        assert atm.wavelength == 500e-9
        assert np.allclose(atm.windDirection_deg, np.array([90, 45, 180]))
        assert np.allclose(atm.windSpeed, np.array([10, 20, 15]))
        logger.info("✅ Initialization test passed")
    
    def test_airmass_calculation(self, valid_config):
        """Test airmass calculation based on zenith angle"""
        logger.info("Starting airmass calculation test")
        atm = atmosphereParameters(valid_config)
        expected_airmass = 1.0 / math.cos(math.radians(30.0))
        assert math.isclose(atm.airmass, expected_airmass)
        
        # Test with different zenith angle
        atm.zenithAngleInDeg = 60.0
        expected_airmass = 1.0 / math.cos(math.radians(60.0))
        assert math.isclose(atm.airmass, expected_airmass)
        logger.info("✅ Airmass calculation test passed")
    
    def test_altitude_scaling(self, valid_config):
        """Test altitude scaling with airmass"""
        logger.info("Starting altitude scaling test")
        atm = atmosphereParameters(valid_config)
        airmass = 1.0 / math.cos(math.radians(30.0))
        expected_altitude = np.array([5000, 10000, 15000]) * airmass
        assert np.allclose(atm.altitude, expected_altitude)
        logger.info("✅ Altitude scaling test passed")
    
    def test_r0_scaling(self, valid_config):
        """Test r0 scaling with zenith angle"""
        logger.info("Starting r0 scaling test")
        atm = atmosphereParameters(valid_config)
        cos_z = math.cos(math.radians(30.0))
        expected_r0 = 0.15 * cos_z**(3/5)
        assert math.isclose(atm.r0, expected_r0)
        logger.info("✅ r0 scaling test passed")
    
    def test_wind_components(self, valid_config):
        """Test wind velocity component calculations"""
        logger.info("Starting wind components test")
        atm = atmosphereParameters(valid_config)
        expected_vx = np.array([0, 20*math.cos(math.radians(45)), -15])
        expected_vy = np.array([10, 20*math.sin(math.radians(45)), 0])
        assert np.allclose(atm.windVx, expected_vx)
        assert np.allclose(atm.windVy, expected_vy)
        logger.info("✅ Wind components test passed")
    
    def test_invalid_nlayer(self, valid_config):
        """Test validation for nLayer"""
        logger.info("Starting nLayer validation test")
        config = valid_config.copy()
        config["atmosphere_parameters"] = valid_config["atmosphere_parameters"].copy()
        
        # Test with non-integer
        config["atmosphere_parameters"]["nLayer"] = 3.5
        with pytest.raises(TypeError):
            atmosphereParameters(config)
        
        # Test with negative value
        config["atmosphere_parameters"]["nLayer"] = -1
        with pytest.raises(ValueError):
            atmosphereParameters(config)
        
        logger.info("✅ nLayer validation test passed")
    
    def test_invalid_zenith_angle(self, valid_config):
        """Test validation for zenith angle"""
        logger.info("Starting zenith angle validation test")
        config = valid_config.copy()
        config["atmosphere_parameters"] = valid_config["atmosphere_parameters"].copy()
        
        # Test with out-of-range value
        config["atmosphere_parameters"]["zenithAngleInDeg"] = 95.0
        with pytest.raises(ValueError):
            atmosphereParameters(config)
        
        # Test with non-numeric value
        config["atmosphere_parameters"]["zenithAngleInDeg"] = "30"
        with pytest.raises(TypeError):
            atmosphereParameters(config)
        
        logger.info("✅ Zenith angle validation test passed")
    
    def test_invalid_altitude(self, valid_config):
        """Test validation for altitude"""
        logger.info("Starting altitude validation test")
        config = valid_config.copy()
        config["atmosphere_parameters"] = valid_config["atmosphere_parameters"].copy()
        
        # Test with negative value
        config["atmosphere_parameters"]["altitude"] = [5, -10, 15]
        with pytest.raises(ValueError):
            atmosphereParameters(config)
        
        # Test with wrong length
        config["atmosphere_parameters"]["altitude"] = [5, 10]
        with pytest.raises(ValueError):
            atmosphereParameters(config)
        
        logger.info("✅ Altitude validation test passed")  
    
    def test_invalid_fractionnal_r0(self, valid_config):
        """Test validation for fractional R0"""
        logger.info("Starting fractional R0 validation test")
        config = valid_config.copy()
        config["atmosphere_parameters"] = valid_config["atmosphere_parameters"].copy()
        
        # Test with negative values
        config["atmosphere_parameters"]["fractionnalR0"] = [0.5, -0.3, 0.8]
        with pytest.raises(ValueError):
            atmosphereParameters(config)
        
        # Test with sum != 1
        config["atmosphere_parameters"]["fractionnalR0"] = [0.5, 0.3, 0.3]
        with pytest.raises(ValueError):
            atmosphereParameters(config)
        
        logger.info("✅ Fractional R0 validation test passed")
    
    def test_invalid_wind_speed(self, valid_config):
        """Test validation for wind speed"""
        logger.info("Starting wind speed validation test")
        config = valid_config.copy()
        config["atmosphere_parameters"] = valid_config["atmosphere_parameters"].copy()
        
        # Test with negative value
        config["atmosphere_parameters"]["windSpeed"] = [10, -20, 15]
        with pytest.raises(ValueError):
            atmosphereParameters(config)
        
        logger.info("✅ Wind speed validation test passed")
    
    def test_invalid_L0(self, valid_config):
        """Test validation for L0"""
        logger.info("Starting L0 validation test")
        config = valid_config.copy()
        config["atmosphere_parameters"] = valid_config["atmosphere_parameters"].copy()
        
        # Test with negative value
        config["atmosphere_parameters"]["L0"] = -10
        with pytest.raises(ValueError):
            atmosphereParameters(config)
        
        # Test with non-numeric value
        config["atmosphere_parameters"]["L0"] = "30"
        with pytest.raises(TypeError):
            atmosphereParameters(config)
        
        logger.info("✅ L0 validation test passed")
    
    def test_invalid_r0_zenith(self, valid_config):
        """Test validation for r0_zenith"""
        logger.info("Starting r0_zenith validation test")
        config = valid_config.copy()
        config["atmosphere_parameters"] = valid_config["atmosphere_parameters"].copy()
        
        # Test with negative value
        config["atmosphere_parameters"]["r0"] = -0.15
        with pytest.raises(ValueError):
            atmosphereParameters(config)
        
        logger.info("✅ r0_zenith validation test passed")
    
    def test_invalid_wavelength(self, valid_config):
        """Test validation for wavelength"""
        logger.info("Starting wavelength validation test")
        config = valid_config.copy()
        config["atmosphere_parameters"] = valid_config["atmosphere_parameters"].copy()
        
        # Test with negative value
        config["atmosphere_parameters"]["wavelength"] = -500e-9
        with pytest.raises(ValueError):
            atmosphereParameters(config)
        
        logger.info("✅ Wavelength validation test passed")
    
    def test_r0_setter(self, valid_config):
        """Test that r0 cannot be set directly"""
        logger.info("Starting r0 setter test")
        atm = atmosphereParameters(valid_config)
        with pytest.raises(AttributeError):
            atm.r0 = 0.2
        
        logger.info("✅ r0 setter test passed")
    
    def test_string_representation(self, valid_config):
        """Test string representation contains key information"""
        logger.info("Starting string representation test")
        atm = atmosphereParameters(valid_config)
        str_repr = str(atm)
        
        # Check that the string contains key information
        assert "Zenith Angle: 30.0°" in str_repr
        assert "Outer Scale (L0): 30.0 m" in str_repr
        assert "Fried Parameter (r0): 0.150 m" in str_repr
        assert "Wavelength: 500.0 nm" in str_repr
        assert "Layer 1" in str_repr
        assert "Layer 2" in str_repr
        assert "Layer 3" in str_repr
        assert "Total fractional R0: 1.0000" in str_repr
        logger.info("✅ String representation test passed")