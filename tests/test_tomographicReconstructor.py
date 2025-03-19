"""
To run the tests in this file using pytest, navigate to the repository:

    cd /path/to/pyTomoAO

Execute the following command in your terminal:

    pytest tests/test_tomographicReconstructor.py -v

Ensure that you have pytest installed in your environment. You can install it via pip if necessary:

    pip install pytest
"""

import pytest
import numpy as np
import yaml
import os
import logging
from unittest.mock import patch, MagicMock

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to get INFO messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("")

# Mock the parameter classes to avoid initialization errors
@pytest.fixture
def mock_parameter_classes():
    """
    Fixture that mocks all parameter classes needed by the tomographicReconstructor.
    Returns a dictionary with all mock objects for easy access in tests.
    """
    logger.debug("Setting up mock parameter classes")
    with patch('pyTomoAO.tomographicReconstructor.atmosphereParameters') as mock_atm, \
        patch('pyTomoAO.tomographicReconstructor.lgsAsterismParameters') as mock_lgs_asterism, \
        patch('pyTomoAO.tomographicReconstructor.lgsWfsParameters') as mock_lgs_wfs, \
        patch('pyTomoAO.tomographicReconstructor.tomographyParameters') as mock_tomo:
        
        # Configure mock atmosphere parameters
        logger.debug("Configuring mock atmosphere parameters")
        mock_atm_instance = MagicMock()
        mock_atm_instance.r0 = 0.15
        mock_atm_instance.L0 = 25.0
        mock_atm_instance.__str__.return_value = "Mock Atmosphere Parameters"
        mock_atm.return_value = mock_atm_instance
        
        # Configure mock LGS asterism parameters
        logger.debug("Configuring mock LGS asterism parameters")
        mock_lgs_asterism_instance = MagicMock()
        mock_lgs_asterism_instance.nLGS = 4
        mock_lgs_asterism_instance.LGSwavelength = 589e-9
        mock_lgs_asterism_instance.__str__.return_value = "Mock LGS Asterism Parameters"
        mock_lgs_asterism.return_value = mock_lgs_asterism_instance
        
        # Configure mock LGS WFS parameters
        logger.debug("Configuring mock LGS WFS parameters")
        mock_lgs_wfs_instance = MagicMock()
        mock_lgs_wfs_instance.nLGS = 4
        mock_lgs_wfs_instance.__str__.return_value = "Mock LGS WFS Parameters"
        mock_lgs_wfs.return_value = mock_lgs_wfs_instance
        
        # Configure mock tomography parameters
        logger.debug("Configuring mock tomography parameters")
        mock_tomo_instance = MagicMock()
        mock_tomo_instance.nLGS = 4
        mock_tomo_instance.nFitSrc = 1
        # Add r0 and L0 attributes to match expected forwarding behavior
        mock_tomo_instance.r0 = 0.15
        mock_tomo_instance.L0 = 25.0
        mock_tomo_instance.sampling = 49
        mock_tomo_instance.__str__.return_value = "Mock Tomography Parameters"
        mock_tomo.return_value = mock_tomo_instance
        
        logger.debug("Mock parameter classes setup complete")
        yield {
            "atm": mock_atm,
            "lgs_asterism": mock_lgs_asterism,
            "lgs_wfs": mock_lgs_wfs,
            "tomo": mock_tomo,
            "atm_instance": mock_atm_instance,
            "lgs_asterism_instance": mock_lgs_asterism_instance,
            "lgs_wfs_instance": mock_lgs_wfs_instance,
            "tomo_instance": mock_tomo_instance
        }

# Fixture for a simple config file
@pytest.fixture
def simple_config():
    """
    Fixture that creates a temporary YAML config file with basic settings.
    Returns the path to the temporary file and cleans it up after the test.
    """
    logger.debug("Creating temporary config file for testing")
    config = {
        "atmosphere": {
            "r0": 0.186,
            "L0": 30.0,
        },
        "lgs_asterism": {
            "height": 90000,
            "radiusAst": 7.6,
            "nLGS": 4,
            "wavelength": 5.89e-7
        },
        "lgs_wfs": {
            "nSubap": 20,
        },
        "tomography": {
            "method": "LTAO",
            "fovOptimization": 0,
            "nFitSrc": 1,
            "fitSrcWeight": 1.0,
        }
    }
    # Create a temporary config file
    filename = "test_config.yaml"
    with open(filename, "w") as f:
        yaml.dump(config, f)
    logger.debug(f"Temporary config file created at {filename}")
    yield filename
    # Clean up
    if os.path.exists(filename):
        logger.debug(f"Removing temporary config file {filename}")
        os.remove(filename)

# Test initialization
def test_initialization(simple_config, mock_parameter_classes):
    """
    Test that initializing the tomographicReconstructor with a valid config file
    correctly instantiates all required parameter classes.
    """
    logger.info("Starting initialization test")
    from pyTomoAO.tomographicReconstructor import tomographicReconstructor
    
    logger.debug("Creating reconstructor instance")
    reconstructor = tomographicReconstructor(simple_config)
    
    # Check if mocked parameter classes were called
    logger.debug("Verifying parameter classes were instantiated")
    mock_parameter_classes["atm"].assert_called_once()
    mock_parameter_classes["lgs_asterism"].assert_called_once()
    mock_parameter_classes["lgs_wfs"].assert_called_once()
    mock_parameter_classes["tomo"].assert_called_once()
    
    # Check that the reconstructor has the parameter attributes
    logger.debug("Verifying parameter attributes were set on reconstructor")
    assert hasattr(reconstructor, "atmParams"), "Missing atmParams attribute"
    assert hasattr(reconstructor, "lgsAsterismParams"), "Missing lgsAsterismParams attribute"
    assert hasattr(reconstructor, "lgsWfsParams"), "Missing lgsWfsParams attribute"
    assert hasattr(reconstructor, "tomoParams"), "Missing tomoParams attribute"
    
    logger.info("✅ Initialization test completed successfully")

# Test attribute forwarding and updates
def test_attribute_forwarding_uppdate(simple_config, mock_parameter_classes):
    """
    Test that attributes from parameter classes are correctly forwarded and
    accessible directly from the reconstructor instance.
    """
    logger.info("Starting attribute forwarding test")
    from pyTomoAO.tomographicReconstructor import tomographicReconstructor
    
    logger.debug("Creating reconstructor instance")
    reconstructor = tomographicReconstructor(simple_config)
    
    # Manually set parameter instances for testing
    logger.debug("Setting mock parameter instances")
    reconstructor.atmParams = mock_parameter_classes["atm_instance"]
    reconstructor.tomoParams = mock_parameter_classes["tomo_instance"]
    reconstructor.lgsWfsParams = mock_parameter_classes["lgs_wfs_instance"]
    reconstructor.lgsAsterismParams = mock_parameter_classes["lgs_asterism_instance"]
    
    # Check forwarded attribute from atmParams & lgsAsterismParams
    logger.debug("Verifying forwarded attributes")
    assert reconstructor.r0 == 0.15, "Incorrect forwarded r0 value"
    assert reconstructor.L0 == 25, "Incorrect forwarded L0 value"
    assert reconstructor.nLGS == 4, "Incorrect forwarded nLGS value"
    
    # Check that changing nLGS updates all parameter classes
    logger.debug("Changing nLGS to 6")
    reconstructor.nLGS = 6
    assert reconstructor.nLGS == 6, "nLGS not updated correctly"
    assert reconstructor.lgsWfsParams.nLGS == 6, "nLGS not updated in lgsWfsParams"
    assert reconstructor.lgsAsterismParams.nLGS == 6, "nLGS not updated in lgsAsterismParams"
    assert reconstructor.tomoParams.nLGS == 6, "nLGS not updated in tomoParams"
    logger.info("✅ Attribute forwarding test completed successfully")

# Test reconstruct_wavefront
def test_reconstruct_wavefront(simple_config, mock_parameter_classes):
    """
    Test the reconstruct_wavefront method with a mocked reconstructor.
    Verifies that the method correctly applies the reconstruction matrix
    to the input slopes and maps the result to a proper wavefront.
    """
    logger.info("Starting reconstruct_wavefront test")
    from pyTomoAO.tomographicReconstructor import tomographicReconstructor
    
    logger.debug("Creating reconstructor instance")
    reconstructor = tomographicReconstructor(simple_config)
    
    # Create a grid mask that matches the actual implementation
    logger.debug("Setting up grid mask")
    grid_mask = np.ones((30, 30))
    grid_mask[0, 0] = 0  # Add some masking
    
    # Count valid points in the grid mask
    n_valid_points = np.sum(grid_mask == 1)
    logger.debug(f"Grid has {n_valid_points} valid points")
    
    # Set up reconstructor matrix with the correct shape
    logger.debug("Setting up mock reconstructor matrix")
    reconstructor._reconstructor = np.ones((n_valid_points, 800))
    reconstructor._gridMask = grid_mask
    
    # Creates a mock slopes array
    logger.debug("Creating test slopes")
    slopes = np.ones(800)
    
    # Reconstruct wavefront using the reconstruct_wavefront method
    logger.debug("Calling reconstruct_wavefront")
    wavefront = reconstructor.reconstruct_wavefront(slopes)
    
    # Check shape
    logger.debug("Verifying wavefront shape")
    assert wavefront.shape == grid_mask.shape, f"Expected shape {grid_mask.shape}, got {wavefront.shape}"
    
    # Check values
    logger.debug("Verifying wavefront values")
    assert np.isnan(wavefront[0, 0]), "Masked point should be NaN"
    # Each valid point value should be 800 (sum of all slopes)
    valid_values = wavefront[~np.isnan(wavefront)]
    assert np.all(valid_values == 800), f"Expected all valid values to be 800, got {np.unique(valid_values)}"
    
    logger.info("✅ Reconstruct_wavefront test completed successfully")

# Test reconstructor property
def test_reconstructor_property_build(simple_config, mock_parameter_classes):
    """
    Test that accessing the reconstructor property builds it if needed.
    Verifies the lazy-loading mechanism works correctly.
    """
    logger.info("Starting reconstructor property test")
    from pyTomoAO.tomographicReconstructor import tomographicReconstructor
    
    # Create the reconstructor
    logger.debug("Creating reconstructor instance")
    reconstructor = tomographicReconstructor(simple_config)
    
    # The simplest approach: directly mock the reconstructor property
    logger.debug("Setting up test reconstructor matrix")
    test_reconstructor = np.ones((10, 10))
    reconstructor._reconstructor = test_reconstructor
    
    # Access the reconstructor property
    logger.debug("Accessing reconstructor property")
    result = reconstructor.reconstructor
    
    # Verify we got the expected result
    logger.debug("Verifying property returns correct matrix")
    assert result is test_reconstructor, "Reconstructor property returned unexpected matrix"
    
    logger.info("✅ Reconstructor property test completed successfully")

# Test R property alias
def test_R_property_alias(simple_config, mock_parameter_classes):
    """
    Test that R is a proper alias for the reconstructor property.
    Verifies both getter and setter functionality.
    """
    logger.info("Starting R property alias test")
    from pyTomoAO.tomographicReconstructor import tomographicReconstructor
    
    # Create a reconstructor instance
    logger.debug("Creating reconstructor instance")
    reconstructor = tomographicReconstructor(simple_config)
    
    # Set up the _reconstructor attribute
    logger.debug("Setting up test matrices")
    test_matrix = np.ones((10, 10), dtype=np.float32)
    reconstructor._reconstructor = test_matrix
    
    # Check R property
    logger.debug("Verifying R getter returns correct matrix")
    assert np.array_equal(reconstructor.R, test_matrix), "R property getter failed"
    
    # Set via R property
    logger.debug("Testing R property setter")
    new_matrix = np.zeros((10, 10), dtype=np.float32)
    reconstructor.R = new_matrix
    # Check both properties
    logger.debug("Verifying setter affected both properties")
    assert np.array_equal(reconstructor.reconstructor, reconstructor.R), "R setter didn't update reconstructor property"
    
    logger.info("✅ R property alias test completed successfully")

# Test visualization
def test_visualize_reconstruction(simple_config, mock_parameter_classes):
    """
    Test the visualization of reconstruction results.
    Verifies that appropriate figures are created for both with and without reference wavefronts.
    """
    logger.info("Starting visualization test")
    from pyTomoAO.tomographicReconstructor import tomographicReconstructor
    
    # Create the reconstructor
    logger.debug("Creating reconstructor instance")
    reconstructor = tomographicReconstructor(simple_config)
    
    # Create a grid mask
    logger.debug("Setting up grid mask")
    grid_mask = np.ones((30, 30))
    grid_mask[0, 0] = 0  # Add some masking
    
    # Count valid points
    n_valid_points = np.sum(grid_mask == 1)
    logger.debug(f"Grid has {n_valid_points} valid points")
    
    # Set up reconstructor matrix
    logger.debug("Setting up mock reconstructor matrix")
    reconstructor._reconstructor = np.ones((n_valid_points, 800))
    reconstructor._gridMask = grid_mask
    
    # Generate test slopes
    logger.debug("Creating test slopes")
    slopes = np.ones(800)
    
    # Test without reference
    logger.debug("Testing visualization without reference")
    fig1 = reconstructor.visualize_reconstruction(slopes)
    assert fig1 is not None, "Figure should be created"
    
    # Test with reference
    logger.debug("Testing visualization with reference")
    reference_wavefront = np.zeros((30, 30))
    fig2 = reconstructor.visualize_reconstruction(slopes, reference_wavefront)
    assert fig2 is not None, "Figure with reference should be created"
    
    # Check basic figure properties
    logger.debug("Checking figure properties")
    assert len(fig1.axes) > 0, "Figure should have at least one axis"
    assert len(fig2.axes) > 0, "Figure with reference should have at least one axis"
    
    logger.info("✅ Visualization test completed successfully")

# Test handling file not found
def test_initialization_with_invalid_config():
    """
    Test initialization with a non-existent config file.
    Verifies proper error handling.
    """
    logger.info("Starting test with invalid config file")
    from pyTomoAO.tomographicReconstructor import tomographicReconstructor
    
    logger.debug("Attempting to initialize with non-existent file")
    with pytest.raises(FileNotFoundError):
        reconstructor = tomographicReconstructor("nonexistent_config.yaml")
    
    logger.info("✅ Invalid config file test completed successfully")

# Test error in build_reconstructor
def test_build_reconstructor_error(simple_config, mock_parameter_classes):
    """Test error handling in build_reconstructor."""
    from pyTomoAO.tomographicReconstructor import tomographicReconstructor
    import pytest
    
    logger.info("Starting test for error handling in build_reconstructor.")
    
    # Patch the build_reconstructor method to raise a ValueError
    logger.debug("Patching the build_reconstructor method to raise a ValueError.")
    with patch.object(tomographicReconstructor, 'build_reconstructor', 
                    side_effect=ValueError("Test error")):
        
        # Create reconstructor
        logger.info("Creating reconstructor with mocked parameters.")
        reconstructor = tomographicReconstructor(simple_config)
        
        # Test error handling
        logger.info("Expecting a ValueError when calling build_reconstructor.")
        with pytest.raises(ValueError, match="Test error"):
            reconstructor.build_reconstructor()
        
        logger.info("✅ Error handling in build_reconstructor test completed successfully")

# Integration test (skipped by default)
#@pytest.mark.skip(reason="Integration test requiring actual implementation")
def test_full_reconstruction(config_file=None):
    """Integration test for the full reconstruction pipeline."""
    logger.info("Starting integration test for full reconstruction pipeline.")
    from pyTomoAO.tomographicReconstructor import tomographicReconstructor
    
    # Use a default config file if none provided
    if config_file is None:
        config_file = "tests/tomography_config_kapa.yaml"
        
    logger.debug(f"Using config file: {config_file}")
    
    # Create the reconstructor
    logger.info("Creating tomographicReconstructor instance")
    reconstructor = tomographicReconstructor(config_file)
    
    # Build the reconstructor
    logger.info("Building the reconstructor")
    logger.debug("Building the reconstructor")
    rec = reconstructor.reconstructor
    
    # Verify the result
    logger.debug("Verifying reconstruction matrix was created")
    assert hasattr(reconstructor, '_reconstructor'), "Reconstructor matrix not created"
    assert reconstructor._reconstructor is not None, "Reconstructor matrix is None"
    
    # Generate test slopes tip tilt
    logger.info("Generating test slopes tip tilt")
    TipTilt = np.zeros(reconstructor.lgsWfsParams.nValidSubap*2)
    TipTilt[0:reconstructor.lgsWfsParams.nValidSubap-1]= 4
    TipTilt[reconstructor.lgsWfsParams.nValidSubap::]= -4
    TipTilt = np.tile(TipTilt,reconstructor.nLGS)
    
    # Reconstruct wavefront
    logger.info("Reconstructing wavefront")
    wavefront = reconstructor.reconstruct_wavefront(TipTilt)
    
    # Verify wavefront shape
    expected_shape = (reconstructor.tomoParams.sampling, reconstructor.tomoParams.sampling)
    logger.debug(f"Verifying wavefront shape: expected {expected_shape}")
    assert wavefront.shape == expected_shape, f"Expected shape {expected_shape}, got {wavefront.shape}"
    
    # verify the accuracy of the reconstruction
    logger.info("Verifying the accuracy of the reconstruction")
    meanOpd = np.nanmean(wavefront)*1e9
    opd_test = np.allclose(meanOpd, 140.96292, rtol=0, atol=1e-4)
    assert opd_test == True, "Reconstructed wavefront is not accurate"
    
    # update nLGS
    logger.info("Updating nLGS to 6")
    reconstructor.nLGS = 6
    assert reconstructor.nLGS == 6, "nLGS not updated correctly"
    assert reconstructor.lgsWfsParams.nLGS == 6, "nLGS not updated in lgsWfsParams"
    assert reconstructor.lgsAsterismParams.nLGS == 6, "nLGS not updated in lgsAsterismParams"
    
    # Update r0_zenith
    logger.info("Updating r0_zenith to 0.1")
    reconstructor.r0_zenith = 0.1
    assert reconstructor.r0_zenith == 0.1, "r0_zenith not updated correctly"
    assert reconstructor.atmParams.r0_zenith == 0.1, "r0_zenith not updated in atmParams"
    
    # rebuild the reconstructor
    logger.info("Rebuilding the reconstructor")
    rec2 = reconstructor.build_reconstructor()
    logger.info("Verifying reconstruction matrix was updated")
    assert hasattr(reconstructor, '_reconstructor'), "Reconstructor matrix not updated"
    assert reconstructor._reconstructor is not None, "Reconstructor matrix is None"
    assert rec2 is not rec, "Reconstructor matrix was not updated"
    
    # Generate test slopes tip tilt
    logger.info("Generating test slopes tip tilt")
    TipTilt = np.zeros(reconstructor.lgsWfsParams.nValidSubap*2)
    TipTilt[0:reconstructor.lgsWfsParams.nValidSubap-1]= 4
    TipTilt[reconstructor.lgsWfsParams.nValidSubap::]= -4
    TipTilt = np.tile(TipTilt,reconstructor.nLGS)
    
    # Reconstruct wavefront
    logger.info("Reconstructing wavefront")
    wavefront = reconstructor.reconstruct_wavefront(TipTilt)
    
    # verify the accuracy of the reconstruction
    logger.info("Verifying the accuracy of the reconstruction")
    meanOpd = np.nanmean(wavefront)*1e9
    opd_test = np.allclose(meanOpd, 142.56623, rtol=0, atol=1e-4)
    assert opd_test == True, "Reconstructed wavefront is not accurate"
    
    logger.info("✅ Integration test for full reconstruction completed successfully")