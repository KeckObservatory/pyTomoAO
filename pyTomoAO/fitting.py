# fitting.py

"""
This module contains the fitting class for handling deformable mirror fitting operations.
It includes methods for computing influence functions, generating fitting matrices, and fitting OPD maps.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
import yaml
import sys
sys.path.append('..')

# Configure logging
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class fitting:
    """A class for handling deformable mirror fitting operations with influence function computation.

    This class provides methods for computing influence functions, generating fitting matrices,
    and fitting optical path difference (OPD) maps for deformable mirror control. The class
    supports both single and double Gaussian influence functions, and can handle different
    grid resolutions and actuator geometries. The class forwards attribute access to dmParams
    when appropriate.

    Parameters
    ----------
    dmParams : object
        An instance of a class containing DM geometry parameters
    logger : logging.Logger, optional
        Logger object for logging messages (default is the module-level logger)

    Returns
    -------
    None
        Initializes the fitting object with the specified parameters.

    Notes
    -----
    The class maintains several internal attributes:
    - modes : numpy.ndarray
        Array containing influence function modes
    - resolution : int
        Resolution of the grid for influence function computation
    - _fitting_matrix : numpy.ndarray
        Matrix used for fitting OPD maps to actuator commands
    - _influence_functions : numpy.ndarray
        Matrix of influence functions for each actuator
    - actuator_coordinates : list
        List of (y, x) coordinate tuples for all actuators
    """
    
    def __init__(self, dmParams, logger=logger):
        """
        Initialize the fitting class.
        
        Parameters
        ----------
        dmParams : object
            An instance of a class containing DM geometry parameters
        logger : logging.Logger, optional
            Logger instance for tracking operations
        """
        logger.info("\n -->> Initializing fitting object <<--")
        self.dmParams = dmParams
        # Initialize other class attributes as needed
        self.modes = None
        self.resolution = 49  # Default resolution
        self._fitting_matrix = np.array([])
        self._influence_functions = np.array([])
        logger.info("\nAll parameters initialized successfully.")
    
    def __getattr__(self, name):
        """
        Forward attribute access to the dmParams class if it contains the requested attribute.
        
        Parameters
        ----------
        name : str
            Name of the attribute to access
            
        Returns
        -------
        object
            The requested attribute from dmParams
            
        Raises
        ------
        AttributeError
            If the attribute doesn't exist in either class
        """
        if hasattr(self.dmParams, name):
            return getattr(self.dmParams, name)
        raise AttributeError(f"Neither 'fitting' nor 'dmParams' has attribute '{name}'")
    
    def __setattr__(self, name, value):
        """
        Forward attribute setting to the dmParams class if it contains the specified attribute.
        
        Parameters
        ----------
        name : str
            Name of the attribute to set
        value : object
            Value to assign to the attribute
        """
        # Special case for initialization and our own attributes
        if name in ["dmParams", "modes", "resolution", "_fitting_matrix", "_influence_functions"] or not hasattr(self, "dmGeometry"):
            object.__setattr__(self, name, value)
        elif hasattr(self.dmParams, name):
            setattr(self.dmParams, name, value)
        else:
            object.__setattr__(self, name, value)
    
    @property
    def F(self):
        """
        Get the fitting matrix.
        
        Returns
        -------
        numpy.ndarray
            The fitting matrix for OPD map to actuator command conversion
        """
        logger.debug("Accessing the fitting matrix property.")
        return self._fitting_matrix
        
    @F.setter
    def F(self, value):
        """
        Set the fitting matrix.
        
        Parameters
        ----------
        value : numpy.ndarray
            The fitting matrix to set
            
        Raises
        ------
        ValueError
            If the value is not a 2D numpy array
        """
        logger.debug("Setting the fitting matrix property.")
        if isinstance(value, np.ndarray) and value.ndim == 2:
            self._fitting_matrix = value
        else:
            logger.error("Invalid fitting matrix value. Must be a 2D numpy array.")
            raise ValueError("Fitting matrix must be a 2D numpy array.")
    
    @property
    def fitting_matrix(self):
        """
        Get the fitting matrix (alias for F property).
        
        Returns
        -------
        numpy.ndarray
            The fitting matrix for OPD map to actuator command conversion
        """
        logger.debug("Accessing the full name fitting_matrix property.")
        return self._fitting_matrix
        
    @fitting_matrix.setter
    def fitting_matrix(self, value):
        """
        Set the fitting matrix (alias for F property).
        
        Parameters
        ----------
        value : numpy.ndarray
            The fitting matrix to set
        """
        self.F = value
    
    @property
    def IF(self):
        """
        Get the influence functions matrix.
        
        Returns
        -------
        numpy.ndarray
            The influence functions matrix
        """
        logger.debug("Accessing the influence functions property.")
        return self._influence_functions
        
    @IF.setter
    def IF(self, value):
        """
        Set the influence functions matrix.
        
        Parameters
        ----------
        value : numpy.ndarray
            The influence functions matrix to set
            
        Raises
        ------
        ValueError
            If the value is not a 2D numpy array
        """
        logger.debug("Setting the influence functions property.")
        if isinstance(value, np.ndarray) and value.ndim == 2:
            self._influence_functions = value
        else:
            logger.error("Invalid influence functions value. Must be a 2D numpy array.")
            raise ValueError("Influence functions must be a 2D numpy array.")
    
    @property
    def influence_functions(self):
        """
        Get the influence functions matrix (alias for IF property).
        
        Returns
        -------
        numpy.ndarray
            The influence functions matrix
        """
        logger.debug("Accessing the full name influence_functions property.")
        return self._influence_functions
        
    @influence_functions.setter
    def influence_functions(self, value):
        """
        Set the influence functions matrix (alias for IF property).
        
        Parameters
        ----------
        value : numpy.ndarray
            The influence functions matrix to set
        """
        self.IF = value
    
    def fit(self, opd_map):
        """
        Multiply the OPD map by the fitting matrix to obtain the command vector.
        
        Parameters
        ----------
        opd_map : numpy.ndarray
            The Optical Path Difference (OPD) map to be fitted
            
        Returns
        -------
        numpy.ndarray
            The command vector to send to the DM
            
        Raises
        ------
        ValueError
            If the fitting matrix is not set
        """
        logger.info("\nPerforming fitting of the OPD map.")
        if self.F.size == 0:
            logger.error("Fitting matrix is not set.")
            raise ValueError("Fitting matrix is not set.")
        
        command_vector = np.dot(self.F, opd_map.flatten())
        logger.debug("Fitting completed. Command vector shape: %s", command_vector.shape)
        return command_vector
    
    def double_gaussian_influence(self, x, y, center_x=0, center_y=0, w1=2, w2=-1, sigma1=0.54, sigma2=0.85):
        """
        Compute the double Gaussian influence function for a deformable mirror.
        
        This function allows placement of a double Gaussian influence function at
        any position on a grid of any dimensions.
        
        Parameters
        ----------
        x, y : float or numpy.ndarray
            Coordinates at which to evaluate the influence function
        center_x, center_y : float
            Center coordinates of the double Gaussian function
        w1, w2 : float
            Weights of the two Gaussian components
        sigma1, sigma2 : float
            Standard deviations of the two Gaussian components
        
        Returns
        -------
        float or numpy.ndarray
            Influence function value at the given coordinates
        """
        # Calculate distances from the center position
        dx = x - center_x
        dy = y - center_y
        
        # Compute the two Gaussian components
        gauss1 = w1 * np.exp(-(dx**2 + dy**2) / (2 * sigma1**2)) / (2 * np.pi * sigma1**2)
        gauss2 = w2 * np.exp(-(dx**2 + dy**2) / (2 * sigma2**2)) / (2 * np.pi * sigma2**2)
        
        return gauss1 + gauss2
    
    def create_influence_grid(self, grid_shape, actuator_pos, w1=2, w2=-1, sigma1=0.5, sigma2=0.85):
        """
        Create a grid of the specified shape with a double Gaussian placed at the given position.
        
        Parameters
        ----------
        grid_shape : tuple
            Shape of the grid (height, width)
        actuator_pos : tuple
            Position (y, x) where the center of the double Gaussian should be placed
        w1, w2 : float
            Weights of the two Gaussian components
        sigma1, sigma2 : float
            Standard deviations of the two Gaussian components
        
        Returns
        -------
        numpy.ndarray
            2D grid with the double Gaussian influence function
        """
        height, width = grid_shape
        y, x = np.ogrid[:height, :width]
        
        # Convert to float coordinates if needed
        center_y, center_x = actuator_pos
        
        # Calculate the influence function on the grid
        influence = self.double_gaussian_influence(x, y, center_x, center_y, w1, w2, sigma1, sigma2)
        
        return influence
    
    def extract_actuator_coordinates(self, valid_actuator_map):
        """
        Extract the (y, x) coordinates of all actuators from the map.
        
        Parameters
        ----------
        valid_actuator_map : numpy.ndarray
            Binary array where 1s indicate valid actuator positions
        
        Returns
        -------
        list
            List of (y, x) coordinate tuples for all actuators
        """
        # Find coordinates where value is 1 (True)
        y_coords, x_coords = np.where(valid_actuator_map)
        
        # Return as list of coordinate tuples
        return list(zip(y_coords, x_coords))
    
    def map_actuators_to_new_grid(self, actuator_coords, original_shape, new_shape, stretch_factor=1.03):
        """
        Map actuator coordinates from original grid to a new grid size.
        
        Maintains relative positions and stretches beyond [-1, 1] by the stretch factor.
        
        Parameters
        ----------
        actuator_coords : list
            List of (y, x) coordinate tuples in the original grid
        original_shape : tuple
            Shape of the original grid (height, width)
        new_shape : tuple
            Shape of the new grid (height, width)
        stretch_factor : float, optional
            Factor to stretch the normalized coordinates (default: 1.03)
            
        Returns
        -------
        list
            List of (y, x) coordinate tuples in the new grid, normalized and stretched
        """
        orig_height, orig_width = original_shape
        new_height, new_width = new_shape
        
        # Create new coordinates list
        new_coords = []
        
        for y, x in actuator_coords:
            # First, convert to [0, 1] range by dividing by original dimensions
            normalized_y = y / (orig_height - 1)  # -1 to account for 0-indexing
            normalized_x = x / (orig_width - 1)
            
            # Then, convert from [0, 1] to [-1, 1] range
            new_y = 2 * normalized_y - 1
            new_x = 2 * normalized_x - 1
            
            # Apply the stretch factor to extend beyond [-1, 1]
            new_y = new_y * stretch_factor
            new_x = new_x * stretch_factor
            
            # Finally, scale to new dimensions if needed
            if new_shape != (-1, -1):  # Assuming (-1, -1) means "keep normalized"
                new_y = new_y * (new_height - 1) / 2 + (new_height - 1) / 2
                new_x = new_x * (new_width - 1) / 2 + (new_width - 1) / 2
            
            new_coords.append((new_y, new_x))
        
        self.actuator_coordinates = new_coords
        return new_coords
    
    def map_actuators_to_new_grid_old(self, actuator_coords, original_shape, new_shape):
        """
        Map actuator coordinates from original grid to a new grid size (legacy method).
        
        Maintains relative positions using direct scaling.
        
        Parameters
        ----------
        actuator_coords : list
            List of (y, x) coordinate tuples in the original grid
        original_shape : tuple
            Shape of the original grid (height, width)
        new_shape : tuple
            Shape of the new grid (height, width)

        Returns
        -------
        list
            List of (y, x) coordinate tuples in the new grid
        """
        orig_height, orig_width = original_shape
        new_height, new_width = new_shape
        # Calculate scaling factors
        y_scale = new_height / orig_height
        x_scale = new_width / orig_width
        # Apply scaling to each coordinate
        new_coords = []
        for y, x in actuator_coords:
            new_y = y * y_scale+0.5#int(y  y_scale)*
            new_x = x * x_scale+0.5#int(x  x_scale)*
            new_coords.append((new_y, new_x))
        
        self.actuator_coordinates = new_coords
        return new_coords
    
    def set_influence_function(self, dmParams=None, resolution=None, display=False, w1=2, w2=-1, sigma1=0.5*2, sigma2=0.85*2):
        """
        Generate a deformable mirror influence function based on the provided parameters.
        
        This method computes the influence functions for each actuator in the DM
        and stores them in the class.
        
        Parameters
        ----------
        dmParams : object, optional
            Contains deformable mirror parameters.
            If None, uses the associated dmParams.
        resolution : int, optional
            Resolution of the output influence function grid.
            If None, uses the class default resolution.
        display : bool, optional
            Whether to display plots of the influence functions.
        w1, w2 : float, optional
            Weights for the double Gaussian function.
        sigma1, sigma2 : float, optional
            Standard deviations for the double Gaussian function.
            
        Returns
        -------
        numpy.ndarray
            2D array representing the influence function for each actuator.
        """
        if dmParams is None:
            dmParams = self.dmParams
        
        if resolution is None:
            resolution = self.resolution
        
        logger.info("\n-->> Computing influence function <<--")
        
        # Extract actuator coordinates from the valid actuator map
        actuator_coords = self.extract_actuator_coordinates(dmParams.validActuatorsSupport)
        
        # Get the original shape of the valid actuator map
        original_shape = dmParams.validActuatorsSupport.shape
        # Map actuator coordinates to the new grid size
        new_actuator_coords = self.map_actuators_to_new_grid(
            actuator_coords, 
            original_shape, 
            new_shape=(resolution, resolution)
        )
        
        # Create modes array to store influence functions
        modes = np.zeros((resolution**2, dmParams.validActuators.sum()))
        
        # Loop through each actuator coordinate
        for i in range(dmParams.validActuators.sum()):
            # Compute the influence function for the current actuator
            IF = self.create_influence_grid((resolution, resolution), new_actuator_coords[i],w1, w2, sigma1, sigma2)
            
            # Flatten and assign to the corresponding column in the modes array
            modes[:, i] = IF.flatten()
            
            if display:
                plt.figure(1)
                plt.clf()
                plt.imshow(IF, interpolation='nearest')
                plt.title(f'Influence Function for Actuator {i}')
                plt.colorbar()
                plt.axis('off')  # Remove the axis from the plot
                plt.show()
                plt.pause(0.01)
        
        # Store the modes in the class and in influence_functions
        self.modes = modes
        self.IF = modes  # Use the property setter
        
        logger.debug("Influence function computed.")
        logger.info("\n-->> Influence function computed <<--")
        return modes

# Main execution block
if __name__ == "__main__":
    from pyTomoAO.tomographicReconstructor import tomographicReconstructor
    from pyTomoAO.fitting import fitting
    # Load the reconstructor
    reconstructor = tomographicReconstructor("../examples/benchmark/tomography_config_kapa_single_channel.yaml")
    reconstructor.build_reconstructor()
    gridMask = reconstructor.gridMask
    
    # Create a fitting instance
    print("\nInitializing fitting object...")
    fit = fitting(reconstructor.dmParams)
    
    # Generate influence functions
    print("\nGenerating influence functions...")
    modes = fit.set_influence_function(resolution=49, display=False, sigma1=0.5*2, sigma2=0.85*2)
    print(f"Generated influence functions with shape: {modes.shape}")
    
    # Display one influence function
    plt.figure()
    plt.imshow(modes[:, 0].reshape(49, 49), cmap='viridis')
    plt.plot(fit.actuator_coordinates[0][1], fit.actuator_coordinates[0][0], 'x', color='black')
    plt.colorbar()
    plt.title("Influence Function for First Actuator")
    plt.show()
    
    # Change the modes size with only valid elements of the gridMask 
#    modes = modes[gridMask.flatten(), :]
#    fit.modes = modes
#    print(f"Modes shape after applying grid mask: {modes.shape}")
    
    # Generate a fitting matrix (pseudo-inverse of the influence functions)
    print("\nCalculating fitting matrix...")
    fit.F = np.linalg.pinv(modes)
    print(f"Fitting matrix shape: {fit.F.shape}")
    
    # Test the aliases
    assert np.array_equal(fit.F, fit.fitting_matrix), "F and fitting_matrix should be the same"
    assert np.array_equal(fit.IF, fit.influence_functions), "IF and influence_functions should be the same"
    
    # Function to apply grid mask and handle NaNs
    def apply_mask(wavefront, mask):
        masked = wavefront * mask
        masked_for_display = masked.copy()
        masked_for_display[masked == 0] = np.nan
        return masked, masked_for_display
    
    # Function to process and display a wavefront
    def process_wavefront(wavefront_name, wavefront, fit, gridMask):
        print(f"\nProcessing {wavefront_name} wavefront...")
        
        # Apply mask
        masked_wavefront, display_wavefront = apply_mask(wavefront, gridMask)
        
        #masked_wavefront = masked_wavefront[reconstructor.gridMask]
        # Plot the original wavefront
        plt.figure()
        plt.imshow(display_wavefront, cmap='RdBu')
        for i in range(len(fit.actuator_coordinates)):
            plt.plot(fit.actuator_coordinates[i][0], fit.actuator_coordinates[i][1], 'x', color='black')
        plt.colorbar()
        plt.title(f"Input Wavefront ({wavefront_name})")
        plt.show()
        
        # Perform the fitting
        print(f"Performing fitting of the {wavefront_name} wavefront...")
        commands = fit.fit(masked_wavefront)
        print(f"Generated command vector with {len(commands)} values")
        
        # Plot the commands
        plt.figure()
        plt.bar(range(len(commands)), commands)
        plt.title(f"DM Actuator Commands for {wavefront_name}")
        plt.xlabel("Actuator Index")
        plt.ylabel("Command Value")
        plt.show()
        
        # Reconstruct the wavefront from the commands
        print(f"Reconstructing {wavefront_name} wavefront from commands...")
        reconstructed = np.dot(modes, commands).reshape(49, 49)
        masked_reconstructed, display_reconstructed = apply_mask(reconstructed, gridMask)
        
        # Calculate fitting error
        residual = display_wavefront - display_reconstructed
        rms_error = np.sqrt(np.nanmean(residual**2))
        print(f"RMS fitting error for {wavefront_name}: {rms_error:.6f}")
        
        # Plot the results
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original wavefront
        im0 = axes[0].imshow(display_wavefront, cmap='RdBu')
        axes[0].set_title(f"Original {wavefront_name} Wavefront")
        plt.colorbar(im0, ax=axes[0])
        
        # Reconstructed wavefront
        im1 = axes[1].imshow(display_reconstructed, cmap='RdBu')
        axes[1].set_title(f"Reconstructed {wavefront_name} Wavefront")
        plt.colorbar(im1, ax=axes[1])
        
        # Residual error
        im2 = axes[2].imshow(residual, cmap='RdBu')
        axes[2].set_title(f"Residual Error (RMS: {rms_error:.6f})")
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.show()
        
        return rms_error
    
    # Create a simple wavefront (OPD) to fit
    print("\nCreating sample wavefronts...")
    x, y = np.meshgrid(np.linspace(-1, 1, 49), np.linspace(-1, 1, 49))
    
    # Defocus wavefront
    radius_squared = x**2 + y**2
    defocus = radius_squared * 200  # Simple defocus wavefront
    
    # Tilt wavefront (x-direction tilt)
    tilt_x = x * 200  # Simple x-direction tilt
    
    # Process each wavefront
    tilt_error = process_wavefront("X-Tilt", tilt_x, fit, gridMask)
    defocus_error = process_wavefront("Defocus", defocus, fit, gridMask)

    # Compare results
    print("\nComparison of fitting errors:")
    print(f"X-Tilt RMS error: {tilt_error:.6f}")
    print(f"Defocus RMS error: {defocus_error:.6f}")
    
    print("\nExample completed successfully!")
    
    print(f"\nModes shape before applying grid mask: {modes.shape}")
    # Change the modes size with only valid elements of the gridMask 
    modes = modes[gridMask.flatten(), :]
    fit.modes = modes
    print(f"Modes shape after applying grid mask: {modes.shape}")
    
    # Generate a fitting matrix (pseudo-inverse of the influence functions)
    print("\nRecalculating fitting matrix...")
    fit.F = np.linalg.pinv(modes)
    print(f"Fitting matrix shape: {fit.F.shape}")