#tomographicReconstructor.py

"""
This module contains the tomographicReconstructor class for computing tomographic reconstructors
for adaptive optics systems. It supports both LTAO and MOAO configurations, with options for
model-based and interaction matrix-based reconstruction approaches.
"""

import yaml
import numpy as np
import logging
import matplotlib.pyplot as plt
from pyTomoAO.atmosphereParametersClass import atmosphereParameters
from pyTomoAO.lgsAsterismParametersClass import lgsAsterismParameters
from pyTomoAO.lgsWfsParametersClass import lgsWfsParameters 
from pyTomoAO.tomographyParametersClass import tomographyParameters
from pyTomoAO.dmParametersClass import dmParameters
from pyTomoAO.fitting import fitting
from scipy.io import loadmat


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:  
    CUDA = True
    import cupy as cp
    from pyTomoAO.tomographyUtilsGPU import _auto_correlation, _cross_correlation, \
        _build_reconstructor_model, _build_reconstructor_im, _sparseGradientMatrixAmplitudeWeighted
    logger.info("\nCUDA is available. Using GPU for computations.")
except:
    CUDA = False
    from pyTomoAO.tomographyUtilsCPU import _auto_correlation, _cross_correlation, \
        _build_reconstructor_model, _build_reconstructor_im, _sparseGradientMatrixAmplitudeWeighted
    logger.info("\nCUDA is not available. Using CPU for computations.")

class tomographicReconstructor:
    """
    A class for computing tomographic reconstructors for adaptive optics systems.
    
    This class computes a tomographic reconstructor from multiple Shack-Hartmann 
    wavefront sensors based on the turbulence model given by atmospheric parameters.
    The reconstruction can be done using either a model-based approach or an 
    interaction matrix (IM) based approach. The class supports both LTAO and MOAO
    configurations, with GPU acceleration using CUDA if available.
    
    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file containing all necessary parameters for
        the tomographic reconstruction.
    logger : logging.Logger, optional
        Logger object for logging messages (default is the module-level logger)

    Returns
    -------
    None
        Initializes the tomographicReconstructor object with the specified configuration.

    Notes
    -----
    The class maintains several internal attributes:
    - _reconstructor : numpy.ndarray
        The tomographic reconstructor matrix
    - _gridMask : numpy.ndarray
        Grid mask used for reconstruction
    - _wavefront2Meter : float
        Conversion factor from wavefront to meters
    - fit : fitting
        Fitting object for DM influence functions
    - modes : numpy.ndarray
        Influence function modes
    - method : str
        Reconstruction method ("Model" or "IM")
    - _FR : numpy.ndarray
        Combined fitting and reconstructor matrix
    """
    # Constructor
    def __init__(self, config_file, logger=logger):
        """
        Initialize the tomographicReconstructor with a configuration file.
        
        Parameters
        ----------
        config_file : str
            Path to the YAML configuration file
        logger : logging.Logger, optional
            Logger object for logging messages (default is the module-level logger)
        """
        logger.info("\n-->> Initializing reconstructor object <<--")
        # Load configuration
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize parameters
        self._initialize_parameters()

        # Initialize properties
        self.valid_constructor_type = [np.float32, np.float64]
        self._reconstructor = None
        self._wavefront2Meter = None
        self._gridMask = None
        self.fit = None
        self.modes = None
        self.method = None
        #self.R = None # Reconstructor
        self._FR = None # Fitting * Reconstructor

    def _initialize_parameters(self):
        """
        Initialize all parameter classes from the configuration file.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Initializes all parameter classes (atmParams, lgsAsterismParams, lgsWfsParams,
            tomoParams, dmParams) with values from the configuration file.
        """
        try:
            self.atmParams = atmosphereParameters(self.config)
            logger.info("\nSuccessfully initialized Atmosphere parameters.")
            logger.info(self.atmParams)
        except (ValueError, TypeError) as e:
            logger.error(f"Configuration Error in Atmosphere parameters: {e}")

        try:
            self.lgsAsterismParams = lgsAsterismParameters(self.config, self.atmParams)
            logger.info("\nSuccessfully initialized LGS asterism parameters.")
            logger.info(self.lgsAsterismParams) 
        except (ValueError, TypeError) as e:
            logger.error(f"\nConfiguration Error in LGS asterism parameters: {e}")

        try:
            self.lgsWfsParams = lgsWfsParameters(self.config, self.lgsAsterismParams)
            logger.info("\nSuccessfully initialized LGS WFS parameters.")
            logger.info(self.lgsWfsParams)
        except (ValueError, TypeError) as e:
            logger.error(f"\nConfiguration Error in LGS WFS parameters: {e}")

        try:
            self.tomoParams = tomographyParameters(self.config)
            logger.info("\nSuccessfully initialized Tomography parameters.")
            logger.info(self.tomoParams) 
        except (ValueError, TypeError) as e:
            logger.error(f"\nConfiguration Error in Tomography parameters: {e}")
            
        try:
            self.dmParams = dmParameters(self.config)
            logger.info("\nSuccessfully initialized DM parameters.")
            logger.info(self.dmParams)
        except (ValueError, TypeError) as e:
            logger.error(f"\nConfiguration Error in DM parameters: {e}")
        logger.info("\nAll parameters initialized successfully.")
    # ======================================================================
    # Properties
    @property
    def reconstructor(self):
        """
        Get the tomographic reconstructor matrix.
        If not already computed, this will build the reconstructor.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            The tomographic reconstructor matrix
        """
        if self._reconstructor is None:
            self.build_reconstructor()
        logger.debug("Accessing the reconstructor property.")
        return self._reconstructor

    @reconstructor.setter
    def reconstructor(self, value):
        """
        Set the tomographic reconstructor matrix.

        Parameters
        ----------
        value : numpy.ndarray
            The reconstructor matrix to set. Must be a 2D numpy array of float type.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the provided value is not a 2D numpy array of valid float type.
        """
        logger.debug("Setting the reconstructor property.")
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.dtype in self.valid_constructor_type:
            self._reconstructor = value
        else:
            logger.error("Invalid reconstructor value. Must be a 2D numpy array of floats.")
            raise ValueError("Reconstructor must be a 2D numpy array of floats.")

    @property
    def R(self):
        """
        Alias for the reconstructor property.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            The tomographic reconstructor matrix
        """
        logger.debug("Accessing the R property.")
        return self.reconstructor

    @R.setter
    def R(self, value):
        """
        Alias setter for the reconstructor property.

        Parameters
        ----------
        value : numpy.ndarray
            The reconstructor matrix to set. Must be a 2D numpy array of float type.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the provided value is not a 2D numpy array of valid float type.
        """
        logger.debug("Setting the R property.")
        self.reconstructor = value
        
    @property
    def FR(self):
        """
        Get the fitting-reconstructor matrix.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            The fitting-reconstructor matrix
        """
        logger.debug("Accessing the FR property.")
        return self._FR
    
    @FR.setter
    def FR(self, value):
        """
        Set the fitting-reconstructor matrix.

        Parameters
        ----------
        value : numpy.ndarray
            The fitting-reconstructor matrix to set

        Returns
        -------
        None
        """
        logger.debug("Setting the FR property.")
        self._FR = value

    @property
    def gridMask(self):
        """
        Get the grid mask used for reconstruction.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            The grid mask for reconstruction
        """
        if self._gridMask is None and self._reconstructor is not None:
            return self._gridMask
        else:
            # Build reconstructor if needed
            self.reconstructor  
            return self._gridMask
    # ======================================================================
    # Magic Methods
    # Getters and Setters
    def __getattr__(self, name):
        """
        Forwards attribute access to parameter classes if they contain the requested attribute.
        
        Parameters
        ----------
        name : str
            Name of the attribute to get
            
        Returns
        -------
        Any
            Value of the requested attribute from the appropriate parameter class
            
        Raises
        ------
        AttributeError
            If the attribute is not found in any parameter class
        """
        logger.debug("Getting attribute '%s' from parameter classes.", name)

        # List parameter classes that are already initialized
        param_classes = []
        for param_name in ['tomoParams', 'lgsWfsParams', 'atmParams', 'lgsAsterismParams']:
            if hasattr(self, param_name) and getattr(self, param_name) is not None:
                param_classes.append(getattr(self, param_name))

        # Check each parameter class for the attribute
        for param in param_classes:
            if hasattr(param, name):
                return getattr(param, name)

        logger.error("Attribute '%s' not found in parameter classes.", name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """
        Forwards attribute setting to parameter classes if they contain the specified attribute.
        When setting nLGS, ensures all parameter classes that have this attribute are updated.
        
        Parameters
        ----------
        name : str
            Name of the attribute to set
        value : Any
            Value to set for the attribute
            
        Returns
        -------
        None
            
        Raises
        ------
        ValueError
            If setting nLGS to a negative value
        """
        logger.debug("Setting attribute '%s'.", name)

        # These attributes are always set directly on the class
        special_attrs = ['tomoParams', 'lgsWfsParams', 'atmParams', 'lgsAsterismParams', '_reconstructor', '_gridMask', '_wavefront2Meter', 'config']
        if name in special_attrs:
            super().__setattr__(name, value)
            return

        # Special handling for nLGS to ensure all relevant parameter classes are updated
        if name == 'nLGS':
            if value < 0:
                raise ValueError("nLGS must be a non-negative integer.")

            # Convert to integer
            value = int(value)

            # Update nLGS in all parameter classes that have this attribute
            attr_set = False
            for param_name in ['tomoParams', 'lgsWfsParams', 'atmParams', 'lgsAsterismParams']:
                if hasattr(self, param_name) and getattr(self, param_name) is not None:
                    param = getattr(self, param_name)
                    if hasattr(param, name):
                        setattr(param, name, value)
                        attr_set = True

            # If attribute wasn't set in any parameter class, set it on the main class
            if not attr_set:
                super().__setattr__(name, value)
        else:
            # Check if attribute exists in any parameter class
            attr_set = False
            for param_name in ['tomoParams', 'lgsWfsParams', 'atmParams', 'lgsAsterismParams']:
                if hasattr(self, param_name) and getattr(self, param_name) is not None:
                    param = getattr(self, param_name)
                    if hasattr(param, name):
                        setattr(param, name, value)
                        attr_set = True
                        break

            # If attribute wasn't set in any parameter class, set it on the main class
            if not attr_set:
                super().__setattr__(name, value)
    # ======================================================================
    # Class Methods
    def sparseGradientMatrixAmplitudeWeighted(self, amplMask=None, overSampling=2, validLenslet=None):
        """
        Computes the sparse gradient matrix (3x3 or 5x5 stencil) with amplitude mask.

        Parameters
        ----------
        amplMask : numpy.ndarray, optional
            Amplitude mask to be applied to the gradient matrix
        overSampling : int, optional
            Oversampling factor (default is 2)
        validLenslet : numpy.ndarray, optional
            Valid lenslet map. If None, uses self.lgsWfsParams.validLLMapSupport

        Returns
        -------
        tuple
            A tuple containing:
            - Gamma : scipy.sparse.csr_matrix
                The sparse gradient matrix
            - gridMask : numpy.ndarray
                The grid mask used for the computation
        """
        logger.info("\n-->> Computing sparse gradient matrix <<--")
        # Use the provided validLenslet if specified, otherwise use the class attribute
        validLenslet = validLenslet if validLenslet is not None else self.lgsWfsParams.validLLMapSupport
        
        Gamma, gridMask = _sparseGradientMatrixAmplitudeWeighted(validLenslet, 
                                                                amplMask, 
                                                                overSampling)
        self._gridMask = gridMask
        self.Gamma = Gamma
        return Gamma, gridMask
    
    def auto_correlation(self):
        """
        Computes the auto-correlation meta-matrix for tomographic atmospheric reconstruction.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            The auto-correlation matrix (Cxx)
        """
        logger.info("\n-->> Computing auto-correlation meta-matrix <<--")
        Cxx = _auto_correlation(self.tomoParams, self.lgsWfsParams, self.atmParams, 
                                self.lgsAsterismParams, self.gridMask)     
        self.Cxx = Cxx
        return Cxx
    
    def cross_correlation(self, gridMask=None):
        """
        Computes the cross-correlation meta-matrix for tomographic atmospheric reconstruction.

        Parameters
        ----------
        gridMask : numpy.ndarray, optional
            Grid mask to be used in the computation. If None, uses self.gridMask

        Returns
        -------
        numpy.ndarray
            The cross-correlation matrix (Cox)
        """
        logger.info("\n-->> Computing cross-correlation meta-matrix <<--")
        Cox = _cross_correlation(self.tomoParams, self.lgsWfsParams, self.atmParams,
                                self.lgsAsterismParams, gridMask)
        self.Cox = Cox
        return Cox

    # Build Reconstructor
    def build_reconstructor(self, IM=None, use_float32=False):
        """
        Build the tomographic reconstructor based on parameters.

        Parameters
        ----------
        IM : numpy.ndarray, optional
            Interaction matrix for interaction matrix-based reconstructor.
            If None, a model-based reconstructor is built (default is None)
        use_float32 : bool, optional
            Whether to use float32 precision for computations to reduce memory usage
            (default is False, which uses float64)

        Returns
        -------
        numpy.ndarray
            The computed tomographic reconstructor matrix

        Notes
        -----
        This method computes different internal matrices depending on whether
        the model-based or IM-based approach is used:
        - Model-based: Gamma, gridMask, Cxx, Cox, Cnz, RecStatSA
        - IM-based: gridMask, Cxx, Cox, Cnz, RecStatSA
        """
        if IM is None:
            # Model based reconstructor
            logger.info("\n-->> Computing model based reconstructor <<--")
            if CUDA:
                _reconstructor, Gamma, gridMask, Cxx, Cox, Cnz, RecStatSA = \
                _build_reconstructor_model(self.tomoParams, self.lgsWfsParams, 
                                    self.atmParams, self.lgsAsterismParams, use_float32=True)
            else:
                _reconstructor, Gamma, gridMask, Cxx, Cox, Cnz, RecStatSA = \
                _build_reconstructor_model(self.tomoParams, self.lgsWfsParams, 
                                    self.atmParams, self.lgsAsterismParams)
            self.method = "Model"
            self._reconstructor = _reconstructor
            self.Gamma = Gamma
            self._gridMask = gridMask
            self.Cxx = Cxx
            self.Cox = Cox
            self.CnZ = Cnz
            self.RecStatSA = RecStatSA
            logger.info("\n-->> Model based reconstructor computed <<--")
        else:
            # IM based reconstructor
            logger.info("\n-->> Computing IM based reconstructor <<--")
            # load IM
            self.IM = IM
            if CUDA:
                _reconstructor, gridMask, Cxx, Cox, Cnz, RecStatSA = \
                _build_reconstructor_im(self.IM, self.tomoParams, self.lgsWfsParams, 
                                    self.atmParams, self.lgsAsterismParams, self.dmParams, use_float32=True)
            else:
                _reconstructor, gridMask, Cxx, Cox, Cnz, RecStatSA = \
                _build_reconstructor_im(self.IM, self.tomoParams, self.lgsWfsParams, 
                                    self.atmParams, self.lgsAsterismParams, self.dmParams)
            self.method = "IM"
            self._reconstructor = _reconstructor
            self._gridMask = gridMask
            self.Cxx = Cxx
            self.Cox = Cox
            self.CnZ = Cnz
            self.RecStatSA = RecStatSA
            logger.info("\n-->> IM based reconstructor computed <<--")
        return _reconstructor

    # Assemble Reconstructor and Fitting
    def assemble_reconstructor_and_fitting(self, nChannels=4, slopesOrder="simu", scalingFactor=1.65e7):
        """
        Assemble the reconstructor and fitting matrices together.

        Parameters
        ----------
        nChannels : int, optional
            Number of wavefront sensor channels (default is 4)
        slopesOrder : str, optional
            Order of slopes in the input data. Options are:
            - "keck": [slopeXY, ..., slopeXY] interleaved X,Y slopes
            - "simu": [slopeX, slopeY] all X slopes followed by all Y slopes
            - "inverted": [slopeY, slopeX] all Y slopes followed by all X slopes
            (default is "simu")
        scalingFactor : float, optional
            Scaling factor applied to the reconstructor (default is 1.65e7)

        Returns
        -------
        numpy.ndarray
            The assembled reconstructor and fitting matrix (FR)
            
        Raises
        ------
        ValueError
            If an invalid slopes order is provided
        """
        # test if reconstructor is already built
        if self._reconstructor is None:
            self.build_reconstructor()
        # test if fitting is already built
        if self.fit is None:
            self.fit = fitting(self.dmParams,logger=logger)
            logger.info("\n-->> Assembling Reconstructor and Fitting <<--")
        
        # Setup the influence functions and mask them to the grid
        logger.info("\nCalculating influence functions")
        self.modes = self.fit.set_influence_function(resolution=self.gridMask.shape[0],
                                                    display=False, sigma1=0.5*2, sigma2=0.85*2)
        self.modes = self.modes[self.gridMask.flatten(), :]
        logger.info(f"\nModes shape after applying grid mask: {self.modes.shape}")
        # Generate a fitting matrix (pseudo-inverse of the influence functions)
        logger.info("\nCalculating fitting matrix")
        self.fit.F = np.linalg.pinv(self.modes)
        logger.info(f"\nFitting matrix shape: {self.fit.F.shape}")
        
        # prepare the reconstructor for single channel
        if nChannels == 1:
            self.reconstructor = self._reconstructor[:, :self.lgsWfsParams.nValidSubap*2]
        
        # Rearrange the reconstructor to accomodate slopes = [slopeX, slopesY]
        if slopesOrder == "simu":
            # Swap X and Y blocks 
            self.reconstructor = self.swap_xy_blocks(self._reconstructor, 
                                                    self.lgsWfsParams.nValidSubap, 
                                                    nChannels)
            # Generate the reconstructor with fitting
            self.FR = -self.fit.F @ self.reconstructor * scalingFactor
        # Rearrange the reconstructor to accomodate slopes = [slopesXY,..,slopesXY]
        elif slopesOrder == "keck":
            # Swap X and Y blocks 
            self._reconstructor = self.swap_xy_blocks(self._reconstructor, 
                                                    self.lgsWfsParams.nValidSubap, 
                                                    nChannels)
            # Rearrange the rows into [XY, ..., XY]
            self.reconstructor = np.apply_along_axis(self.sort_row, 1, self._reconstructor)
            # Generate the reconstructor with fitting
            self.FR = -self.fit.F @ self.reconstructor * scalingFactor
        # Rearrange the reconstructor to accomodate slopes = [slopeY, slopesX]    
        elif slopesOrder == "inverted":
            self.reconstructor = self._reconstructor
            # Generate the reconstructor with fitting
            self.FR = -self.fit.F @ self.reconstructor * scalingFactor
        else:
            raise ValueError("Invalid slopes order. Use 'simu', 'keck' or 'inverted'.")
        logger.info("\n-->> Reconstructor and Fitting assembled <<--")
        
        return self._FR
    
    # Sort row into [XY, ..., XY]
    def sort_row(self, row):
        """
        Sorts a row into [XY, ..., XY] format (interleaved X and Y measurements).

        Parameters
        ----------
        row : numpy.ndarray
            Input row with X and Y measurements in separate blocks

        Returns
        -------
        numpy.ndarray
            Row rearranged into interleaved [XY, ..., XY] format
        """
        row2 = row.copy()
        row2[::2] = row[:row.shape[0]//2]
        row2[1::2] = row[row.shape[0]//2:]
        return row2

    def swap_xy_blocks(self, matrix, n_valid_subap, nChannels=1):
        """
        Swap the X and Y column blocks in a matrix, preserving channel organization.

        Parameters
        ----------
        matrix : numpy.ndarray
            The input matrix to swap columns
        n_valid_subap : int
            Number of valid subapertures
        nChannels : int, optional
            Number of wavefront sensor channels (default is 1)

        Returns
        -------
        numpy.ndarray
            Matrix with swapped X and Y column blocks for each channel
        """
        new_col_order = []
        
        # Total columns per channel
        cols_per_channel = n_valid_subap * 2
        
        # Process each channel separately
        for ch in range(nChannels):
            # Calculate start index for this channel
            ch_start = ch * cols_per_channel
            
            # X columns are in the second half of each channel block
            cols_X = np.arange(ch_start + n_valid_subap, ch_start + 2 * n_valid_subap)
            
            # Y columns are in the first half of each channel block
            cols_Y = np.arange(ch_start, ch_start + n_valid_subap)
            
            # Swap X and Y for this channel
            new_col_order.extend(cols_X)
            new_col_order.extend(cols_Y)
        
        # Convert to numpy array and return reordered matrix
        new_col_order = np.array(new_col_order)
        return matrix[:, new_col_order]

    # Mask DM actuators
    def mask_DM_actuators(self, actuIndex):
        """
        Masks specific DM actuators in the reconstructor.

        Parameters
        ----------
        actuIndex : int or list of int
            Index or indices of the actuator(s) to be masked (set to zero)

        Returns
        -------
        numpy.ndarray
            The reconstructor with masked actuators

        Raises
        ------
        ValueError
            If the reconstruction method is not defined or the reconstructor is not built
        """
        if self.method == "IM":
            if self._reconstructor is None:
                logger.error("IM based reconstructor is not defined. Please build the reconstructor first.")
                raise ValueError("IM based reconstructor is not defined. Please build the reconstructor first.")
            else:
                logger.info("\n-->> Masking DM actuators <<--")
                # Mask the DM actuators
                self._reconstructor[actuIndex, :] = 0
                return self._reconstructor
        elif self.method == "Model":
            if self._FR is None:
                logger.error("Model based reconstructor is not defined. Please build the reconstructor first.")
                raise ValueError("Model based reconstructor is not defined. Please build the reconstructor first.")
            else:
                logger.info("\n-->> Masking DM actuators <<--")
                # Mask the DM actuators
                self._FR[actuIndex, :] = 0
                return self._FR
        else:
            logger.error("Invalid method. Please build the reconstructor first.")
            raise ValueError("Invalid method. Please build the reconstructor first.")

    # Reconstruct Wavefront
    def reconstruct_wavefront(self, slopes):
        """
        Reconstruct the wavefront from slope measurements using the computed reconstructor.

        Parameters
        ----------
        slopes : numpy.ndarray
            Slope measurements from wavefront sensors

        Returns
        -------
        numpy.ndarray
            Reconstructed wavefront as a 2D array with NaN values where the grid mask is zero

        Raises
        ------
        ValueError
            If the reconstructor is not built
        """
        # Ensure reconstructor is built
        if self._reconstructor is None:
            self.build_reconstructor()

        # Reconstruct the wavefront
        wavefront = self._reconstructor @ slopes
        wavefront = wavefront.flatten()

        # Apply mask
        mask = np.array(self._gridMask*1, dtype=np.float64)
        ones_indices = np.where(mask == 1)
        mask[ones_indices] = wavefront

        # Set masked values to NaN for visualization
        mask[mask==0] = np.nan

        return mask

    # Visualize Commands
    def visualize_commands(self, slopes):
        """
        Visualize the DM commands derived from slope measurements.

        Parameters
        ----------
        slopes : numpy.ndarray
            Slope measurements from wavefront sensors

        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the visualization of DM commands as a bar plot
            and the DM surface as a 2D image

        Raises
        ------
        ValueError
            If the reconstruction method is not defined or the reconstructor is not built
        """
        # get the DM command
        if self.method == "Model":
            dm_commands = self.FR @ slopes
        elif self.method == "IM":
            dm_commands = self._reconstructor @ slopes
        else:
            raise ValueError("Invalid method. Please build the reconstructor first.")
        
        # project the commands on the DM surface
        cmd_mask = np.array(self.dmParams.validActuatorsSupport*1, dtype=np.float64)
        ones_indices = np.where(cmd_mask == 1)       
        cmd_mask[ones_indices] = dm_commands
        # Set masked values to NaN for visualization
        cmd_mask[cmd_mask==0] = np.nan
        # display the DM commands   
        fig, (ax1, ax2) = plt.subplots(1, 2) 
        # display the DM commands
        ax1.bar(np.arange(349),dm_commands)
        ax1.set_xlabel('DM actuator')
        ax1.set_ylabel('Command Value')
        ax1.set_title('DM Commands')
        # display the DM surface
        im2 = ax2.imshow(cmd_mask, cmap='RdBu', origin='lower')
        ax2.set_title('DM Surface')
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.colorbar(im2, ax=ax2, shrink=0.5)
        plt.tight_layout()
        ax1.set_aspect(0.375)
        return fig

    # Visualize Reconstruction
    def visualize_reconstruction(self, slopes, reference_wavefront=None):
        """
        Visualize the reconstruction results and optionally compare with reference.

        Parameters
        ----------
        slopes : numpy.ndarray
            Slope measurements from wavefront sensors
        reference_wavefront : numpy.ndarray, optional
            Reference wavefront for comparison

        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the visualization of reconstructed wavefront
        """
        # Reconstruct wavefront
        reconstructed_wavefront = self.reconstruct_wavefront(slopes)

        if reference_wavefront is None:
            # Single plot
            fig, ax = plt.subplots(figsize=(8, 6))
            img = ax.imshow(reconstructed_wavefront.T, origin='lower')
            fig.colorbar(img, ax=ax, fraction=0.046)
            ax.set_aspect('equal')
            ax.set_title(f'Reconstructed OPD\nMean value: {np.nanmean(reconstructed_wavefront)*1e9:.2f} [nm]')
        else:
            # Comparison plot
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

            img1 = ax1.imshow(reconstructed_wavefront.T, origin='lower')
            fig.colorbar(img1, ax=ax1, fraction=0.047)
            ax1.set_aspect('equal')
            ax1.set_title(f'Reconstructed OPD\nMean value: {np.nanmean(reconstructed_wavefront)*1e9:.2f} [nm]')

            img2 = ax2.imshow(reference_wavefront, origin='lower')
            fig.colorbar(img2, ax=ax2, fraction=0.047)
            ax2.set_aspect('equal')
            ax2.set_title(f'Reference OPD\nMean value: {np.nanmean(reference_wavefront)*1e9:.2f} [nm]')

            diff = reference_wavefront - reconstructed_wavefront.T
            img3 = ax3.imshow(diff, origin='lower')
            fig.colorbar(img3, ax=ax3, fraction=0.047)
            ax3.set_aspect('equal')
            ax3.set_title(f'Difference (Reference-Reconstructed)\nMean difference: {np.nanmean(diff)*1e9:.2f} [nm]')

        plt.tight_layout()
        return fig
    # ======================================================================
    # Test Methods
    def _test_against_matlab(self, matlab_data_dir):
        """
        Test the reconstructor against MATLAB results.

        Parameters
        ----------
        matlab_data_dir : str
            Directory containing MATLAB test data files

        Returns
        -------
        dict
            Dictionary containing test results for various matrices and components
        """
        logger.info("\nTesting reconstructor against MATLAB results...")
        results = {}

        # Test Gamma matrix
        try:
            mat_data = loadmat(f'{matlab_data_dir}/Gamma.mat')
            Gamma_matlab = mat_data['Gamma']
            gamma_test = np.allclose(Gamma_matlab.toarray(), self.Gamma.toarray())
            results['gamma_test'] = gamma_test
            logger.info(f"\nGamma matrix test: {'PASSED' if gamma_test else 'FAILED'}")
        except Exception as e:
            logger.error(f"Error testing Gamma matrix: {e}")
            results['gamma_test'] = False

        # Test auto-correlation matrix
        try:
            mat_data = loadmat(f'{matlab_data_dir}/Cxx.mat')
            Cxx_matlab = mat_data['Cxx']
            cxx_test = np.allclose(Cxx_matlab, self.Cxx, rtol=5e-4)
            results['cxx_test'] = cxx_test
            logger.info(f"\nAuto-correlation matrix test: {'PASSED' if cxx_test else 'FAILED'}")
        except Exception as e:
            logger.error(f"Error testing auto-correlation matrix: {e}")
            results['cxx_test'] = False

        # Test cross-correlation matrix
        try:
            mat_data = loadmat(f'{matlab_data_dir}/Cox.mat')
            Cox_matlab = mat_data['Cox']
            cox_test = np.allclose(Cox_matlab, self.Cox, rtol=5e-4)
            results['cox_test'] = cox_test
            logger.info(f"\nCross-correlation matrix test: {'PASSED' if cox_test else 'FAILED'}")
        except Exception as e:
            logger.error(f"Error testing cross-correlation matrix: {e}")
            results['cox_test'] = False

        # Test CnZ matrix
        try:
            mat_data = loadmat(f'{matlab_data_dir}/CnZ.mat')
            CnZ_matlab = mat_data['CnZ']
            cnz_test = np.allclose(CnZ_matlab, self.CnZ, rtol=5e-4)
            results['cnz_test'] = cnz_test
            logger.info(f"\nCnZ test: {'PASSED' if cnz_test else 'FAILED'}")
        except Exception as e:
            logger.error(f"Error testing CnZ matrix: {e}")
            results['cnz_test'] = False

        # Test invCss matrix
        try:
            mat_data = loadmat(f'{matlab_data_dir}/invCss.mat')
            invCss_matlab = mat_data['invCss']
            invCss_test = np.allclose(invCss_matlab, self.invCss, atol=5e-3)
            results['invCss_test'] = invCss_test
            logger.info(f"\ninvCss test: {'PASSED' if invCss_test else 'FAILED'}")
        except Exception as e:
            logger.error(f"Error testing invCss matrix: {e}")
            results['invCss_test'] = False

        # Test reconstructor matrix
        try:
            mat_data = loadmat(f'{matlab_data_dir}/RecStatSAsuperRes.mat')
            RecStatSA_matlab = mat_data['RecStatSAsuperRes']
            rec_test = np.allclose(RecStatSA_matlab, self.RecStatSA, atol=5e-3)
            results['rec_test'] = rec_test
            logger.info(f"\nReconstructor matrix test: {'PASSED' if rec_test else 'FAILED'}")
        except Exception as e:
            logger.error(f"Error testing reconstructor matrix: {e}")
            results['rec_test'] = False

        # Test with slopes generated with Matlab
        try:
            for i in range(2, 4):
                mat_data = loadmat(f'{matlab_data_dir}/slopes_{i}.mat')
                slopes = mat_data[f'slopes_{i}']

                # Load reconstructed wavefront from Matlab
                mat_data = loadmat(f'{matlab_data_dir}/wavefront_{i}.mat')
                wavefront = mat_data[f'wavefront_{i}']

                # Visualize the comparison
                fig = self.visualize_reconstruction(slopes, wavefront)
                plt.show()

        except Exception as e:
            logger.error(f"Error testing with slopes: {e}")

        return results


# Example usage
if __name__ == "__main__":
    # Use a path relative to the script's location
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located
    config_path = os.path.join(script_dir, "..", "examples", "benchmark", "tomography_config_kapa_single_channel.yaml")
    
    # Check if the file exists, otherwise prompt for a different path
    if not os.path.exists(config_path):
        print(f"Warning: Configuration file not found at {config_path}")
        print("Current directory is:", os.getcwd())
        print("Please provide the full path to your configuration file:")
        user_config_path = input()
        if user_config_path and os.path.exists(user_config_path):
            config_path = user_config_path
        else:
            print("No valid configuration file provided. Exiting.")
            import sys
            sys.exit(1)
    
    # Create the reconstructor
    reconstructor = tomographicReconstructor(config_path)

    # Build the model based reconstructor. To build the IM based reconstructor,
    # pass the IM matrix as an argument.
    # R = reconstructor.build_reconstructor(IM, use_float32=True) 
    R = reconstructor.build_reconstructor(use_float32=True)
    print(f"Reconstructor matrix shape: {R.shape}")

    # This step is only required for the model based reconstructor.
    # Assemble the reconstructor and fitting for single channel case
    reconstructor.assemble_reconstructor_and_fitting(nChannels=1, 
                                                        slopesOrder="simu", 
                                                        scalingFactor=1.5e7)
    # mask central actuator
    reconstructor.mask_DM_actuators(174)
    FR = reconstructor.FR
    
    print(f"Reconstructor and fitting matrix shape: {FR.shape}")

    # Visualize the reconstructor
    fig = plt.figure(figsize=(10, 8))
    im = plt.imshow(FR)
    cbar = plt.colorbar(im, fraction=0.028, pad=0.02)
    plt.title('Fitting * Reconstructor (Single Channel)')
    plt.xlabel('Slopes')
    plt.ylabel('Actuators')
    plt.tight_layout()
    plt.show()
    
    # Build the IM based reconstructor
    # IM = np.load('../sandbox/IM_sim.npy')
    # R = reconstructor.build_reconstructor(IM, use_float32=True)
    # print(f"Reconstructor matrix shape: {R.shape}")
    
    # Test against MATLAB results if needed
    # results = reconstructor._test_against_matlab('/Users/urielconod/tomographyDataTest')

    # Example of wavefront reconstruction from slopes
    # (assuming you have slopes data available)
    # slopes = ...
    # wavefront = reconstructor.reconstruct_wavefront(slopes)
    # fig = reconstructor.visualize_reconstruction(slopes)
    # plt.show()
    