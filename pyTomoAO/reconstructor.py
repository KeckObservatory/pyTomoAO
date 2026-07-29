"""
This module contains the tomographicReconstructor class for computing tomographic reconstructors
for adaptive optics systems, with options for model-based and interaction matrix-based
reconstruction approaches.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.io import loadmat

from pyTomoAO import backend
from pyTomoAO.atmosphereParametersClass import atmosphereParameters
from pyTomoAO.dm_fitting import fitting
from pyTomoAO.dmParametersClass import dmParameters
from pyTomoAO.lgsAsterismParametersClass import lgsAsterismParameters
from pyTomoAO.lgsWfsParametersClass import lgsWfsParameters
from pyTomoAO.tomographyParametersClass import tomographyParameters

# Module logger. Handlers and levels are the application's business; the package
# only attaches a NullHandler (see pyTomoAO/__init__.py).
logger = logging.getLogger(__name__)

#: Whether the CuPy kernels are importable. Read-only: the backend a given reconstructor
#: actually uses is resolved per instance (see ``force_cpu``) and exposed as ``rec.backend``.
CUDA = backend.cuda_available()


class tomographicReconstructor:
    """
    A class for computing tomographic reconstructors for adaptive optics systems.

    This class computes a tomographic reconstructor from multiple Shack-Hartmann
    wavefront sensors based on the turbulence model given by atmospheric parameters.
    The reconstruction can be done using either a model-based approach or an
    interaction matrix (IM) based approach.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file containing all necessary parameters for
        the tomographic reconstruction.
    logger : logging.Logger, optional
        Logger object for logging messages (default is the module-level logger)
    force_cpu : bool, optional
        Force CPU usage even when CUDA is available (default is False)

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
    def __init__(self, config_file, logger=logger, force_cpu=False):
        """
        Initialize the tomographicReconstructor with a configuration file.

        Parameters
        ----------
        config_file : str
            Path to the YAML configuration file
        logger : logging.Logger, optional
            Logger object for logging messages (default is the module-level logger)
        force_cpu : bool, optional
            Force CPU usage even when CUDA is available (default is False)
        """
        # First, initialize the object's dictionary directly to avoid attribute access issues
        object.__setattr__(self, "_reconstructor", None)
        object.__setattr__(self, "_gridMask", None)
        object.__setattr__(self, "_wavefront2Meter", None)
        object.__setattr__(self, "fit", None)
        object.__setattr__(self, "modes", None)
        object.__setattr__(self, "method", None)
        object.__setattr__(self, "_FR", None)
        object.__setattr__(self, "valid_constructor_type", [np.float32, np.float64])
        object.__setattr__(self, "atmParams", None)
        object.__setattr__(self, "lgsAsterismParams", None)
        object.__setattr__(self, "lgsWfsParams", None)
        object.__setattr__(self, "tomoParams", None)
        object.__setattr__(self, "dmParams", None)

        # Resolve the kernel backend for this instance. This used to flip a module-level
        # CUDA flag, which did not work: the GPU functions were already bound at import, so
        # force_cpu logged that it was forcing the CPU and then ran on the GPU anyway. It
        # also changed the backend for every other reconstructor in the process.
        object.__setattr__(self, "_backend", backend.get_backend("cpu" if force_cpu else "auto"))
        if force_cpu:
            logger.info("\nForcing CPU usage for computations.")

        logger.info("\n-->> Initializing reconstructor object <<--")
        # Load configuration
        with open(config_file) as f:
            config_data = yaml.safe_load(f)
            object.__setattr__(self, "config", config_data)

        # Initialize parameters
        self._initialize_parameters()

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
            atm_params = atmosphereParameters(self.config)
            object.__setattr__(self, "atmParams", atm_params)
            logger.info("\nSuccessfully initialized Atmosphere parameters.")
            logger.info(atm_params)
        except (ValueError, TypeError) as e:
            logger.error(f"Configuration Error in Atmosphere parameters: {e}")
            raise  # Re-raise to prevent continuing with invalid parameters

        try:
            lgs_asterism_params = lgsAsterismParameters(self.config, self.atmParams)
            object.__setattr__(self, "lgsAsterismParams", lgs_asterism_params)
            logger.info("\nSuccessfully initialized LGS asterism parameters.")
            logger.info(lgs_asterism_params)
        except (ValueError, TypeError) as e:
            logger.error(f"\nConfiguration Error in LGS asterism parameters: {e}")
            raise  # Re-raise to prevent continuing with invalid parameters

        try:
            lgs_wfs_params = lgsWfsParameters(self.config, self.lgsAsterismParams)
            object.__setattr__(self, "lgsWfsParams", lgs_wfs_params)
            logger.info("\nSuccessfully initialized LGS WFS parameters.")
            logger.info(lgs_wfs_params)
        except (ValueError, TypeError) as e:
            logger.error(f"\nConfiguration Error in LGS WFS parameters: {e}")
            raise  # Re-raise to prevent continuing with invalid parameters

        try:
            tomo_params = tomographyParameters(self.config)
            object.__setattr__(self, "tomoParams", tomo_params)
            logger.info("\nSuccessfully initialized Tomography parameters.")
            logger.info(tomo_params)
        except (ValueError, TypeError) as e:
            logger.error(f"\nConfiguration Error in Tomography parameters: {e}")
            raise  # Re-raise to prevent continuing with invalid parameters

        try:
            dm_params = dmParameters(self.config)
            object.__setattr__(self, "dmParams", dm_params)
            logger.info("\nSuccessfully initialized DM parameters.")
            logger.info(dm_params)
        except (ValueError, TypeError) as e:
            logger.error(f"\nConfiguration Error in DM parameters: {e}")
            raise  # Re-raise to prevent continuing with invalid parameters

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
        if (
            isinstance(value, np.ndarray)
            and value.ndim == 2
            and value.dtype in self.valid_constructor_type
        ):
            super().__setattr__("_reconstructor", value)
        else:
            logger.error("Invalid reconstructor value. Must be a 2D numpy array of floats.")
            raise ValueError("Reconstructor must be a 2D numpy array of floats.")

    @property
    def backend(self):
        """
        Name of the kernel backend this reconstructor uses.

        Parameters
        ----------
        None

        Returns
        -------
        str
            ``"gpu"`` or ``"cpu"``. Fixed when the object is constructed; pass
            ``force_cpu=True`` to select the CPU kernels even where CuPy is available.
        """
        return self._backend.name

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
        super().__setattr__("_FR", value)

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
        # Accessing the property builds the reconstructor if it is not built yet
        _ = self.reconstructor
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
        # First log the request
        logger.debug(f"Getting attribute '{name}' from parameter classes.")

        # List of parameter class attributes
        param_attrs = ["tomoParams", "lgsWfsParams", "atmParams", "lgsAsterismParams", "dmParams"]

        # Check each parameter class for the attribute
        for param_name in param_attrs:
            try:
                # First check if the parameter object exists
                param = object.__getattribute__(self, param_name)
                # Then check if the parameter object has the requested attribute
                if param is not None and hasattr(param, name):
                    return getattr(param, name)
            except (AttributeError, TypeError):
                # Skip if the parameter object doesn't exist or isn't properly initialized
                continue

        # If we get here, the attribute wasn't found
        logger.error(f"Attribute '{name}' not found in parameter classes.")
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
        logger.debug(f"Setting attribute '{name}'.")

        # These attributes are always set directly on the class
        special_attrs = [
            "_reconstructor",
            "_gridMask",
            "_wavefront2Meter",
            "_backend",
            "config",
            "valid_constructor_type",
            "fit",
            "modes",
            "method",
            "_FR",
            "dmParams",
            "tomoParams",
            "lgsWfsParams",
            "atmParams",
            "lgsAsterismParams",
        ]

        if name in special_attrs:
            object.__setattr__(self, name, value)
            return

        # Special handling for nLGS to ensure all relevant parameter classes are updated
        if name == "nLGS":
            if value < 0:
                raise ValueError("nLGS must be a non-negative integer.")

            # Convert to integer
            value = int(value)

            # Update nLGS in all parameter classes that have this attribute
            attr_set = False
            param_attrs = [
                "tomoParams",
                "lgsWfsParams",
                "atmParams",
                "lgsAsterismParams",
                "dmParams",
            ]

            for param_name in param_attrs:
                try:
                    # Get the parameter object directly
                    param = object.__getattribute__(self, param_name)
                    if param is not None and hasattr(param, name):
                        setattr(param, name, value)
                        attr_set = True
                except (AttributeError, TypeError):
                    # Skip if parameter doesn't exist
                    continue

            # If attribute wasn't set in any parameter class, set it on the main class
            if not attr_set:
                object.__setattr__(self, name, value)
        else:
            # Check if attribute exists in any parameter class
            attr_set = False
            param_attrs = [
                "tomoParams",
                "lgsWfsParams",
                "atmParams",
                "lgsAsterismParams",
                "dmParams",
            ]

            for param_name in param_attrs:
                try:
                    # Get the parameter object directly
                    param = object.__getattribute__(self, param_name)
                    if param is not None and hasattr(param, name):
                        setattr(param, name, value)
                        attr_set = True
                        break
                except (AttributeError, TypeError):
                    # Skip if parameter doesn't exist
                    continue

            # If attribute wasn't set in any parameter class, set it on the main class
            if not attr_set:
                object.__setattr__(self, name, value)

    # ======================================================================
    # Class Methods
    def sparseGradientMatrixAmplitudeWeighted(
        self, amplMask=None, overSampling=2, validLenslet=None
    ):
        """Computes the sparse gradient matrix (3x3 or 5x5 stencil) with amplitude mask.

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
        validLenslet = (
            validLenslet if validLenslet is not None else self.lgsWfsParams.validLLMapSupport
        )

        Gamma, gridMask = self._backend._sparseGradientMatrixAmplitudeWeighted(
            validLenslet, amplMask, overSampling
        )
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
        Cxx = self._backend._auto_correlation(
            self.tomoParams,
            self.lgsWfsParams,
            self.atmParams,
            self.lgsAsterismParams,
            self.gridMask,
        )
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
        Cox = self._backend._cross_correlation(
            self.tomoParams, self.lgsWfsParams, self.atmParams, self.lgsAsterismParams, gridMask
        )
        self.Cox = Cox
        return Cox

    # Build Reconstructor
    def build_reconstructor(self, IM=None, use_float32=False, alpha=10):  # noqa: ARG002
        # NOTE: use_float32 is currently not forwarded. The GPU kernels are always
        # called with use_float32=True and the CPU kernels with their default. See
        # the "Precision" section of docs/source/user-guide/gpu.md.
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
        alpha : float, optional
            Regularization parameter for the reconstructor (default is 10)

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
        # Only the GPU kernels take use_float32; the CPU ones always work in float64.
        precision = {"use_float32": True} if self._backend.is_gpu else {}

        if IM is None:
            # Model based reconstructor
            logger.info("\n-->> Computing model based reconstructor <<--")
            _reconstructor, Gamma, gridMask, Cxx, Cox, Cnz, RecStatSA = (
                self._backend._build_reconstructor_model(
                    self.tomoParams,
                    self.lgsWfsParams,
                    self.atmParams,
                    self.lgsAsterismParams,
                    alpha=alpha,
                    **precision,
                )
            )
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
            _reconstructor, gridMask, Cxx, Cox, Cnz, RecStatSA = (
                self._backend._build_reconstructor_im(
                    self.IM,
                    self.tomoParams,
                    self.lgsWfsParams,
                    self.atmParams,
                    self.lgsAsterismParams,
                    self.dmParams,
                    alpha=alpha,
                    **precision,
                )
            )
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
    def assemble_reconstructor_and_fitting(
        self,
        nChannels=4,
        slopesOrder="simu",
        scalingFactor=1.65e7,
        stretch_factor=1.03,
        rotation=None,
        flip=None,
    ):
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
        stretch_factor : float, optional
            Stretch factor for the influence functions (default is 1.03)
        rotation : int, optional
            Rotation of the modes (0, 1, 2, or 3) to apply to the reconstructor
            (default is None, no rotation)

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
        # test if reconstruction method is "Model"
        if self.method != "Model":
            logger.error(
                "Reconstructor is not built using the model method. Please build it first."
            )
            raise ValueError(
                "Reconstructor is not built using the model method. Please build it first."
            )
        # test if fitting is already built
        if self.fit is None:
            self.fit = fitting(self.dmParams, logger=logger)
            logger.info("\n-->> Assembling Reconstructor and Fitting <<--")

        # Setup the influence functions and mask them to the grid
        logger.info("\nCalculating influence functions")
        self.modes = self.fit.set_influence_function(
            resolution=self.gridMask.shape[0],
            display=False,
            sigma1=0.5 * 2,
            sigma2=0.85 * 2,
            stretch_factor=stretch_factor,
        )

        if rotation is not None:
            # apply rotation of the modes (corresponding to a rotation of the DM
            # with respect to the WFS)
            reshaped_array = self.modes.T.reshape(
                self.fit.modes.shape[1], self.gridMask.shape[0], self.gridMask.shape[0]
            )
            rotated_array = np.zeros_like(reshaped_array)
            for i in range(self.modes.shape[1]):
                rotated_array[i] = np.rot90(reshaped_array[i], rotation)
                # flip the rotated array
                if flip is not None:
                    rotated_array[i] = np.flipud(rotated_array[i])
            self.modes = rotated_array.reshape(self.modes.shape[1], -1).T

        self.modes = self.modes[self.gridMask.flatten(), :]
        logger.info(f"\nModes shape after applying grid mask: {self.modes.shape}")

        # Generate a fitting matrix (pseudo-inverse of the influence functions)
        logger.info("\nCalculating fitting matrix")
        self.fit.F = np.linalg.pinv(self.modes)
        logger.info(f"\nFitting matrix shape: {self.fit.F.shape}")

        # The column reordering below is derived into a local rather than written back into
        # self._reconstructor. Assigning it back made the method non-idempotent: a second
        # call swapped the X and Y blocks a second time, silently undoing the first swap and
        # producing a different, wrong FR. self._reconstructor stays as build_reconstructor
        # left it, which is also what reconstruct_wavefront assumes.
        R = self._reconstructor

        # prepare the reconstructor for single channel
        if nChannels == 1:
            R = R[:, : self.lgsWfsParams.nValidSubap * 2]

        # Rearrange the reconstructor to accomodate slopes = [slopeX, slopeY]
        if slopesOrder == "simu":
            # Swap X and Y blocks
            R = self.swap_xy_blocks(R, self.lgsWfsParams.nValidSubap, nChannels)
        # Rearrange the reconstructor to accomodate slopes = [slopesXY,..,slopesXY]
        elif slopesOrder == "keck":
            # Swap X and Y blocks, then rearrange the rows into [XY, ..., XY]
            R = self.swap_xy_blocks(R, self.lgsWfsParams.nValidSubap, nChannels)
            R = np.apply_along_axis(self.sort_row, 1, R)
        # Slopes already ordered [slopeY, slopeX] need no rearranging
        elif slopesOrder != "inverted":
            logger.error("Invalid slopes order. Use 'simu', 'keck' or 'inverted'.")
            raise ValueError("Invalid slopes order. Use 'simu', 'keck' or 'inverted'.")

        # Generate the reconstructor with fitting
        self.FR = -self.fit.F @ R * scalingFactor
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
        row2[::2] = row[: row.shape[0] // 2]
        row2[1::2] = row[row.shape[0] // 2 :]
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
                logger.error(
                    "IM based reconstructor is not defined. Please build the reconstructor first."
                )
                raise ValueError(
                    "IM based reconstructor is not defined. Please build the reconstructor first."
                )
            logger.info("\n-->> Masking DM actuators <<--")
            # Mask the DM actuators
            self._reconstructor[actuIndex, :] = 0
            return self._reconstructor
        if self.method == "Model":
            if self._FR is None:
                msg = (
                    "Model based reconstructor is not defined. "
                    "Please build the reconstructor first."
                )
                logger.error(msg)
                raise ValueError(msg)
            logger.info("\n-->> Masking DM actuators <<--")
            # Mask the DM actuators
            self._FR[actuIndex, :] = 0
            return self._FR
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
        # test if reconstruction method is "Model"
        if self.method != "Model":
            logger.error(
                "Reconstructor is not built using the model method. Please build it first."
            )
            raise ValueError(
                "Reconstructor is not built using the model method. Please build it first."
            )
        # Reconstruct the wavefront
        wavefront = self._reconstructor @ slopes
        wavefront = wavefront.flatten()

        # Scatter onto the pupil grid, leaving points outside the mask as NaN. The mask
        # itself decides what is invalid: testing the reconstructed values for zero would
        # also blank out valid points that happen to reconstruct to exactly zero.
        valid = np.asarray(self._gridMask, dtype=bool)
        out = np.full(valid.shape, np.nan, dtype=np.float64)
        out[valid] = wavefront

        return out

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
            logger.error("Invalid method. Please build the reconstructor first.")
            raise ValueError("Invalid method. Please build the reconstructor first.")

        # Project the commands on the DM surface. As in reconstruct_wavefront, the actuator
        # map decides what is invalid, so that a genuinely zero command stays visible.
        valid_act = np.asarray(self.dmParams.validActuatorsSupport, dtype=bool)
        cmd_mask = np.full(valid_act.shape, np.nan, dtype=np.float64)
        cmd_mask[valid_act] = dm_commands
        # display the DM commands
        fig, (ax1, ax2) = plt.subplots(1, 2)
        # display the DM commands
        ax1.bar(np.arange(dm_commands.shape[0]), dm_commands)
        ax1.set_xlabel("DM actuator")
        ax1.set_ylabel("Command Value")
        ax1.set_title("DM Commands")
        # display the DM surface
        im2 = ax2.imshow(cmd_mask, cmap="RdBu", origin="lower")
        ax2.set_title("DM Surface")
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
            img = ax.imshow(reconstructed_wavefront, origin="lower")
            fig.colorbar(img, ax=ax, fraction=0.046)
            ax.set_aspect("equal")
            mean_nm = np.nanmean(reconstructed_wavefront) * 1e9
            ax.set_title(f"Reconstructed OPD\nMean value: {mean_nm:.2f} [nm]")
        else:
            # Comparison plot
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

            img1 = ax1.imshow(reconstructed_wavefront, origin="lower")
            fig.colorbar(img1, ax=ax1, fraction=0.047)
            ax1.set_aspect("equal")
            mean_nm = np.nanmean(reconstructed_wavefront) * 1e9
            ax1.set_title(f"Reconstructed OPD\nMean value: {mean_nm:.2f} [nm]")

            img2 = ax2.imshow(reference_wavefront, origin="lower")
            fig.colorbar(img2, ax=ax2, fraction=0.047)
            ax2.set_aspect("equal")
            ax2.set_title(
                f"Reference OPD\nMean value: {np.nanmean(reference_wavefront) * 1e9:.2f} [nm]"
            )

            diff = reference_wavefront - reconstructed_wavefront
            img3 = ax3.imshow(diff, origin="lower")
            fig.colorbar(img3, ax=ax3, fraction=0.047)
            ax3.set_aspect("equal")
            diff_nm = np.nanmean(diff) * 1e9
            ax3.set_title(
                f"Difference (Reference-Reconstructed)\nMean difference: {diff_nm:.2f} [nm]"
            )

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
            mat_data = loadmat(f"{matlab_data_dir}/Gamma.mat")
            Gamma_matlab = mat_data["Gamma"]
            gamma_test = np.allclose(Gamma_matlab.toarray(), self.Gamma.toarray())
            results["gamma_test"] = gamma_test
            logger.info(f"\nGamma matrix test: {'PASSED' if gamma_test else 'FAILED'}")
        except Exception as e:
            logger.error(f"Error testing Gamma matrix: {e}")
            results["gamma_test"] = False

        # Test auto-correlation matrix
        try:
            mat_data = loadmat(f"{matlab_data_dir}/Cxx.mat")
            Cxx_matlab = mat_data["Cxx"]
            cxx_test = np.allclose(Cxx_matlab, self.Cxx, rtol=5e-4)
            results["cxx_test"] = cxx_test
            logger.info(f"\nAuto-correlation matrix test: {'PASSED' if cxx_test else 'FAILED'}")
        except Exception as e:
            logger.error(f"Error testing auto-correlation matrix: {e}")
            results["cxx_test"] = False

        # Test cross-correlation matrix
        try:
            mat_data = loadmat(f"{matlab_data_dir}/Cox.mat")
            Cox_matlab = mat_data["Cox"]
            cox_test = np.allclose(Cox_matlab, self.Cox, rtol=5e-4)
            results["cox_test"] = cox_test
            logger.info(f"\nCross-correlation matrix test: {'PASSED' if cox_test else 'FAILED'}")
        except Exception as e:
            logger.error(f"Error testing cross-correlation matrix: {e}")
            results["cox_test"] = False

        # Test CnZ matrix
        try:
            mat_data = loadmat(f"{matlab_data_dir}/CnZ.mat")
            CnZ_matlab = mat_data["CnZ"]
            cnz_test = np.allclose(CnZ_matlab, self.CnZ, rtol=5e-4)
            results["cnz_test"] = cnz_test
            logger.info(f"\nCnZ test: {'PASSED' if cnz_test else 'FAILED'}")
        except Exception as e:
            logger.error(f"Error testing CnZ matrix: {e}")
            results["cnz_test"] = False

        # Test invCss matrix
        try:
            mat_data = loadmat(f"{matlab_data_dir}/invCss.mat")
            invCss_matlab = mat_data["invCss"]
            invCss_test = np.allclose(invCss_matlab, self.invCss, atol=5e-3)
            results["invCss_test"] = invCss_test
            logger.info(f"\ninvCss test: {'PASSED' if invCss_test else 'FAILED'}")
        except Exception as e:
            logger.error(f"Error testing invCss matrix: {e}")
            results["invCss_test"] = False

        # Test reconstructor matrix
        try:
            mat_data = loadmat(f"{matlab_data_dir}/RecStatSAsuperRes.mat")
            RecStatSA_matlab = mat_data["RecStatSAsuperRes"]
            rec_test = np.allclose(RecStatSA_matlab, self.RecStatSA, atol=5e-3)
            results["rec_test"] = rec_test
            logger.info(f"\nReconstructor matrix test: {'PASSED' if rec_test else 'FAILED'}")
        except Exception as e:
            logger.error(f"Error testing reconstructor matrix: {e}")
            results["rec_test"] = False

        # Test with slopes generated with Matlab
        try:
            for i in range(2, 4):
                mat_data = loadmat(f"{matlab_data_dir}/slopes_{i}.mat")
                slopes = mat_data[f"slopes_{i}"]

                # Load reconstructed wavefront from Matlab
                mat_data = loadmat(f"{matlab_data_dir}/wavefront_{i}.mat")
                wavefront = mat_data[f"wavefront_{i}"]

                # Visualize the comparison
                self.visualize_reconstruction(slopes, wavefront)
                plt.show()

        except Exception as e:
            logger.error(f"Error testing with slopes: {e}")

        return results


# Example usage
if __name__ == "__main__":
    # The reference configurations ship with the package, so this resolves wherever
    # pyTomoAO is installed from. Pass a path on the command line to use your own.
    import sys

    from pyTomoAO import example_config

    config_path = sys.argv[1] if len(sys.argv) > 1 else example_config("kapa-single-channel")

    # Create the reconstructor
    reconstructor = tomographicReconstructor(config_path)

    # Build the model based reconstructor. To build the IM based reconstructor,
    # pass the IM matrix as an argument.
    # R = reconstructor.build_reconstructor(IM, use_float32=True)
    R = reconstructor.build_reconstructor(use_float32=True)
    print(f"Reconstructor matrix shape: {R.shape}")

    # This step is only required for the model based reconstructor.
    # Assemble the reconstructor and fitting for single channel case
    reconstructor.assemble_reconstructor_and_fitting(
        nChannels=1, slopesOrder="simu", scalingFactor=1.5e7
    )
    # mask central actuator
    reconstructor.mask_DM_actuators(174)
    FR = reconstructor.FR

    print(f"Reconstructor and fitting matrix shape: {FR.shape}")

    # Visualize the reconstructor
    fig = plt.figure(figsize=(10, 8))
    im = plt.imshow(FR)
    cbar = plt.colorbar(im, fraction=0.028, pad=0.02)
    plt.title("Fitting * Reconstructor (Single Channel)")
    plt.xlabel("Slopes")
    plt.ylabel("Actuators")
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
