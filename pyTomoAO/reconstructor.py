"""
This module contains the tomographicReconstructor class for computing tomographic reconstructors
for adaptive optics systems, with options for model-based and interaction matrix-based
reconstruction approaches.
"""

import logging

import numpy as np
import yaml

from pyTomoAO import backend
from pyTomoAO._plotting import pyplot
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

    # Declaring the attribute surface does three things that the previous __getattr__ /
    # __setattr__ forwarding could not: assigning an unrecognised name raises AttributeError
    # instead of silently creating one (a misspelled parameter used to be swallowed, and the
    # reconstructor then built with the old value), dir() and IDE completion show something
    # useful, and ordinary assignment no longer runs a five-way hasattr search that can
    # trigger array-computing property getters.
    # Grouped by role rather than alphabetically: the grouping is what makes this readable
    # as documentation of the object's state.
    __slots__ = (  # noqa: RUF023
        # Configuration and the parameter objects parsed from it
        "config",
        "atmParams",
        "lgsAsterismParams",
        "lgsWfsParams",
        "tomoParams",
        "dmParams",
        # Backend kernels resolved for this instance
        "_backend",
        # Reconstruction products
        "_reconstructor",
        "_gridMask",
        "_wavefront2Meter",
        "_FR",
        "method",
        # Intermediate matrices, kept for inspection
        "Gamma",
        "Cxx",
        "Cox",
        "CnZ",
        "RecStatSA",
        "IM",
        # DM fitting
        "fit",
        "modes",
    )

    #: dtypes accepted by the ``reconstructor`` setter.
    valid_constructor_type = (np.float32, np.float64)

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
        self._reconstructor = None
        self._gridMask = None
        self._wavefront2Meter = None
        self._FR = None
        self.fit = None
        self.modes = None
        self.method = None
        self.atmParams = None
        self.lgsAsterismParams = None
        self.lgsWfsParams = None
        self.tomoParams = None
        self.dmParams = None
        # Intermediate matrices, populated by build_reconstructor.
        self.Gamma = None
        self.Cxx = None
        self.Cox = None
        self.CnZ = None
        self.RecStatSA = None
        self.IM = None

        # Resolve the kernel backend for this instance. This used to flip a module-level
        # CUDA flag, which did not work: the GPU functions were already bound at import, so
        # force_cpu logged that it was forcing the CPU and then ran on the GPU anyway. It
        # also changed the backend for every other reconstructor in the process.
        self._backend = backend.get_backend("cpu" if force_cpu else "auto")
        if force_cpu:
            logger.info("\nForcing CPU usage for computations.")

        logger.info("\n-->> Initializing reconstructor object <<--")
        # Load configuration
        with open(config_file) as f:
            self.config = yaml.safe_load(f)

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
            self.atmParams = atm_params
            logger.info("\nSuccessfully initialized Atmosphere parameters.")
            logger.info(atm_params)
        except (ValueError, TypeError) as e:
            logger.error(f"Configuration Error in Atmosphere parameters: {e}")
            raise  # Re-raise to prevent continuing with invalid parameters

        try:
            lgs_asterism_params = lgsAsterismParameters(self.config, self.atmParams)
            self.lgsAsterismParams = lgs_asterism_params
            logger.info("\nSuccessfully initialized LGS asterism parameters.")
            logger.info(lgs_asterism_params)
        except (ValueError, TypeError) as e:
            logger.error(f"\nConfiguration Error in LGS asterism parameters: {e}")
            raise  # Re-raise to prevent continuing with invalid parameters

        try:
            lgs_wfs_params = lgsWfsParameters(self.config, self.lgsAsterismParams)
            self.lgsWfsParams = lgs_wfs_params
            logger.info("\nSuccessfully initialized LGS WFS parameters.")
            logger.info(lgs_wfs_params)
        except (ValueError, TypeError) as e:
            logger.error(f"\nConfiguration Error in LGS WFS parameters: {e}")
            raise  # Re-raise to prevent continuing with invalid parameters

        try:
            tomo_params = tomographyParameters(self.config)
            self.tomoParams = tomo_params
            logger.info("\nSuccessfully initialized Tomography parameters.")
            logger.info(tomo_params)
        except (ValueError, TypeError) as e:
            logger.error(f"\nConfiguration Error in Tomography parameters: {e}")
            raise  # Re-raise to prevent continuing with invalid parameters

        try:
            dm_params = dmParameters(self.config)
            self.dmParams = dm_params
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
            self._reconstructor = value
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
        if self._gridMask is None:
            # Accessing the property builds the reconstructor, which sets the grid mask.
            _ = self.reconstructor
        return self._gridMask

    # ======================================================================
    # Forwarded parameters
    #
    # These four names are reached directly on the reconstructor across the tests, docs and
    # examples, so they stay available here. Everything else lives on the parameter object
    # that owns it -- rec.atmParams.altitude, rec.lgsWfsParams.nValidSubap and so on -- which
    # is discoverable, unambiguous, and does not need a runtime search across five objects.
    @property
    def nLGS(self):
        """
        Number of laser guide stars.

        Parameters
        ----------
        None

        Returns
        -------
        int
            The guide star count, taken from the LGS asterism parameters.
        """
        return self.lgsAsterismParams.nLGS

    @nLGS.setter
    def nLGS(self, value):
        """
        Set the number of laser guide stars on every parameter object that tracks it.

        Parameters
        ----------
        value : int
            New guide star count. Must be non-negative.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``value`` is negative.
        """
        if value < 0:
            raise ValueError("nLGS must be a non-negative integer.")
        value = int(value)
        # Three parameter objects carry nLGS and they must not disagree; lgsWfsParams also
        # resizes its per-sensor rotation and offset arrays when this changes.
        for params in (self.lgsAsterismParams, self.lgsWfsParams, self.tomoParams):
            if params is not None and hasattr(params, "nLGS"):
                params.nLGS = value

    @property
    def r0(self):
        """
        Fried parameter at the observing zenith angle, in metres.

        Parameters
        ----------
        None

        Returns
        -------
        float
            Derived from ``r0_zenith`` and the zenith angle; set ``r0_zenith`` to change it.
        """
        return self.atmParams.r0

    @property
    def r0_zenith(self):
        """
        Fried parameter at zenith, in metres.

        Parameters
        ----------
        None

        Returns
        -------
        float
        """
        return self.atmParams.r0_zenith

    @r0_zenith.setter
    def r0_zenith(self, value):
        """
        Set the Fried parameter at zenith.

        Parameters
        ----------
        value : float
            Fried parameter in metres. Must be positive.

        Returns
        -------
        None
        """
        self.atmParams.r0_zenith = value

    @property
    def L0(self):
        """
        Turbulence outer scale, in metres.

        Parameters
        ----------
        None

        Returns
        -------
        float
        """
        return self.atmParams.L0

    @L0.setter
    def L0(self, value):
        """
        Set the turbulence outer scale.

        Parameters
        ----------
        value : float
            Outer scale in metres. Must be positive.

        Returns
        -------
        None
        """
        self.atmParams.L0 = value

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

        plt = pyplot()

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
        plt = pyplot()
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
