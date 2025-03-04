import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class FittingStep:
    """
    A class for performing least squares fitting of OPD maps to DM influence functions.

    Attributes
    ----------
    dmGeometry : object
        An object containing the Deformable Mirror (DM) geometry and influence functions.
    fitting_matrix : ndarray
        A 2D numpy array representing the fitting matrix.
    influence_functions : ndarray
        A 2D numpy array representing the influence functions matrix.
    
    Methods
    -------
    __init__(dmGeometry)
        Initializes the fitting process with the provided DM geometry object.
    
    fit(opd_map)
        Multiplies the OPD map by the fitting matrix to obtain the command vector.
    
    __getattr__(name)
        Forwards attribute access to the dmGeometry class if it contains the requested attribute.
    
    __setattr__(name, value)
        Forwards attribute setting to the dmGeometry class if it contains the specified attribute.
    """
    
    def __init__(self, dmGeometry):
        logger.debug("Initializing TomographyFitting with DM geometry.")
        self.dmGeometry = dmGeometry
        self._fitting_matrix = np.array([])  # Initialize as an empty 2D numpy array
        self._influence_functions = np.array([])  # Initialize as an empty 2D numpy array

    @property
    def F(self):
        logger.debug("Accessing the fitting matrix property.")
        return self._fitting_matrix

    @F.setter
    def F(self, value):
        logger.debug("Setting the fitting matrix property.")
        if isinstance(value, np.ndarray) and value.ndim == 2:
            self._fitting_matrix = value
        else:
            logger.error("Invalid fitting matrix value. Must be a 2D numpy array.")
            raise ValueError("Fitting matrix must be a 2D numpy array.")

    @property
    def fitting_matrix(self):
        logger.debug("Accessing the full name fitting_matrix property.")
        return self._fitting_matrix

    @fitting_matrix.setter
    def fitting_matrix(self, value):
        self.F = value

    @property
    def IF(self):
        logger.debug("Accessing the influence functions property.")
        return self._influence_functions

    @IF.setter
    def IF(self, value):
        logger.debug("Setting the influence functions property.")
        if isinstance(value, np.ndarray) and value.ndim == 2:
            self._influence_functions = value
        else:
            logger.error("Invalid influence functions value. Must be a 2D numpy array.")
            raise ValueError("Influence functions must be a 2D numpy array.")

    @property
    def influence_functions(self):
        logger.debug("Accessing the full name influence_functions property.")
        return self._influence_functions

    @influence_functions.setter
    def influence_functions(self, value):
        self.IF = value

    def fit(self, opd_map):
        """
        Multiplies the OPD map by the fitting matrix to obtain the command vector.

        :param opd_map: The Optical Path Difference (OPD) map to be fitted.
        :return: The command vector to send to the DM.
        """
        logger.info("Performing fitting of the OPD map.")
        if self.F.size == 0:
            logger.error("Fitting matrix is not set.")
            raise ValueError("Fitting matrix is not set.")
        
        command_vector = np.dot(self.F, opd_map.flatten())
        logger.debug("Fitting completed. Command vector shape: %s", command_vector.shape)
        return command_vector

    def __getattr__(self, name):
        """
        Forwards attribute access to the dmGeometry class if it contains the requested attribute.
        """
        logger.debug("Getting attribute '%s' from dmGeometry.", name)
        if hasattr(self.dmGeometry, name):
            return getattr(self.dmGeometry, name)
        logger.error("Attribute '%s' not found in dmGeometry.", name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """
        Forwards attribute setting to the dmGeometry class if it contains the specified attribute.
        """
        logger.debug("Setting attribute '%s'.", name)
        if name in ['dmGeometry', '_fitting_matrix', '_influence_functions']:
            super().__setattr__(name, value)
        else:
            if hasattr(self.dmGeometry, name):
                setattr(self.dmGeometry, name, value)
                return
            super().__setattr__(name, value)

if __name__ == "__main__":
    # Example of how the class would be instantiated and used
    # Assuming dmGeometry is a predefined object with the necessary attributes
    # This object should be created or imported from relevant modules

    # Example instantiation (replace with actual parameter object)
    # dmGeometry = ...

    # Create an instance of the TomographyFitting
    logger.info("Creating an instance of TomographyFitting.")
    fitting = FittingStep(dmGeometry)
    
    # Example setting of the fitting matrix
    # fitting.F = np.random.rand(10, 10)  # Replace with actual fitting matrix

    # Example setting of the influence functions
    # fitting.IF = np.random.rand(10, 10)  # Replace with actual influence functions

    # Example OPD map
    # opd_map = np.random.rand(10)  # Replace with actual OPD map

    # Perform fitting
    # command_vector = fitting.fit(opd_map)
    # logger.info("Command vector: %s", command_vector)
    # print(command_vector)