
class TomographyFitting:
    def __init__(self, dm):
        """
        Initialize the TomographyFitting class with a Deformable Mirror (DM) object.

        :param dm: A Deformable Mirror object containing actuator layout and influence functions.
        """
        self.dm = dm

    def fit(self, opd_map):
        """
        Perform least squares fitting of the OPD map to the DM influence functions.

        :param opd_map: The Optical Path Difference (OPD) map to be fitted.
        :return: The command vector to send to the DM.
        """
        # Extract influence functions from the DM object
        influence_functions = self.dm.get_influence_functions()

        # Perform least squares fitting
        # This is a placeholder for the actual least squares computation
        # You would typically use a library like numpy or scipy for this
        command_vector = self.least_squares_projection(opd_map, influence_functions)

        return command_vector

    def least_squares_projection(self, opd_map, influence_functions):
        """
        Perform least squares projection to fit the OPD map to the influence functions.

        :param opd_map: The OPD map to be fitted.
        :param influence_functions: The influence functions from the DM.
        :return: The command vector to send to the DM.
        """
        import numpy as np

        # Reshape the OPD map and influence functions to 2D arrays if necessary
        opd_map_flat = opd_map.flatten()
        influence_functions_flat = np.array([f.flatten() for f in influence_functions])

        # Perform least squares fitting using numpy.linalg.lstsq
        command_vector, residuals, rank, s = np.linalg.lstsq(influence_functions_flat.T, opd_map_flat, rcond=None)

        return command_vector