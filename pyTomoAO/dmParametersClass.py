import numpy as np
from numbers import Number

class dmParameters:
    """
    Encapsulates Deformable Mirror (DM) parameters with validation and type conversion.
    
    Handles:
    - Mirror heights and actuator spacing
    - Cross-coupling coefficients
    - Actuator counts and validity maps
    """

    def __init__(self, config: dict):
        """
        Initialize from configuration dictionary.
        
        Args:
            config: Dictionary containing "dm_parameters" key with subkeys:
                    dmHeights, dmPitch, dmCrossCoupling, nActuators, validActuators
        """
        self._config = config["dm_parameters"]
        self._initialize_properties()

    def _initialize_properties(self):
        """Initialize all properties using their setters for validation"""
        params = self._config
        self.dmHeights = params["dmHeights"]
        self.dmPitch = params["dmPitch"]
        self.dmCrossCoupling = params["dmCrossCoupling"]
        self.nActuators = params["nActuators"]
        self.validActuators_list = params["validActuators"]

    # === Core Mirror Properties ===
    @property
    def dmHeights(self) -> np.ndarray:
        """Mirror heights in meters (1D array, non-negative)"""
        return self._dmHeights

    @dmHeights.setter
    def dmHeights(self, value):
        value = self._convert_to_array(value, "dmHeights", float)
        if (value < 0).any():
            raise ValueError("All DM heights must be non-negative")
        self._dmHeights = value

    @property
    def dmPitch(self) -> np.ndarray:
        """Actuator pitch spacing in meters (1D array, positive)"""
        return self._dmPitch

    @dmPitch.setter
    def dmPitch(self, value):
        value = self._convert_to_array(value, "dmPitch", float)
        if (value <= 0).any():
            raise ValueError("All pitch values must be positive")
        self._dmPitch = value

    # === Control Parameters ===
    @property
    def dmCrossCoupling(self) -> float:
        """Actuator coupling coefficient (0-1)"""
        return self._dmCrossCoupling

    @dmCrossCoupling.setter
    def dmCrossCoupling(self, value):
        if not isinstance(value, Number):
            raise TypeError("Cross-coupling must be numeric")
        if not 0 <= value <= 1:
            raise ValueError("Cross-coupling must be between 0 and 1")
        self._dmCrossCoupling = float(value)

    # === Actuator Configuration ===
    @property
    def nActuators(self) -> np.ndarray:
        """Number of actuators per dimension (1D array, non-negative integers)"""
        return self._nActuators

    @nActuators.setter
    def nActuators(self, value):
        value = self._convert_to_array(value, "nActuators", int)
        if (value < 0).any():
            raise ValueError("Actuator counts cannot be negative")
        self._nActuators = value
        
    @property
    def nActuatorsSupport(self) -> np.ndarray:
        """Number of actuators per dimension plus 2 (1D array, non-negative integers)"""
        return self._nActuators + 4

    # === Actuator Map Handling ===
    @property
    def validActuators_list(self) -> list:
        """2D list representation of valid actuators"""
        return self._validActuators_list

    @validActuators_list.setter
    def validActuators_list(self, value):
        """Validate and store actuator map"""
        try:
            arr = np.array(value, dtype=bool)
            if arr.ndim != 2:
                raise ValueError("Actuator map must be 2D")
        except Exception as e:
            raise ValueError(f"Invalid actuator map: {e}") from None
            
        self._validActuators_list = value

    @property
    def validActuators(self) -> np.ndarray:
        """2D boolean array of active actuators"""
        return np.array(self.validActuators_list, dtype=bool)
    
    @property
    def validActuatorsSupport(self) -> np.ndarray:
        """
        2D boolean array of active actuators with zero padding of 2 on all sides.
        Creates a larger array with zeros around the original validActuators.
        """
        valid_act = self.validActuators
        # Create a new array with 2 extra elements in each dimension (padding)
        shape = (valid_act.shape[0] + 4, valid_act.shape[1] + 4)
        padded = np.zeros(shape, dtype=bool)
        
        # Place the original array in the center of the padded array
        padded[2:-2, 2:-2] = valid_act
        
        return padded

    # === Helper Methods ===
    def _convert_to_array(self, value, name: str, dtype):
        """Safe conversion to numpy array with validation"""
        if isinstance(value, list):
            value = np.array(value, dtype=dtype)
        elif isinstance(value, np.ndarray):
            if not np.issubdtype(value.dtype, np.dtype(dtype).type):
                value = value.astype(dtype)
        else:
            raise TypeError(f"{name} must be list/array, got {type(value)}")
        
        if value.size == 0:
            raise ValueError(f"{name} cannot be empty")
        return value

    def __str__(self):
        """Human-readable string representation of DM parameters"""
        valid_actuators = np.sum(self.validActuators)
        total_actuators = np.prod(self.validActuators.shape)
        
        valid_support = np.sum(self.validActuatorsSupport)
        total_support = np.prod(self.validActuatorsSupport.shape)
        
        return (
            "Deformable Mirror Parameters:\n"
            f"  - Actuator Grid: {self.nActuators} (Total: {np.prod(self.nActuators)} actuators)\n"
            f"  - Support Grid: {self.nActuatorsSupport} (Total: {np.prod(self.nActuatorsSupport)} actuators)\n"
            f"  - Actuator Pitch: {self._format_array_stats(self.dmPitch, unit='m')}\n"
            f"  - Mirror Heights: {self._format_array_stats(self.dmHeights, unit='m')}\n"
            f"  - Cross-Coupling: {self.dmCrossCoupling*100:.1f}%\n"
            f"  - Valid Actuators: {valid_actuators}/{total_actuators} "
            f"({valid_actuators/total_actuators:.1%})\n"
            f"  - Valid Support Actuators: {valid_support}/{total_support} "
            f"({valid_support/total_support:.1%})\n"
            #f"  - Actuator Map Preview:\n{self._format_actuator_preview()}"
        )

    def _format_array_stats(self, arr: np.ndarray, unit: str = "") -> str:
        """Helper to format array statistics"""
        if np.allclose(arr, arr[0]):
            return f"{arr[0]:.3f} {unit} (uniform)"
        return f"{np.min(arr):.3f}-{np.max(arr):.3f} {unit} (mean: {np.mean(arr):.3f})"

# Example Usage
if __name__ == "__main__":
    config = {
        "dm_parameters": {
            "dmHeights": [0.0, 0.001, 0.002],
            "dmPitch": [0.5, 0.5, 0.5],
            "dmCrossCoupling": 0.15,
            "nActuators": [20, 20, 20],
            "validActuators": [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1]
            ]
        }
    }

    try:
        dmParams = dmParameters(config)
        print("Successfully initialized DM parameters.")
        print(dmParams)
        print(f"Original nActuators: {dmParams.nActuators}")
        print(f"Support Grid (nActuatorsSupport): {dmParams.nActuatorsSupport}")
        
        print("\nOriginal validActuators:")
        print(dmParams.validActuators)
        print("\nPadded validActuatorsSupport:")
        print(dmParams.validActuatorsSupport)
    except (ValueError, TypeError) as e:
        print(f"Configuration Error: {e}")