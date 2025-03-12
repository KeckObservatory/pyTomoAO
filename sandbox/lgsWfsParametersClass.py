import numpy as np
from numbers import Number
from lgsAsterismParametersClass import lgsAsterismParameters
from atmosphereParametersClass import atmosphereParameters

class lgsWfsParameters:
    """
    Encapsulates Laser Guide Star Wavefront Sensor (LGS WFS) parameters with validation.
    
    Handles:
    - Telescope characteristics
    - Lenslet array configuration
    - Field stop parameters
    - Valid actuator/lenslet maps
    """

    def __init__(self, config: dict, lgsAsterism_params: lgsAsterismParameters):
        """
        Initialize from configuration dictionary.
        
        Args:
            config: Dictionary containing "lgs_wfs_parameters" key with subkeys:
                    D, nLenslet, nPx, fieldStopSize, nLGS, validLLMap, validActuatorMap
        """
        self._config = config["lgs_wfs_parameters"]
        self._lgsAsterism_params = lgsAsterism_params
        
        # Initialize nLGS first since other properties depend on it
        self._nLGS = self._lgsAsterism_params.nLGS
        
        # Initialize default values for arrays before they're set properly
        self._wfsLensletsRotation = np.zeros(self._nLGS, dtype=float)
        self._wfsLensletsOffset = np.zeros((2, self._nLGS), dtype=float)
        
        self._initialize_properties()

    def _initialize_properties(self):
        """Initialize all properties using their setters for validation"""
        params = self._config
        self.D = params["D"]
        self.nLenslet = params["nLenslet"]
        self.nPx = params["nPx"]
        self.fieldStopSize = params["fieldStopSize"]
        self.validLLMap_list = params["validLLMap"]
        self.validActuatorMap_list = params["validActuatorMap"]
        self.wfsLensletsRotation = params.get("wfsLensletsRotation", [0]*self._nLGS)
        self.wfsLensletsOffset = params.get("wfsLensletsOffset", np.zeros((2, self._nLGS), dtype=float))
    
    # === Core Telescope Properties ===
    @property
    def D(self) -> float:
        """Telescope diameter in meters (positive value)"""
        return self._D

    @D.setter
    def D(self, value):
        if not isinstance(value, Number):
            raise TypeError("Telescope diameter must be numeric")
        if value <= 0:
            raise ValueError("Telescope diameter must be positive")
        self._D = float(value)

    # === Lenslet Array Configuration ===
    @property
    def nLenslet(self) -> int:
        """Number of lenslets per dimension (positive integer)"""
        return self._nLenslet

    @nLenslet.setter
    def nLenslet(self, value):
        if not isinstance(value, int):
            raise TypeError("Lenslet count must be integer")
        if value <= 0:
            raise ValueError("Lenslet count must be positive")
        self._nLenslet = value

    @property
    def nPx(self) -> int:
        """Pixels per lenslet (positive integer)"""
        return self._nPx

    @nPx.setter
    def nPx(self, value):
        if not isinstance(value, int):
            raise TypeError("Pixel count must be integer")
        if value <= 0:
            raise ValueError("Pixel count must be positive")
        self._nPx = value

    # === Field Stop Configuration ===
    @property
    def fieldStopSize(self) -> float:
        """Field stop size in arcseconds (positive value)"""
        return self._fieldStopSize

    @fieldStopSize.setter
    def fieldStopSize(self, value):
        if not isinstance(value, Number):
            raise TypeError("Field stop size must be numeric")
        if value <= 0:
            raise ValueError("Field stop size must be positive")
        self._fieldStopSize = float(value)

    # === Guide Star Configuration ===
    @property
    def nLGS(self) -> int:
        """Number of laser guide stars (non-negative integer)"""
        return self._nLGS

    @nLGS.setter
    def nLGS(self, value):
        if not isinstance(value, int):
            raise TypeError("LGS count must be integer")
        if value < 0:
            raise ValueError("LGS count cannot be negative")
        
        # Only update arrays if the value is actually changing
        if value != self._nLGS:
            old_value = self._nLGS
            self._nLGS = value
            
            # Update the rotation array
            # If increasing, pad with zeros; if decreasing, truncate
            if value > old_value:
                # Create a new array with the right size
                new_rotation = np.zeros(value, dtype=float)
                # Copy existing values
                new_rotation[:old_value] = self._wfsLensletsRotation
                self._wfsLensletsRotation = new_rotation
            else:
                # Truncate to the new length
                self._wfsLensletsRotation = self._wfsLensletsRotation[:value]
            
            # Update the offset array
            if value > old_value:
                # Create a new array with the right size
                new_offset = np.zeros((2, value), dtype=float)
                # Copy existing values
                new_offset[:, :old_value] = self._wfsLensletsOffset[:, :old_value]
                self._wfsLensletsOffset = new_offset
            else:
                # Truncate to the new length
                self._wfsLensletsOffset = self._wfsLensletsOffset[:, :value]
        else:
            self._nLGS = value

    # === Validation Maps ===
    @property
    def validLLMap_list(self) -> list:
        """2D list representation of valid lenslet/lenslet map"""
        return self._validLLMap_list

    @validLLMap_list.setter
    def validLLMap_list(self, value):
        """Validate and store lenslet map"""
        try:
            arr = np.array(value, dtype=bool)
            if arr.ndim != 2:
                raise ValueError("Lenslet map must be 2D")
        except Exception as e:
            raise ValueError(f"Invalid lenslet map: {e}") from None
        self._validLLMap_list = value

    @property
    def validLLMap(self) -> np.ndarray:
        """2D boolean array of valid lenslet pairs"""
        return np.array(self.validLLMap_list, dtype=bool)

    @validLLMap.setter
    def validLLMap(self, value: np.ndarray):
        """Set the valid lenslet map from a 2D boolean array"""
        if not isinstance(value, np.ndarray):
            raise TypeError("validLLMap must be a numpy array")
        if value.ndim != 2:
            raise ValueError("validLLMap must be 2D")
        self.validLLMap_list = value.tolist()  # Convert back to list for storage

    @property
    def validActuatorMap_list(self) -> list:
        """2D list representation of valid actuators"""
        return self._validActuatorMap_list

    @validActuatorMap_list.setter
    def validActuatorMap_list(self, value):
        """Validate and store actuator map"""
        try:
            arr = np.array(value, dtype=bool)
            if arr.ndim != 2:
                raise ValueError("Actuator map must be 2D")
        except Exception as e:
            raise ValueError(f"Invalid actuator map: {e}") from None
        self._validActuatorMap_list = value

    @property
    def validActuatorMap(self) -> np.ndarray:
        """2D boolean array of valid actuators"""
        return np.array(self.validActuatorMap_list, dtype=bool)
    
    @property
    def validLLMapSupport(self) -> np.ndarray:
        """Padded valid lenslet map with super-resolution support"""
        return np.pad(self.validLLMap, pad_width=2, mode='constant', constant_values=0)

    @property
    def DSupport(self) -> float:
        """Effective diameter accounting for support padding"""
        return self.D * self.validLLMapSupport.shape[0] / self.nLenslet
    
    @property
    def wfsLensletsRotation(self) -> np.ndarray:
        """Rotation angles of WFS lenslets in radians"""
        return self._wfsLensletsRotation

    @wfsLensletsRotation.setter 
    def wfsLensletsRotation(self, value):
        if value is None:
            value = [0] * self.nLGS
        arr = np.array(value, dtype=float)
        if arr.ndim != 1:
            raise ValueError("wfsLensletsRotation must be 1D array")
        if len(arr) != self.nLGS:
            raise ValueError(f"wfsLensletsRotation length ({len(arr)}) must match nLGS ({self.nLGS})")
        self._wfsLensletsRotation = arr

    @property
    def wfsLensletsOffset(self) -> np.ndarray:
        """Offsets of WFS lenslets in subapertures"""
        return self._wfsLensletsOffset

    @wfsLensletsOffset.setter 
    def wfsLensletsOffset(self, value):
        if value is None:
            value = np.zeros((2, self.nLGS), dtype=float)
        arr = np.array(value, dtype=float)
        if arr.ndim != 2:
            raise ValueError("wfsLensletsOffset must be 2D array")
        if arr.shape[1] != self.nLGS:
            raise ValueError(f"wfsLensletsOffset length ({arr.shape[1]}) must match nLGS ({self.nLGS})")
        self._wfsLensletsOffset = arr

    def __str__(self):
        """Human-readable string representation with new properties"""
        # Existing calculations
        ll_valid = np.sum(self.validLLMap)
        ll_total = np.prod(self.validLLMap.shape)
        act_valid = np.sum(self.validActuatorMap)
        act_total = np.prod(self.validActuatorMap.shape)

        # New properties
        support_shape = self.validLLMapSupport.shape
        support_ratio = support_shape[0] / self.nLenslet

        return (
            "Laser Guide Star WFS Parameters:\n"
            f"  - Telescope Diameter: {self.D:.2f} m (Support-adjusted: {self.DSupport:.2f} m)\n"
            f"  - Lenslet Array: {self.nLenslet}x{self.nLenslet} → Support: {support_shape[0]}x{support_shape[1]}\n"
            f"  - Pixels per Lenslet: {self.nPx}\n"
            f"  - Field Stop: {self.fieldStopSize:.2f} arcsec\n"
            f"  - Number of LGS: {self.nLGS}\n"
            f"  - WFS Lenslets Rotation: \n       {np.rad2deg(self.wfsLensletsRotation)} deg\n"
            f"  - WFS Lenslets Offset: \n       {self.wfsLensletsOffset[0,:]}\n       {self.wfsLensletsOffset[1,:]} subap"
            
            "\nValidation Maps:"
            "\n  - Valid Lenslet Map:"
            f"\n    Valid Elements: {ll_valid}/{ll_total} ({ll_valid/ll_total:.1%})"
#            f"\n    Preview:\n{self._format_map_preview(self.validLLMap)}"
            "\n\n  - Padded Support Map:"
            f"\n    Scaling Factor: {support_ratio:.2f}x"
#            f"\n    Preview:\n{self._format_map_preview(self.validLLMapSupport)}"
            "\n\n  - Valid Actuator Map:"
            f"\n    Valid Elements: {act_valid}/{act_total} ({act_valid/act_total:.1%})"
#            f"\n    Preview:\n{self._format_map_preview(self.validActuatorMap)}"
        )


# Example usage
if __name__ == "__main__":
    atmConfig = {
        "atmosphere_parameters": {
            "nLayer": 3,
            "zenithAngleInDeg": 0.0,
            "altitude": [5, 10, 15],  # In kilometers
            "L0": 30.0,
            "r0": 0.15,
            "fractionnalR0": [0.5, 0.3, 0.2],
            "wavelength": 500e-9,
            "windDirection": [90, 45, 180],
            "windSpeed": [10, 20, 15]
        }
    }
    
    atmParams = atmosphereParameters(atmConfig)
    
    lgsAsterismConfig = {
        "lgs_asterism": {
            "radiusAst": 30.0,       # arcseconds
            "LGSwavelength": 589e-9, # meters (sodium wavelength)
            "baseLGSHeight": 90000,   # meters (90km nominal sodium layer height)
            "nLGS": 4
        }
    }
    
    lgsAsterismParams = lgsAsterismParameters(lgsAsterismConfig, atmParams)
    
    config = {
        "lgs_wfs_parameters": {
            "D": 8.2,
            "nLenslet": 40,
            "nPx": 16,
            "fieldStopSize": 2.5,
            "validLLMap": [
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1]
            ],
            "validActuatorMap": [
                [1, 1, 0],
                [0, 1, 1],
                [1, 0, 1]
            ],
            "wfsLensletsRotation": [0,1,0,0],
            "wfsLensletsOffset": [[0.1,-0.1,-0.1,0.1],[0.1,0.1,-0.1,-0.1]]
        }
    }
    
    try:
        lgsWfsParams = lgsWfsParameters(config, lgsAsterismParams)
        print("Successfully initialized LGS WFS parameters.")
        print(lgsWfsParams)
        
        # Test changing nLGS to demonstrate auto-adjustment of arrays
        print("\n--- Testing nLGS change ---")
        print(f"Original nLGS: {lgsWfsParams.nLGS}")
        print(f"Original rotation array shape: {lgsWfsParams.wfsLensletsRotation.shape}")
        print(f"Original offset array shape: {lgsWfsParams.wfsLensletsOffset.shape}")
        
        # Increase nLGS
        lgsWfsParams.nLGS = 6
        print(f"\nChanged nLGS to: {lgsWfsParams.nLGS}")
        print(f"New rotation array shape: {lgsWfsParams.wfsLensletsRotation.shape}")
        print(f"New rotation values: {lgsWfsParams.wfsLensletsRotation}")
        print(f"New offset array shape: {lgsWfsParams.wfsLensletsOffset.shape}")
        
        # Decrease nLGS
        lgsWfsParams.nLGS = 2
        print(f"\nChanged nLGS to: {lgsWfsParams.nLGS}")
        print(f"New rotation array shape: {lgsWfsParams.wfsLensletsRotation.shape}")
        print(f"New rotation values: {lgsWfsParams.wfsLensletsRotation}")
        print(f"New offset array shape: {lgsWfsParams.wfsLensletsOffset.shape}")
        
    except (ValueError, TypeError) as e:
        print(f"Configuration Error: {e}")