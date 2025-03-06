# lgsWfsParametersClass.py
import numpy as np
from numbers import Number
from lgsAsterismParametersClass import lgsAsterismParameters

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
        self._initialize_properties()

    def _initialize_properties(self):
        """Initialize all properties using their setters for validation"""
        params = self._config
        self.D = params["D"]
        self.nLenslet = params["nLenslet"]
        self.nPx = params["nPx"]
        self.fieldStopSize = params["fieldStopSize"]
        #self.nLGS = self.lgsAsterism_params.nLGS
        self.validLLMap_list = params["validLLMap"]
        self.validActuatorMap_list = params["validActuatorMap"]
        self.wfs_lenslets_rotation = params.get("wfs_lenslets_rotation", [0]*self._lgsAsterism_params.nLGS)

    @property
    def wfs_lenslets_rotation(self) -> np.ndarray:
        """Rotation angles of WFS lenslets in radians"""
        return self._wfs_lenslets_rotation

    @wfs_lenslets_rotation.setter 
    def wfs_lenslets_rotation(self, value):
        if value is None:
            value = [0] * self._lgsAsterism_params.nLGS
        arr = np.array(value, dtype=float)
        if arr.ndim != 1:
            raise ValueError("wfs_lenslets_rotation must be 1D array")
        if len(arr) != self._lgsAsterism_params.nLGS:
            raise ValueError(f"wfs_lenslets_rotation length ({len(arr)}) must match nLGS ({self._lgsAsterism_params.nLGS})")
        self._wfs_lenslets_rotation = arr

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
            f"  - Number of LGS: {self._lgsAsterism_params.nLGS}\n"
            f"  - WFS Lenslets Rotation: {np.rad2deg(self.wfs_lenslets_rotation)} deg"

#            "\nValidation Maps:"
#            "\n  - Valid Lenslet Map:"
#            f"\n    Valid Elements: {ll_valid}/{ll_total} ({ll_valid/ll_total:.1%})"
#            f"\n    Preview:\n{self._format_map_preview(self.validLLMap)}"
#            "\n\n  - Padded Support Map:"
#            f"\n    Scaling Factor: {support_ratio:.2f}x"
#            f"\n    Preview:\n{self._format_map_preview(self.validLLMapSupport)}"
#            "\n\n  - Valid Actuator Map:"
#            f"\n    Valid Elements: {act_valid}/{act_total} ({act_valid/act_total:.1%})"
#            f"\n    Preview:\n{self._format_map_preview(self.validActuatorMap)}"
        )


#    def _format_map_preview(self, arr: np.ndarray, size: int = 5) -> str:
#        """Format a preview of a boolean 2D array"""
#        # Handle non-square arrays
#        rows = min(size, arr.shape[0])
#        cols = min(size, arr.shape[1])
#        
#        preview = arr[:rows, :cols]
#        preview_str = np.array2string(
#            preview,
#            prefix='    ',
#            formatter={'bool': lambda x: '█' if x else '░'}
#        ).replace('[','').replace(']','')
#        
#        shape_note = f"\n    Full Array Shape: {arr.shape}" if (rows < arr.shape[0] or cols < arr.shape[1]) else ""
#        
#        return f"    {preview_str}{shape_note}"

if __name__ == "__main__":
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
            ]
        }
    }

    try:
        lgsWfsParams = lgsWfsParameters(config)
        print("Successfully initialized LGS WFS parameters.")
        print(f"- validLLMap: {lgsWfsParams.validLLMap}")
    except (ValueError, TypeError) as e:
        print(f"Configuration Error: {e}")
        
