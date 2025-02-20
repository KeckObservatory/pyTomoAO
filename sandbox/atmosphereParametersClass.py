# atmosphereParametersClass.py
import numpy as np
import math
from numbers import Number

class atmosphereParameters:
    """
    Encapsulates atmospheric parameters for optical turbulence modeling.
    
    Handles:
    - Layer altitudes and turbulence strengths
    - Wind profiles
    - Zenith angle effects
    - Derived parameters (airmass, r0, etc.)
    """

    def __init__(self, config: dict):
        """
        Initialize from configuration dictionary.
        
        Args:
            config: Dictionary containing "atmosphere_parameters" key with subkeys:
                    nLayer, zenithAngleInDeg, altitude, L0, r0, fractionnalR0, 
                    wavelength, windDirection, windSpeed
        """
        self._config = config["atmosphere_parameters"]
        self._initialize_properties()

    def _initialize_properties(self):
        """Initialize all properties using their setters for validation"""
        params = self._config
        
        # Load fundamental parameters first
        self.nLayer = params["nLayer"]
        
        # Load array parameters (order matters for length validation)
        self.altitude_km = params["altitude"]
        self.windDirection_deg = params["windDirection"]
        self.windSpeed = params["windSpeed"]
        self.fractionnalR0 = params["fractionnalR0"]
        
        # Load remaining parameters
        self.zenithAngleInDeg = params["zenithAngleInDeg"]
        self.L0 = params["L0"]
        self.r0_zenith = params["r0"]
        self.wavelength = params["wavelength"]

    # === Core Properties ===
    @property
    def nLayer(self) -> int:
        """Number of atmospheric layers (positive integer)"""
        return self._nLayer

    @nLayer.setter
    def nLayer(self, value):
        if not isinstance(value, int):
            raise TypeError("nLayer must be an integer")
        if value <= 0:
            raise ValueError("nLayer must be positive")
        self._nLayer = value

    # === Zenith Angle Effects ===
    @property
    def zenithAngleInDeg(self) -> float:
        """Observation zenith angle in degrees (0-90)"""
        return self._zenithAngleInDeg

    @zenithAngleInDeg.setter
    def zenithAngleInDeg(self, value):
        if not isinstance(value, Number):
            raise TypeError("Zenith angle must be numeric")
        if not 0 <= value <= 90:
            raise ValueError("Zenith angle must be between 0° and 90°")
        self._zenithAngleInDeg = float(value)

    @property
    def airmass(self) -> float:
        """Airmass (1/cos(zenith angle))"""
        return 1.0 / math.cos(math.radians(self.zenithAngleInDeg))

    # === Altitude Handling ===
    @property
    def altitude_km(self) -> np.ndarray:
        """Layer altitudes in kilometers (nLayer-length array)"""
        return self._altitude_km

    @altitude_km.setter
    def altitude_km(self, value):
        arr = self._validate_array(value, "altitude", min_value=0)
        self._altitude_km = arr
        self._base_altitude_m = arr * 1e3  # Store in meters

    @property
    def altitude(self) -> np.ndarray:
        """Scaled layer altitudes in meters (base altitude × airmass)"""
        return self._base_altitude_m * self.airmass

    # === Turbulence Parameters ===
    @property
    def L0(self) -> float:
        """Outer scale in meters (positive value)"""
        return self._L0

    @L0.setter
    def L0(self, value):
        if not isinstance(value, Number):
            raise TypeError("L0 must be numeric")
        if value <= 0:
            raise ValueError("L0 must be positive")
        self._L0 = float(value)

    @property
    def r0_zenith(self) -> float:
        """Fried parameter at zenith in meters (positive value)"""
        return self._r0_zenith

    @r0_zenith.setter
    def r0_zenith(self, value):
        if not isinstance(value, Number):
            raise TypeError("r0_zenith must be numeric")
        if value <= 0:
            raise ValueError("r0_zenith must be positive")
        self._r0_zenith = float(value)

    @property
    def r0(self) -> float:
        """Observed Fried parameter (scaled by zenith angle)"""
        cos_z = math.cos(math.radians(self.zenithAngleInDeg))
        return self.r0_zenith * cos_z**(3/5)

    @property
    def fractionnalR0(self) -> np.ndarray:
        """Relative turbulence strength per layer (sums to 1)"""
        return self._fractionnalR0

    @fractionnalR0.setter
    def fractionnalR0(self, value):
        arr = self._validate_array(value, "fractionnalR0")
        if (arr < 0).any():
            raise ValueError("All fractional R0 values must be non-negative")
        if not math.isclose(np.sum(arr), 1.0, rel_tol=1e-6):
            raise ValueError("Fractional R0 values must sum to 1")
        self._fractionnalR0 = arr

    # === Wind Parameters ===
    @property
    def windDirection_deg(self) -> np.ndarray:
        """Wind directions in degrees (nLayer-length array)"""
        return self._windDirection_deg

    @windDirection_deg.setter
    def windDirection_deg(self, value):
        arr = self._validate_array(value, "windDirection")
        self._windDirection_deg = arr
        self._windDirection_rad = np.deg2rad(arr)

    @property
    def windDirection(self) -> np.ndarray:
        """Wind directions in radians"""
        return self._windDirection_rad

    @property
    def windSpeed(self) -> np.ndarray:
        """Wind speeds in m/s (nLayer-length array, positive)"""
        return self._windSpeed

    @windSpeed.setter
    def windSpeed(self, value):
        arr = self._validate_array(value, "windSpeed", min_value=0)
        self._windSpeed = arr
        
    # === Wind Velocity Components ===
    @property
    def windVx(self) -> np.ndarray:
        """Eastward wind component (m/s)"""
        return self.windSpeed * np.cos(self.windDirection)

    @property
    def windVy(self) -> np.ndarray:
        """Northward wind component (m/s)"""
        return self.windSpeed * np.sin(self.windDirection)

    # === Wavelength ===
    @property
    def wavelength(self) -> float:
        """Observation wavelength in meters (positive value)"""
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        if not isinstance(value, Number):
            raise TypeError("Wavelength must be numeric")
        if value <= 0:
            raise ValueError("Wavelength must be positive")
        self._wavelength = float(value)

    # === Validation Helpers ===
    def _validate_array(self, value, name: str, min_value=None):
        """Validate and convert array parameters"""
        arr = np.array(value, dtype=float)
        
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1D array")
            
        if len(arr) != self.nLayer:
            raise ValueError(f"{name} array length ({len(arr)}) "
                             f"must match nLayer ({self.nLayer})")
                             
        if min_value is not None and (arr < min_value).any():
            raise ValueError(f"All {name} values must be ≥ {min_value}")
            
        return arr

    def __str__(self):
        """Updated string representation with wind components"""
        # Existing atmospheric parameters string
        base_str = (
            "Atmospheric Parameters:\n"
            f"  - Zenith Angle: {self.zenithAngleInDeg:.1f}° (Airmass: {self.airmass:.2f})\n"
            f"  - Outer Scale (L0): {self.L0:.1f} m\n"
            f"  - Fried Parameter (r0): {self.r0_zenith:.3f} m @ zenith → {self.r0:.3f} m @ current airmass\n"
            f"  - Observation Wavelength: {self.wavelength*1e9:.1f} nm\n"
            "\nTurbulence Profile:"
        )
        
        # Add wind components to layer information
        layer_info = []
        for i in range(self.nLayer):
            layer_info.append(
                f"    Layer {i+1}: {self.altitude[i]:.1f} m "
                f"(frac: {self.fractionnalR0[i]:.2f}, "
                f"wind: {self.windSpeed[i]:.1f} m/s @ {self.windDirection_deg[i]:.0f}° "
                f"[Vx: {self.windVx[i]:.1f}, Vy: {self.windVy[i]:.1f}]"
            )
        
        return "\n".join([base_str] + layer_info + [f"    Total fractional R0: {np.sum(self.fractionnalR0):.4f}"])

# Example Usage
if __name__ == "__main__":
    config = {
        "atmosphere_parameters": {
            "nLayer": 3,
            "zenithAngleInDeg": 45.0,
            "altitude": [5, 10, 15],  # In kilometers
            "L0": 30.0,
            "r0": 0.15,
            "fractionnalR0": [0.5, 0.3, 0.2],
            "wavelength": 500e-9,
            "windDirection": [90, 45, 180],
            "windSpeed": [10, 20, 15]
        }
    }

    try:
        atm = atmosphereParameters(config)
        print("Successfully initialized atmosphere parameters:")
        print(f"- Scaled altitudes: {atm.altitude} m")
        print(f"- Airmass: {atm.airmass:.2f}")
        print(f"- Effective r0: {atm.r0:.3f} m")
        print(f"- Wind directions (rad): {atm.windDirection}")
        
    except (ValueError, TypeError) as e:
        print(f"Configuration Error: {e}")
        
