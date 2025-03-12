# lgsAsterismParametersClass.py
from numbers import Number
from atmosphereParametersClass import atmosphereParameters
import math
import numpy as np

class lgsAsterismParameters:
    """
    Encapsulates Laser Guide Star (LGS) Asterism parameters with validation
    and enhanced string representation.
    """

    def __init__(self, config: dict, atm_params: atmosphereParameters):
        self._config = config["lgs_asterism"]
        self._atm_params = atm_params
        self._initialize_properties()

    def _initialize_properties(self):
        params = self._config
        self.radiusAst = params["radiusAst"]
        self.LGSwavelength = params["LGSwavelength"]
        self.baseLGSHeight = params["baseLGSHeight"]
        self.nLGS = params["nLGS"]

    @property
    def radiusAst(self) -> float:
        return self._radiusAst

    @radiusAst.setter
    def radiusAst(self, value):
        # Validation logic remains the same
        if not isinstance(value, Number):
            raise TypeError("Radius must be numeric")
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radiusAst = float(value)

    @property
    def LGSwavelength(self) -> float:
        return self._LGSwavelength

    @LGSwavelength.setter
    def LGSwavelength(self, value):
        # Validation logic remains the same
        if not isinstance(value, Number):
            raise TypeError("Wavelength must be numeric")
        if value <= 0:
            raise ValueError("Wavelength must be positive")
        self._LGSwavelength = float(value)

    @property
    def baseLGSHeight(self) -> float:
        return self._baseLGSHeight

    @baseLGSHeight.setter
    def baseLGSHeight(self, value):
        # Validation logic remains the same
        if not isinstance(value, Number):
            raise TypeError("Base height must be numeric")
        if value <= 0:
            raise ValueError("Base height must be positive")
        self._baseLGSHeight = float(value)
        
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

    @property
    def LGSheight(self) -> float:
        return self.baseLGSHeight * self._atm_params.airmass

    @property
    def LGSdirections(self) -> np.ndarray:
        """
        Compute LGS asterism directions in polar coordinates (radians).
        
        Returns:
            np.ndarray: Array of shape (4, 2) containing [radius, azimuth] pairs
                        for four LGS positions in a square pattern:
                        - Radius: sqrt(2)*radiusAst (converted to radians)
                        - Azimuth: 45° spaced angles starting from 225°
        """
        arcsec_to_rad = math.pi / (180 * 3600)  # 1 arcsec in radians
        
        # Calculate base radius in radians
        base_radius = self.radiusAst * arcsec_to_rad
        
        # Define azimuth angles in degrees (225°, 135°, 315°, 45°)
        azimuth_deg =np.zeros(self.nLGS)
        for i in range(self.nLGS):
            azimuth_deg[i] = i*360/self.nLGS
        
        return np.array([
            [base_radius, math.radians(angle)]
            for angle in azimuth_deg
        ])
        
    @property
    def directionVectorLGS(self) -> np.ndarray:
        """
        Compute 3D direction vectors for LGS in observer's coordinate system.
        
        Returns:
            np.ndarray: 3xN array where N = nLGS, with vectors:
                        [ [x1, x2, ...],
                        [y1, y2, ...],
                        [z1, z2, ...] ]
                        where z = 1 (optical axis) and x/y are transverse components
        """
        n_lgs = self.nLGS
        vectors = np.zeros((3, n_lgs))
        
        for i in range(n_lgs):
            zenith = self.LGSdirections[i, 0]
            azimuth = self.LGSdirections[i, 1]
            
            # Compute transverse components
            tan_zenith = math.tan(zenith)
            vectors[0, i] = tan_zenith * math.cos(azimuth)
            vectors[1, i] = tan_zenith * math.sin(azimuth)
            
            # Optical axis component normalized to 1
            vectors[2, i] = 1.0

        return vectors
        
    def __str__(self):
        """Human-readable string representation with full geometry details"""
        arcsec_to_deg = 1/3600  # Conversion from arcseconds to degrees
        rad_to_arcsec = 180 * 3600 / math.pi  # Radians to arcseconds conversion

        # Core parameters
        base_str = (
            "LGS Asterism Parameters:\n"
            f"  - Number of LGS: {self.nLGS}\n"
            f"  - Base Radius: {self.radiusAst:.2f} arcsec\n"
            f"  - Wavelength: {self._format_wavelength()}\n"
            f"  - Base Height: {self.baseLGSHeight/1000:.1f} km\n"
            f"  - Current Airmass: {self._atm_params.airmass:.2f}\n"
            f"  - Effective Height: {self.LGSheight/1000:.1f} km\n"
            "\nAsterism Geometry:"
        )

        # Direction coordinates section
        directions_str = ["  Polar Coordinates (per LGS):"]
        for i in range(self.nLGS):
            radius_arcsec = self.LGSdirections[i, 0] * rad_to_arcsec
            azimuth_deg = math.degrees(self.LGSdirections[i, 1]) % 360
            directions_str.append(
                f"    LGS {i+1}: {radius_arcsec:.2f} arcsec "
                f"@ {azimuth_deg:.1f}°"
            )

        # Direction vectors section
        vectors_str = [
            "\n  Direction Vectors (x,y,z normalization):",
            np.array2string(
                self.directionVectorLGS,
                prefix='    ',
                formatter={
                    'float_kind': lambda x: f"{x:.2e}" if abs(x) < 1e-3 else f"{x:.4f}"
                }
            ).replace('[', '    [')
        ]

        return "\n".join([base_str] + directions_str + vectors_str)

    def _format_wavelength(self) -> str:
        """Format wavelength with unit conversion"""
        nm = self.LGSwavelength * 1e9
        return f"{nm:.1f} nm ({self.LGSwavelength:.2e} m)"

    @property
    def atmospheric_parameters(self) -> atmosphereParameters:
        return self._atm_params

# Example Usage
if __name__ == "__main__":
    # Example config and atmosphere
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
    
    config = {
        "lgs_asterism": {
            "radiusAst": 30.0,       # arcseconds
            "LGSwavelength": 589e-9, # meters (sodium wavelength)
            "baseLGSHeight": 90000,   # meters (90km nominal sodium layer height)
            "nLGS": 4
        }
    }
    
    try:
        lgsAsterismParams = lgsAsterismParameters(config, atmParams)
        print("Successfully initialized LGS asterism parameters.")
        print(lgsAsterismParams)
    except (ValueError, TypeError) as e:
        print(f"Configuration Error: {e}")