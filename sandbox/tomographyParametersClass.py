# tomographyParametersClass.py
import numpy as np
from numbers import Number

class tomographyParameters:
    """
    Encapsulates tomography optimization parameters with validation, geometry computation,
    and enhanced string representation.
    """

    def __init__(self, config: dict):
        self._config = config["tomography_parameters"]
        self._initialize_properties()

    def _initialize_properties(self):
        params = self._config
        self.nFitSrc = params["nFitSrc"]
        self.fovOptimization = params["fovOptimization"]
        self.fitSrcHeight = np.inf

    @property
    def nFitSrc(self) -> int:
        return self._nFitSrc

    @nFitSrc.setter
    def nFitSrc(self, value):
        if not isinstance(value, int):
            raise TypeError("nFitSrc must be an integer")
        if value <= 0:
            raise ValueError("nFitSrc must be positive")
        self._nFitSrc = value

    @property
    def fovOptimization(self) -> float:
        return self._fovOptimization

    @fovOptimization.setter
    def fovOptimization(self, value):
        if not isinstance(value, Number):
            raise TypeError("fovOptimization must be numeric")
        if value < 0:
            raise ValueError("fovOptimization cannot be negative")
        if self.nFitSrc > 1 and value == 0:
            raise ValueError("fovOptimization must be positive when nFitSrc > 1")
        self._fovOptimization = float(value)

    def compute_optimization_geometry(self):
        """(Previous implementation remains unchanged)"""
        arcsec_to_rad = np.pi / (180 * 3600)
        if self.nFitSrc == 1:
            return np.array([0.0]), np.array([0.0])
        x = np.linspace(-self.fovOptimization/2, self.fovOptimization/2, self.nFitSrc)
        x_grid, y_grid = np.meshgrid(x, x)
        azimuth = np.arctan2(y_grid, x_grid)
        zenith_arcsec = np.sqrt(x_grid**2 + y_grid**2)
        return (
            zenith_arcsec.flatten() * arcsec_to_rad,
            azimuth.T.flatten()
        )

    @property
    def optimization_shape(self) -> tuple:
        return (self.nFitSrc, self.nFitSrc)

    def __str__(self):
        """Human-readable string representation of the tomography parameters"""
        lines = [
            "Tomography Parameters:",
            f"Number of fitting sources (nFitSrc): {self.nFitSrc}",
            f"Altitude of fitting sources (fitSrcHeight): {self.fitSrcHeight}",
            f"Field of View (arcsec): {self.fovOptimization}",
            f"Optimization grid shape: {self.optimization_shape}",
            f"Total optimization points: {self.nFitSrc**2}"
        ]
        
        # Add geometry preview
        zenith, azimuth = self.compute_optimization_geometry()
        lines.append("\nGeometry Preview:")
        lines.append(f"Zenith angles (rad): {self._format_array_preview(zenith)}")
        lines.append(f"Azimuth angles (rad): {self._format_array_preview(azimuth)}")
        
        return "\n".join(lines)

    def _format_array_preview(self, arr: np.ndarray, max_items: int = 5) -> str:
        """Helper to format array previews"""
        if len(arr) <= max_items:
            return np.array2string(arr, precision=4, separator=', ')
        return f"[{', '.join(f'{x:.4f}' for x in arr[:max_items])}, ...] ({len(arr)} total)"
    
# Example Usage
if __name__ == "__main__":
    config = {
        "tomography_parameters": {
            "nFitSrc": 3,
            "fovOptimization": 4.0  # arcseconds
        }
    }

    try:
        tomo = tomographyParameters(config)
        zenith, azimuth = tomo.compute_optimization_geometry()
        
        print(f"Optimization points: {len(zenith)}")
        print(f"Zenith angles (rad):\n{zenith}")
        print(f"Azimuth angles (rad):\n{azimuth}")
        
    except (ValueError, TypeError) as e:
        print(f"Configuration Error: {e}")