#%%
import numpy as np
import matplotlib.pyplot as plt

class LayeredAtmosphere:
    """
    A container for specifying the altitudes and Cn^2 distribution of multiple turbulent layers.
    Provides utility methods for optical turbulence calculations and standard atmosphere profiles.
    """

    def __init__(self, altitudes, cn2_values, name=None):
        """
        Parameters
        ----------
        altitudes : array_like, shape (L,)
            Altitudes of each turbulent layer [m or km].
        cn2_values : array_like, shape (L,)
            Cn^2 values for each layer (absolute or relative).
        name : str, optional
            An optional profile name or identifier.
        """
        self.altitudes = np.array(altitudes, dtype=float)
        self.cn2_values = np.array(cn2_values, dtype=float)
        
        # Normalize the turbulence weights by default
        self.weights = self.cn2_values / np.sum(self.cn2_values)
        
        self.n_layers = len(self.altitudes)
        self.name = name if name is not None else "CustomProfile"

    def __repr__(self):
        return (f"LayeredAtmosphere(name={self.name!r}, "
                f"n_layers={self.n_layers}, "
                f"altitudes={self.altitudes}, "
                f"cn2_values={self.cn2_values})")

    # ---------------------------------------------------------------------
    # Basic inspection and property methods
    # ---------------------------------------------------------------------

    def total_cn2(self):
        """
        Returns
        -------
        float
            The total (integrated) Cn^2 across all layers.
        """
        return np.sum(self.cn2_values)

    def layer_fraction(self, idx):
        """
        Get the fraction of total turbulence in a given layer by index.
        
        Parameters
        ----------
        idx : int
            Layer index.
        
        Returns
        -------
        float
            Fraction of the total Cn^2 in the specified layer.
        """
        return self.cn2_values[idx] / self.total_cn2()

    def cumulative_profile(self):
        """
        Returns
        -------
        altitudes : np.ndarray
            Altitudes of each layer (sorted in ascending order).
        cum_weights : np.ndarray
            Cumulative sum of turbulence weights, normalized to 1.
        """
        sorted_indices = np.argsort(self.altitudes)
        sorted_altitudes = self.altitudes[sorted_indices]
        sorted_weights = self.weights[sorted_indices]
        cum_weights = np.cumsum(sorted_weights)
        return sorted_altitudes, cum_weights

    def mean_altitude(self):
        """
        Computes the mean altitude weighted by the Cn^2 distribution.

        Returns
        -------
        float
            Weighted mean altitude.
        """
        return np.sum(self.altitudes * self.weights)

    def std_altitude(self):
        """
        Computes the standard deviation of altitude (weighted by Cn^2).

        Returns
        -------
        float
            Weighted standard deviation of the altitude.
        """
        mean_alt = self.mean_altitude()
        return np.sqrt(np.sum(self.weights * (self.altitudes - mean_alt)**2))

    # ---------------------------------------------------------------------
    # Optical turbulence calculations
    # ---------------------------------------------------------------------

    def fried_parameter(self, wavelength):
        """
        Compute the Fried parameter r0 for the given wavelength based on
        the integrated Cn^2 profile.
        
        Uses the approximate relation:
        
          r0(λ) = [ 0.423 * (2π / λ)^2 * ∫ Cn^2(z) dz ]^(-3/5)
        
        where the integral is taken over the path. In a discrete model,
        sum(cn2_values) is used as a proxy for the integral.
        
        Parameters
        ----------
        wavelength : float
            Wavelength [meters].
        
        Returns
        -------
        float
            Fried parameter r0 [meters].
        """
        cn2_integral = np.sum(self.cn2_values)
        k = 2.0 * np.pi / wavelength  # wave number
        r0 = (0.423 * k**2 * cn2_integral)**(-3.0/5.0)
        return r0

    def seeing_fwhm(self, wavelength):
        """
        Compute approximate seeing FWHM (Full Width at Half Maximum)
        in arcseconds given the Fried parameter r0 at this wavelength.
        
        Seeing ≈ 0.98 * λ / r0  (in radians).
        Convert to arcseconds via 1 rad = 206265 arcseconds.
        
        Parameters
        ----------
        wavelength : float
            Wavelength [meters].
        
        Returns
        -------
        float
            Approximate seeing (FWHM) in arcseconds.
        """
        r0 = self.fried_parameter(wavelength)
        seeing_rad = 0.98 * wavelength / r0
        seeing_arcsec = seeing_rad * 206265
        return seeing_arcsec

    # ---------------------------------------------------------------------
    # Profile manipulation
    # ---------------------------------------------------------------------

    def normalize_profile(self):
        """
        Normalize the cn2_values so they sum to 1. This modifies 
        the profile in-place such that the total Cn^2 becomes 1.0.
        """
        total = np.sum(self.cn2_values)
        if total == 0:
            raise ValueError("Total Cn^2 is zero; cannot normalize.")
        self.cn2_values /= total
        self.weights = self.cn2_values / np.sum(self.cn2_values)

    def scale_profile(self, factor):
        """
        Multiply all cn2_values by a given factor (in-place).
        
        Parameters
        ----------
        factor : float
            Scaling factor for the cn2_values.
        """
        self.cn2_values *= factor
        self.weights = self.cn2_values / np.sum(self.cn2_values)

    def shift_altitudes(self, offset):
        """
        Add a constant offset to all altitudes (in-place).
        
        Parameters
        ----------
        offset : float
            Offset to be added to each altitude (e.g., for ground-level shift).
        """
        self.altitudes += offset

    def sample_profile(self, target_altitudes):
        """
        Interpolate the Cn^2 profile at the specified altitudes.
        
        Parameters
        ----------
        target_altitudes : array_like
            Altitudes at which to interpolate Cn^2.
        
        Returns
        -------
        interpolated_cn2 : np.ndarray
            Interpolated Cn^2 values at the target altitudes.
        """
        return np.interp(target_altitudes, self.altitudes, self.cn2_values)

    # ---------------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------------

    def plot_profile(self, show=True):
        """
        Plot the layered Cn^2 profile against altitude.
        
        Parameters
        ----------
        show : bool
            If True, calls plt.show() at the end.
        """
        plt.figure(figsize=(6, 4))
        plt.plot(self.altitudes, self.cn2_values, 'o-', label='Cn^2')
        plt.title(f'LayeredAtmosphere: {self.name}')
        plt.xlabel('Altitude')
        plt.ylabel('Cn^2')
        plt.grid(True)
        plt.legend()
        if show:
            plt.show()

    def plot_cumulative_profile(self, show=True):
        """
        Plot the cumulative distribution of turbulence as a function of altitude.
        
        Parameters
        ----------
        show : bool
            If True, calls plt.show() at the end.
        """
        alt, cum_weights = self.cumulative_profile()
        plt.figure(figsize=(6, 4))
        plt.plot(alt, cum_weights, 'o-', label='Cumulative Turbulence Fraction')
        plt.title(f'Cumulative Turbulence Profile: {self.name}')
        plt.xlabel('Altitude')
        plt.ylabel('Cumulative Fraction')
        plt.ylim(0, 1.0)
        plt.grid(True)
        plt.legend()
        if show:
            plt.show()

    # ---------------------------------------------------------------------
    # OPD Map Generation for an On-Axis Point Source
    # ---------------------------------------------------------------------

    def generate_opd_map(self, telescope_diameter, npix, wavelength, seed=None):
        """
        Generate a random optical path difference (OPD) map across a telescope pupil
        due to this layered turbulence profile for an on-axis point source at infinity.

        This function:
          1) Computes the total Fried parameter r0_total from the integrated profile.
          2) Splits that turbulence into partial r0_i for each layer based on layer fraction.
          3) Generates a random Kolmogorov phase screen for each layer (in meters).
          4) Sums the layers to produce the total OPD map at the pupil.

        Parameters
        ----------
        telescope_diameter : float
            Diameter of the telescope's circular aperture [meters].
        npix : int
            Number of pixels across the pupil plane grid (NxN).
        wavelength : float
            Observation wavelength [meters].
        seed : int, optional
            Random seed for reproducible phase screens.

        Returns
        -------
        opd_map : 2D numpy.ndarray, shape (npix, npix)
            Optical path difference (meters) across the pupil plane.
            Points outside the circular aperture will also have values
            but are typically masked out in subsequent analysis.
        """

        # Set random seed if provided
        rng = np.random.default_rng(seed)

        # 1) Compute total Fried parameter
        r0_total = self.fried_parameter(wavelength)

        # 2) For each layer i, define fraction_i and partial r0_i
        #    such that (r0_total)^(-5/3) = sum_i (r0_i)^(-5/3).
        #    We'll assume fraction_i = self.weights[i].
        #    Then (r0_i)^(-5/3) = fraction_i * (r0_total)^(-5/3).
        #    => r0_i = r0_total / fraction_i^(3/5).
        r0_layer = []
        alpha = (r0_total)**(-5.0/3.0)
        for w in self.weights:
            if w > 0:
                r0_i = r0_total / (w**(3.0/5.0))
            else:
                r0_i = 1e10  # effectively no turbulence if weight=0
            r0_layer.append(r0_i)

        # 3) Generate phase screens for each layer, then sum
        opd_map_total = np.zeros((npix, npix), dtype=np.float64)

        for i, r0_i in enumerate(r0_layer):
            # Generate a random Kolmogorov screen in meters of OPD
            layer_opd = self._kolmogorov_phase_screen(
                npix, telescope_diameter, r0_i, wavelength, rng
            )
            opd_map_total += layer_opd

        return opd_map_total

    def _kolmogorov_phase_screen(self, N, D, r0, wavelength, rng):
        """
        Generate a 2D Kolmogorov phase screen (in *meters* of optical path) using 
        a basic Fourier-based approach. The returned array is the OPD, not the 
        phase in radians. (Phase in radians would be 2π/λ * OPD.)
        
        Parameters
        ----------
        N : int
            Grid size (N x N).
        D : float
            Physical diameter of the simulation grid [meters].
        r0 : float
            Fried parameter for this layer [meters].
        wavelength : float
            Wavelength [meters].
        rng : np.random.Generator
            A NumPy random generator instance.
        
        Returns
        -------
        opd_screen : 2D np.ndarray, shape (N, N)
            Optical path difference in meters for the Kolmogorov screen.
        """
        # Grid spacing in the pupil plane
        delta = D / N  # [m/pixel]

        # Frequency coordinates in x and y (unshifted)
        fx = np.fft.fftfreq(N, d=delta)  # cycles/m
        fy = np.fft.fftfreq(N, d=delta)
        fx, fy = np.meshgrid(fx, fy)

        # Radial frequency
        f = np.sqrt(fx**2 + fy**2)

        # Convert frequency f to wavenumber k = 2π f
        k = 2.0 * np.pi * f

        # Avoid singularity at the DC bin
        k[0, 0] = 1e-6

        # Kolmogorov phase PSD coefficient
        # PSD_phase(k) = 0.023 * r0^(-5/3) * k^(-11/3)
        C_phi = 0.023 * (r0**(-5.0/3.0))
        psd_phase = C_phi * k**(-11.0/3.0)

        # Convert phase PSD (in rad^2 * m^2) to OPD PSD (in m^2 * m^2)
        # Because OPD = (φ * λ) / (2π)  => PSD_opd = PSD_φ * (λ/(2π))^2
        factor_opd = (wavelength / (2.0 * np.pi))**2
        psd_opd = psd_phase * factor_opd

        # Each frequency bin has amplitude ~ sqrt(PSD_opd * area_in_freq_space)
        # area_in_freq_space = delta_f^2 = (1 / (N*delta))^2
        delta_f = 1.0 / (N * delta)
        amplitude = np.sqrt(psd_opd * (delta_f**2))

        # Random complex field in frequency space: real and imag ~ Normal(0,1)
        random_complex = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
        fourier_field = amplitude * random_complex

        # Inverse FFT to get real-space OPD. 
        # No shifting here, because we used unshifted coordinates above.
        opd_map = np.fft.ifft2(fourier_field)
        opd_map = np.real(opd_map)

        return opd_map

    # ---------------------------------------------------------------------
    # Factory / standard profiles
    # ---------------------------------------------------------------------

    @classmethod
    def from_mauna_kea_median(cls):
        """
        Example approximate median Cn^2 profile from Mauna Kea.
        (Values are purely illustrative; substitute real data if you have it.)
        """
        altitudes = np.array([0, 500, 1000, 2000, 4000, 8000, 12000, 16000])
        cn2_values = np.array([1.2e-15, 8.0e-16, 6.0e-16, 3.5e-16, 
                               2.0e-16, 1.0e-16, 3.0e-17, 1.0e-17])
        return cls(altitudes, cn2_values, name="MaunaKeaMedian")

    @classmethod
    def from_generic_high_altitude_site(cls):
        """
        Another example of a typical high-altitude site profile.
        (Simplified data; for demonstration only.)
        """
        altitudes = np.array([0, 1000, 2000, 3000, 4000, 8000, 10000, 15000])
        cn2_values = np.array([1.0e-15, 7.0e-16, 5.0e-16, 2.8e-16,
                               1.5e-16, 7.0e-17, 2.5e-17, 8.0e-18])
        return cls(altitudes, cn2_values, name="GenericHighAltitude")

    @classmethod
    def from_hufnagel_valley5(cls):
        """
        Constructs a Hufnagel-Valley 5 (HV5) model as a discrete approximation.
        
        HV5 model for Cn^2(h):
        Cn^2(h) = 0.00594 * (v/27)^2 * (10^-5 * h)^10 * exp(-h/1000) 
                  + 2.7e-16 * exp(-h/1500) 
                  + A * exp(-h/100)
        with typical parameters: v=21 m/s, A=1e-13.
        """
        h = np.linspace(0, 20000, 50)  # altitudes in meters
        v = 21.0  # wind speed in m/s
        A = 1e-13
        
        term1 = 0.00594 * (v / 27.0)**2 * (1e-5 * h)**10 * np.exp(-h/1000.0)
        term2 = 2.7e-16 * np.exp(-h/1500.0)
        term3 = A * np.exp(-h/100.0)
        
        cn2_hv5 = term1 + term2 + term3
        
        return cls(h, cn2_hv5, name="HufnagelValley5")


# %% -------------------------------------------------------------------------
# Example of usage:
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Create a small custom layered atmosphere
    alt = [0, 500, 1000, 2000]
    cn2 = [1.0e-15, 0.8e-15, 0.5e-15, 0.2e-15]
    atmosphere = LayeredAtmosphere(alt, cn2, name="CustomExample")
    
    # Print summary
    print(atmosphere)
    
    # Generate a random OPD map for a 4 m telescope, 256 x 256, at 500 nm
    opd = atmosphere.generate_opd_map(
        telescope_diameter=4.0,
        npix=256,
        wavelength=500e-9,  # 500 nm
        seed=42
    )
    
    print("OPD map shape:", opd.shape)
    print("OPD map stats: mean={:.2e} m, std={:.2e} m".format(np.mean(opd), np.std(opd)))
    
    # Show a quick image of the OPD map
    plt.imshow(opd, extent=[-2, 2, -2, 2], cmap='RdBu')
    plt.colorbar(label='OPD (m)')
    plt.title("Simulated OPD Map (4m pupil, 500 nm)")
    plt.xlabel("Pupil X [m]")
    plt.ylabel("Pupil Y [m]")
    plt.show()
#%%