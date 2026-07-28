# reconstructorAnalyzerRevolt.py
"""
Adaptive Optics Reconstructor Analysis Script
"""
import numpy as np
import matplotlib.pyplot as plt
from pyTomoAO import example_config
from matplotlib.gridspec import GridSpec
from pyTomoAO.fitting import fitting
from pyTomoAO.tomographicReconstructor import tomographicReconstructor
from scipy.linalg import block_diag

# Set dark mode style
plt.style.use('dark_background')

# --- Utility Functions ---
def cart2pol(x, y):
    """Convert Cartesian to polar coordinates"""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def spatial_derivatives(array, pixel_size=1.0):
    """Calculate spatial derivatives (slopes) in x and y directions of a 2D array."""
    # Initialize derivative arrays
    slope_x = np.zeros_like(array)
    slope_y = np.zeros_like(array)
    
    # Calculate x derivatives (using central difference for interior points)
    slope_x[:, 1:-1] = (array[:, 2:] - array[:, :-2]) / (2 * pixel_size)
    slope_x[:, 0] = (array[:, 1] - array[:, 0]) / pixel_size  # Forward difference at left edge
    slope_x[:, -1] = (array[:, -1] - array[:, -2]) / pixel_size  # Backward difference at right edge
    
    # Calculate y derivatives (using central difference for interior points)
    slope_y[1:-1, :] = (array[2:, :] - array[:-2, :]) / (2 * pixel_size)
    slope_y[0, :] = (array[1, :] - array[0, :]) / pixel_size  # Forward difference at top edge
    slope_y[-1, :] = (array[-1, :] - array[-2, :]) / pixel_size  # Backward difference at bottom edge
    
    return slope_x, slope_y

# --- Zernike Polynomial Functions ---
def zernike_defocus(rho, phi):
    """Zernike polynomial Z(2,0) - Defocus"""
    return np.sqrt(3) * (2 * rho**2 - 1)

def zernike_astigmatism_45(rho, phi):
    """Zernike polynomial Z(2,-2) - Astigmatism at 45/135 degrees"""
    return np.sqrt(6) * (rho**2) * np.sin(2*phi)

def zernike_astigmatism_0_90(rho, phi):
    """Zernike polynomial Z(2,2) - Astigmatism at 0/90 degrees"""
    return np.sqrt(6) * (rho**2) * np.cos(2*phi)

def zernike_trefoil_0(rho, phi):
    """Zernike polynomial Z(3,3) - Trefoil at 0 degrees"""
    return np.sqrt(8) * (rho**3) * np.cos(3*phi)

def zernike_trefoil_30(rho, phi):
    """Zernike polynomial Z(3,-3) - Trefoil at 30 degrees"""
    return np.sqrt(8) * (rho**3) * np.sin(3*phi)

class reconstructorAnalyzer:
    def __init__(self, config_path):
        self.reconstructor = tomographicReconstructor(config_path)
        
        # # Create a fitting instance
        self.fit = fitting(self.reconstructor.dmParams)
        
        # Load reconstructors
        self.setup_reconstructors()
        
        # Get sizes for properly setting up meshgrids
        self.wfs_size = self.reconstructor.lgsWfsParams.validLLMapSupport.shape[0]
        
        # Setup meshgrid for wavefront generation
        self.setup_meshgrid()
        
        # Prepare masks
        self.setup_masks()
    
    def setup_reconstructors(self):
        """Load different reconstructors for comparison"""
        # Create the model based reconstructor
        #self.reconstructor.nLGS = 1
        self.alpha_model = 0.001
        self.stretch_factor = 1.1
        self.alpha_im = 10000
        
        self.reconstructor.build_reconstructor(alpha=self.alpha_model)
        

        # Create model base reconstructor
        self.reconstructor.assemble_reconstructor_and_fitting(nChannels=1, slopesOrder="keck", 
                                                            scalingFactor=1.0e4, stretch_factor=self.stretch_factor, 
                                                            rotation=1, flip=1)
        #self.reconstructor.mask_DM_actuators(174)
        # mask outer ring of actuators
        self.masked_actuators = np.load("Masked_actuators_revolt.npy")
        #self.reconstructor.mask_DM_actuators(self.masked_actuators)
        
        self.corner_actuators = np.array([0, 6, 72, 90, 186,204,270,276])
        self.centreExtrapIndex = [118,119,137,138,139,156,157];

        #self.reconstructor.mask_DM_actuators(self.corner_actuators)
        self.reconstructor.mask_DM_actuators(self.centreExtrapIndex)
        self.reconstructor.mask_DM_actuators(self.masked_actuators)

        #self.R = np.flipud(self.reconstructor.FR)
        self.R = self.reconstructor.FR 
        self.FR = self.R
        # Create IM based reconstructor
        IM = np.load('../examples/benchmark/IM_revolt.npy')
        nLGS = self.reconstructor.nLGS
        matrices = [IM] * nLGS
        IM = block_diag(*matrices)

        self.R_im = self.reconstructor.build_reconstructor(IM, alpha=self.alpha_im)
        #self.reconstructor.mask_DM_actuators(self.corner_actuators)
        #self.reconstructor.mask_DM_actuators(self.centreExtrapIndex)
        self.reconstructor.mask_DM_actuators(self.masked_actuators)
        self.R_im = self.R_im[:, :self.reconstructor.lgsWfsParams.nValidSubap*2] 
        # Load alternative reconstructors
        self.R_svd = np.load("reconstructor_revolt_svd.npy")
    
    def setup_meshgrid(self):
        """Create meshgrid for wavefront generation"""
        # Create meshgrid matching the WFS size (24x24 in the original code)
        self.x, self.y = np.meshgrid(
            np.linspace(-1, 1, self.wfs_size), 
            np.linspace(-1, 1, self.wfs_size)
        )
        self.rho, self.phi = cart2pol(self.x, self.y)
    
    def setup_masks(self):
        """Prepare DM and WFS masks"""
        # DM mask
        self.cmd_mask = np.array(self.fit.dmParams.validActuatorsSupport*1, dtype=np.float64)
        self.ones_indices = np.where(self.cmd_mask == 1)
        
        # WFS mask
        self.wfs_mask = np.array(self.reconstructor.lgsWfsParams.validLLMapSupport*1, dtype=np.float64)
        self.ones_indices_wfs = np.where(self.wfs_mask == 1)
    
    def generate_wavefront(self, zernike_func):
        """Generate wavefront and slopes using specified Zernike function"""
        # Generate wavefront using the correct mesh size
        wavefront = zernike_func(self.rho, self.phi)
        
        # Calculate slopes
        slopes_x, slopes_y = spatial_derivatives(wavefront)
        slopes_x = slopes_x.flatten()
        slopes_y = slopes_y.flatten()
        
        # Create slopes vector
        slopes = np.concatenate((
            slopes_x[self.reconstructor.lgsWfsParams.validLLMapSupport.flatten()],
            slopes_y[self.reconstructor.lgsWfsParams.validLLMapSupport.flatten()]
        ))
        
        # Create keck-format slopes (interleaved x,y)
        slopes_keck = np.zeros(self.reconstructor.lgsWfsParams.nValidSubap*2)
        slopes_keck[::2] = np.squeeze(slopes[:self.reconstructor.lgsWfsParams.nValidSubap])
        slopes_keck[1::2] = np.squeeze(slopes[self.reconstructor.lgsWfsParams.nValidSubap:])
        
        # Create flipped slopes for our reconstructors
        slopes_flipped = np.concatenate((
            slopes_y[self.reconstructor.lgsWfsParams.validLLMapSupport.flatten()],
            slopes_x[self.reconstructor.lgsWfsParams.validLLMapSupport.flatten()]
        ))
        
        return wavefront, slopes_x, slopes_y, slopes, slopes_keck, slopes_flipped
    
    def plot_slopes(self, slopes_x, slopes_y, title_prefix):
        """Plot X and Y slopes of a wavefront"""
        fig = plt.figure(figsize=(10, 5), facecolor='black')
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])
        
        # X slopes
        ax1 = fig.add_subplot(gs[0, 0])
        temp_mask = np.copy(self.wfs_mask)
        temp_mask[self.ones_indices_wfs] = \
        slopes_x[self.reconstructor.lgsWfsParams.validLLMapSupport.flatten()]
        im1 = ax1.imshow(temp_mask, cmap='viridis')
        ax1.set_title(f'{title_prefix} X Slopes', color='white')
        ax1.set_xlabel('X (pixels)', color='white')
        ax1.set_ylabel('Y (pixels)', color='white')
        ax1.set_aspect('auto')
        
        # Y slopes
        ax2 = fig.add_subplot(gs[0, 1])
        temp_mask = np.copy(self.wfs_mask)
        temp_mask[self.ones_indices_wfs] = \
        slopes_y[self.reconstructor.lgsWfsParams.validLLMapSupport.flatten()]
        im2 = ax2.imshow(temp_mask, cmap='viridis')
        ax2.set_title(f'{title_prefix} Y Slopes', color='white')
        ax2.set_xlabel('X (pixels)', color='white')
        ax2.set_ylabel('Y (pixels)', color='white')
        ax2.set_aspect('auto')
        
        plt.tight_layout()
        return fig
    
    def plot_reconstructions(self, wavefront, slopes, slopes_keck, slopes_flipped, title_prefix):
        """Plot wavefront and reconstructed DM commands using different reconstructors"""
        fig = plt.figure(figsize=(20, 5), facecolor='black')
        gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 1])
        
        # Original wavefront - handle potential dimension mismatch
        ax1 = fig.add_subplot(gs[0, 0])
        
        if isinstance(wavefront, np.ndarray):
            # Case when wavefront is a numpy array (from Zernike functions)
            # Apply mask
            masked_wavefront = wavefront * self.wfs_mask
            masked_wavefront[masked_wavefront == 0] = np.nan  # Mask 0 values to nan for display
            im1 = ax1.imshow(masked_wavefront, cmap='viridis')
            
        else:
            # Case when wavefront is from reconstructor.reconstruct_wavefront
            try:
                reshaped_wavefront = np.reshape(wavefront, self.wfs_mask.shape)
                reshaped_wavefront[reshaped_wavefront == 0] = np.nan
                im1 = ax1.imshow(reshaped_wavefront, cmap='viridis')
                
            except ValueError:
                print(f"Warning: Could not reshape wavefront of shape {wavefront.shape} to {self.wfs_mask.shape}")
                im1 = ax1.imshow(wavefront, cmap='viridis')
                
        ax1.set_title(f'{title_prefix} Wavefront', color='white')
        ax1.set_xlabel('X (pixels)', color='white')
        ax1.set_ylabel('Y (pixels)', color='white')
        ax1.set_aspect('auto')
        
        # SVD reconstruction
        ax2 = fig.add_subplot(gs[0, 1])
        temp_mask = np.copy(self.cmd_mask)
        temp_mask[self.ones_indices] = self.R_svd @ slopes_keck
        temp_mask[temp_mask == 0] = np.nan
        im2 = ax2.imshow(temp_mask.T, cmap='viridis')
        ax2.set_title('DM commands (R_REVOLT (SVD))', color='white')
        ax2.set_xlabel('X (pixels)', color='white')
        ax2.set_ylabel('Y (pixels)', color='white')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # Tomo model based reconstruction
        ax4 = fig.add_subplot(gs[0, 2])
        temp_mask = np.copy(self.cmd_mask)
        temp_mask[self.ones_indices] = self.R @ slopes_keck
        temp_mask[temp_mask == 0] = np.nan
        im4 = ax4.imshow(temp_mask.T, cmap='viridis')
        ax4.set_title('DM commands (R_Tomo (Model))', color='white')
        ax4.set_xlabel('X (pixels)', color='white')
        ax4.set_ylabel('Y (pixels)', color='white')
        plt.colorbar(im4, ax=ax4, shrink=0.8)
        
        # Tomo IM based reconstruction
        ax5 = fig.add_subplot(gs[0, 3])
        temp_mask = np.copy(self.cmd_mask)
        temp_mask[self.ones_indices] = self.R_im @ slopes_keck
        temp_mask[temp_mask == 0] = np.nan
        im5 = ax5.imshow(temp_mask.T, cmap='viridis')
        ax5.set_title('DM commands (R_Tomo (IM))', color='white')
        ax5.set_xlabel('X (pixels)', color='white')
        ax5.set_ylabel('Y (pixels)', color='white')
        plt.colorbar(im5, ax=ax5, shrink=0.8)
        
        plt.tight_layout()
        
        # display command vector in a separate figure
        fig2 = plt.figure(figsize=(10, 5), facecolor='black')
        plt.plot(self.R_svd @ slopes_keck, label='R_REVOLT (SVD)', color='cyan')
        plt.plot(self.R @ slopes_keck, label='R_Tomo (Model)', color='yellow')
        plt.plot(self.R_im @ slopes_keck, label='R_Tomo (IM)', color='magenta')
        plt.legend()
        plt.title('DM commands', color='white')
        plt.xlabel('DM actuator', color='white')
        plt.ylabel('Command value', color='white')
        plt.grid(True, color='gray', alpha=0.3)
        
        return fig, fig2
    
    def analyze_tip_tilt(self):
        """Analyze tip-tilt wavefront"""
        print("\nAnalyzing Tip-Tilt wavefront...")
        
        # Create tip-tilt slopes directly
        TipTilt = np.zeros(self.reconstructor.lgsWfsParams.nValidSubap * 2)
        TipTilt[0:self.reconstructor.lgsWfsParams.nValidSubap-1] = 1
        TipTilt[self.reconstructor.lgsWfsParams.nValidSubap::] = -1
        #TipTilt = np.tile(TipTilt, self.reconstructor.nLGS)
        
        TT = TipTilt[:self.reconstructor.lgsWfsParams.nValidSubap * 2]
        
        slopes_TT_keck = np.zeros(self.reconstructor.lgsWfsParams.nValidSubap * 2)
        slopes_TT_keck[::2] = np.squeeze(TT[:self.reconstructor.lgsWfsParams.nValidSubap])
        slopes_TT_keck[1::2] = np.squeeze(TT[self.reconstructor.lgsWfsParams.nValidSubap:])
        
        slopes_TT = np.concatenate((np.squeeze(TT[:self.reconstructor.lgsWfsParams.nValidSubap]), 
                                    np.squeeze(TT[self.reconstructor.lgsWfsParams.nValidSubap:])))
        
        # Reconstruct wavefront
        wavefront = self.reconstructor.reconstruct_wavefront(np.tile(slopes_TT, self.reconstructor.nLGS))
        
        # Plot reconstructions
        fig = self.plot_reconstructions(wavefront, slopes_TT_keck, slopes_TT, "Tip-Tilt")
        return fig
    
    def analyze_wavefront(self, zernike_func, title):
        """Analyze a specific Zernike wavefront"""
        print(f"\nAnalyzing {title} wavefront...")
        
        # Generate wavefront and slopes
        wavefront, slopes_x, slopes_y, slopes, slopes_keck, slopes_flipped = self.generate_wavefront(zernike_func)
        
        # Plot slopes
        #fig_slopes = self.plot_slopes(slopes_x, slopes_y, title)
        
        # Plot reconstructions
        fig_recon = self.plot_reconstructions(wavefront, slopes, slopes_keck, slopes_flipped, title)
        
        return fig_recon #, fig_slopes

    def load_interaction_matrix(self, filename):
        """
        Load an interaction matrix matrix from a file.
        
        Parameters:
        -----------
        filename : str
            Path to the file containing the interaction matrix matrix.
        Raises:
        -------
        ValueError
            If the file does not exist or is not in the correct format.
        Returns:
        --------
        self
            For method chaining
        """
        
        try:
            with open(filename, 'rb') as f:
                IM = np.fromfile(f, dtype='>f4').reshape((608, 349))
            print(f"Interaction matrix loaded from {filename}")
        except ValueError:
            raise ValueError("Interaction matrix must be generated first")
        
        return IM

    def save_reconstructor(self, filename):
        """
        Save the generated reconstructor matrix to a file.
        
        Parameters:
        -----------
        filename : str
            Path to save the Reconstructor matrix.
        Raises:
        -------
        ValueError
            If the reconstructor matrix is not generated yet.
        Returns:
        --------
        self
            For method chaining
        """
        
        if self.reconstructor.method == "IM":
            try:
                # Save in the same format as the input
                self.R.astype('>f4').tofile(filename)
                print(f"Reconstructor IM based saved to {filename}")
            except ValueError:
                raise ValueError("Reconstructor IM based must be generated first")
        elif self.reconstructor.method == "Model":
            try:
                # Save in the same format as the input
                self.FR.astype('>f4').tofile(filename)
                print(f"Reconstructor Model based saved to {filename}")
            except ValueError:
                raise ValueError("Reconstructor Model based must be generated first")    
        return self

def main():
    """Main function to run the analysis"""
    # Initialize the analyzer
    analyzer = reconstructorAnalyzer(example_config("revolt"))
    # remove central actuator
    #analyzer.reconstructor.mask_DM_actuators(174)
    # Analyze different Zernike modes
    analyzer.analyze_wavefront(zernike_defocus, "Defocus")
    analyzer.analyze_wavefront(zernike_astigmatism_45, "Astigmatism 45°")
    analyzer.analyze_wavefront(zernike_astigmatism_0_90, "Astigmatism 0/90°")
    analyzer.analyze_wavefront(zernike_trefoil_30, "Trefoil 30°")
    analyzer.analyze_wavefront(zernike_trefoil_0, "Trefoil 0°")
    
    plt.show()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), facecolor='black')
    
    im1 = ax1.imshow(analyzer.R, cmap='viridis')
    plt.colorbar(im1, ax=ax1, shrink=0.5)
    ax1.set_title(f"Model based reconstructor\n(Sum: {np.sum(analyzer.R):.2f})", color='white')
    print(f"Sum of Model based reconstructor: {np.sum(analyzer.R)}")
    
    im2 = ax2.imshow(analyzer.R_im, cmap='viridis')
    plt.colorbar(im2, ax=ax2, shrink=0.5)
    ax2.set_title(f"IM based reconstructor\n(Sum: {np.sum(analyzer.R_im):.2f})", color='white')
    print(f"Sum of IM based reconstructor: {np.sum(analyzer.R_im)}")
    
    im3 = ax3.imshow(analyzer.R_svd, cmap='viridis')
    plt.colorbar(im3, ax=ax3, shrink=0.5)
    ax3.set_title(f"SVD based reconstructor\n(Sum: {np.sum(analyzer.R_svd):.2f})", color='white')
    print(f"Sum of SVD based reconstructor: {np.sum(analyzer.R_svd)}")
    
    plt.tight_layout()
    
    def save_reconstructor_fits(reconstructor, method, alpha, stretch_factor=None, base_name='CM_pyTomoAO'):
        """
        Save reconstructor matrix to FITS file with timestamp and alpha value in filename
        
        Parameters:
        -----------
        reconstructor : numpy.ndarray
            The reconstructor matrix to save
        method : str
            Method used to generate reconstructor ('Model' or 'IM')
        alpha : float
            Alpha value used in reconstructor computation
        base_name : str
            Base name for the output file
        """
        from astropy.io import fits
        from datetime import datetime
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if method == "Model":
            filename = f'{base_name}_{method}_alpha_{alpha:.2f}_stretch_{stretch_factor:.2f}_tel_{timestamp}.fits'
        else:# Create filename with timestamp and alpha
            filename = f'{base_name}_{method}_alpha_{alpha:.2f}_tel_{timestamp}.fits'
        
        # Create and save FITS file
        hdu = fits.PrimaryHDU(reconstructor)
        hdul = fits.HDUList([hdu])
        hdul.writeto(filename, overwrite=True)
        hdul.close()
        
        print(f"Saved {method} reconstructor to {filename}")
    
    
    def create_filtered_reconstructor_known_pattern(M, focus_slope_pattern, focus_fraction=0.5):
        """
        Create filtered reconstructor using known focus pattern in slope space
        
        Parameters:
        M: (277, 376) original reconstructor matrix
        focus_slope_pattern: (376,) known focus pattern in slopes
        focus_fraction: fraction to filter out
        
        Returns:
        M_filtered: (277, 376) filtered reconstructor matrix
        """
        
        # Normalize focus pattern
        focus_mode = focus_slope_pattern / np.linalg.norm(focus_slope_pattern)
        
        # Remove mean from focus pattern
        #focus_demean = focus_slope_pattern - np.mean(focus_slope_pattern)
        #focus_demean_norm = focus_demean / np.linalg.norm(focus_demean)
        
        # Create filter matrix
        P_focus = np.outer(focus_mode, focus_mode)
        I = np.eye(376)
        focus_filter = I - focus_fraction * P_focus
        
        # Premultiply
        M_filtered = M @ focus_filter
        
        return M_filtered
    
    wavefront, slopes_x, slopes_y, slopes, slopes_keck, slopes_flipped = analyzer.generate_wavefront(zernike_defocus)
    R_filtered = create_filtered_reconstructor_known_pattern(analyzer.R, slopes_keck, focus_fraction=0.8)
    analyzer.R = R_filtered
    analyzer.analyze_wavefront(zernike_defocus, "Defocus")

    # Save both reconstructors
    #save_reconstructor_fits(analyzer.R, 'Model', analyzer.alpha_model, analyzer.stretch_factor)
    save_reconstructor_fits(analyzer.R_im, 'IM', analyzer.alpha_im)
    
if __name__ == "__main__":
    main()