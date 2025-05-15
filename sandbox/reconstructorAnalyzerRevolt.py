# reconstructorAnalyzerKeck.py
"""
Adaptive Optics Reconstructor Analysis Script
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pyTomoAO.fitting import fitting
from pyTomoAO.tomographicReconstructor import tomographicReconstructor
from scipy.linalg import block_diag


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
        
        # Create a fitting instance
        self.fit = fitting(self.reconstructor.dmParams)
        
        # Setup the influence functions
        self.modes = self.fit.set_influence_function(resolution=41, display=False, sigma1=0.5*2, sigma2=0.85*2, stretch_factor=1.13)
        self.modes = self.modes[self.reconstructor.gridMask.flatten(), :]
        print(f"Modes shape after applying grid mask: {self.modes.shape}")
        
        # Generate a fitting matrix (pseudo-inverse of the influence functions)
        print("\nCalculating fitting matrix")
        self.fit.F = np.linalg.pinv(self.modes)
        print(f"Fitting matrix shape: {self.fit.F.shape}")
        
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
        # Create model base reconstructor
        self.reconstructor.assemble_reconstructor_and_fitting(nChannels=1, slopesOrder="keck", scalingFactor=0, stretch_factor=1.13)
        #self.reconstructor.mask_DM_actuators(174)
        self.R = self.reconstructor.FR
        self.FR = self.R
        
        # Create IM based reconstructor
        IM = np.load('../examples/benchmark/IM_revolt.npy')
        nLGS = self.reconstructor.nLGS
        matrices = [IM] * nLGS
        IM = block_diag(*matrices)
        self.R_im = self.reconstructor.build_reconstructor(IM, alpha=100000)
        self.R_im = self.R_im[:, :self.reconstructor.lgsWfsParams.nValidSubap*2] * 1/4*10

        # Load alternative reconstructors
        self.R_svd = np.load("reconstructor_revolt_svd.npy")
#        self.R_keck = np.load("reconstructor_revolt_svd.npy")
    
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
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])
        
        # X slopes
        ax1 = fig.add_subplot(gs[0, 0])
        temp_mask = np.copy(self.wfs_mask)
        temp_mask[self.ones_indices_wfs] = \
        slopes_x[self.reconstructor.lgsWfsParams.validLLMapSupport.flatten()]
        im1 = ax1.imshow(temp_mask, cmap='gray')
        ax1.set_title(f'{title_prefix} X Slopes')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.set_aspect('auto')
        
        # Y slopes
        ax2 = fig.add_subplot(gs[0, 1])
        temp_mask = np.copy(self.wfs_mask)
        temp_mask[self.ones_indices_wfs] = \
        slopes_y[self.reconstructor.lgsWfsParams.validLLMapSupport.flatten()]
        im2 = ax2.imshow(temp_mask, cmap='gray')
        ax2.set_title(f'{title_prefix} Y Slopes')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        ax2.set_aspect('auto')
        
        plt.tight_layout()
        return fig
    
    def plot_reconstructions(self, wavefront, slopes, slopes_keck, slopes_flipped, title_prefix):
        """Plot wavefront and reconstructed DM commands using different reconstructors"""
        fig = plt.figure(figsize=(20, 5))
        gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 1])
        
        # Original wavefront - handle potential dimension mismatch
        ax1 = fig.add_subplot(gs[0, 0])
        
        if isinstance(wavefront, np.ndarray):
            # Case when wavefront is a numpy array (from Zernike functions)
            # Apply mask
            masked_wavefront = wavefront * self.wfs_mask
            im1 = ax1.imshow(masked_wavefront, cmap='RdBu')
        else:
            # Case when wavefront is from reconstructor.reconstruct_wavefront
            # Need to reshape it properly
            try:
                # Try to reshape to match the WFS size
                reshaped_wavefront = np.reshape(wavefront, self.wfs_mask.shape)
                im1 = ax1.imshow(reshaped_wavefront, cmap='RdBu')
            except ValueError:
                # If reshape fails, just show the original wavefront
                print(f"Warning: Could not reshape wavefront of shape {wavefront.shape} to {self.wfs_mask.shape}")
                im1 = ax1.imshow(wavefront, cmap='RdBu')
                
        ax1.set_title(f'{title_prefix} Wavefront')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.set_aspect('auto')
        
        # SVD reconstruction
        ax2 = fig.add_subplot(gs[0, 1])
        temp_mask = np.copy(self.cmd_mask)
        temp_mask[self.ones_indices] = self.R_svd @ slopes_keck
        im2 = ax2.imshow(temp_mask.T, cmap='RdBu')
        ax2.set_title('DM commands (R_REVOLT (SVD))')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=ax2)
        
        # Tomo model based reconstruction
        ax4 = fig.add_subplot(gs[0, 2])
        temp_mask = np.copy(self.cmd_mask)
        temp_mask[self.ones_indices] = self.R @ slopes_keck

        im4 = ax4.imshow(temp_mask, cmap='RdBu')
        ax4.set_title('DM commands (R_Tomo (Model))')
        ax4.set_xlabel('X (pixels)')
        ax4.set_ylabel('Y (pixels)')
        plt.colorbar(im4, ax=ax4)
        
        # Tomo IM based reconstruction
        ax5 = fig.add_subplot(gs[0, 3])
        temp_mask = np.copy(self.cmd_mask)
        temp_mask[self.ones_indices] = self.R_im @ slopes_keck
        im5 = ax5.imshow(temp_mask.T, cmap='RdBu')
        ax5.set_title('DM commands (R_Tomo (IM))')
        ax5.set_xlabel('X (pixels)')
        ax5.set_ylabel('Y (pixels)')
        plt.colorbar(im5, ax=ax5)
        
        plt.tight_layout()
        
        # display command vector in a separate figure
        fig2 = plt.figure(figsize=(10, 5))
        plt.plot(self.R @ slopes_keck, label='R_Tomo (Model)')
        plt.plot(self.R_im @ slopes_keck, label='R_Tomo (IM)')
        plt.plot(self.R_svd @ slopes_keck, label='R_REVOLT (SVD)')
        plt.legend()
        plt.title('DM commands')
        plt.xlabel('DM actuator')
        plt.ylabel('Command value')
        plt.grid()
        
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
        elif self.reconstructor.method == "model":
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
    analyzer = reconstructorAnalyzer("../examples/benchmark/reconstructor_config_revolt.yaml")
    # remove central actuator
    #analyzer.reconstructor.mask_DM_actuators(174)
    # Analyze different Zernike modes
    analyzer.analyze_wavefront(zernike_defocus, "Defocus")
    analyzer.analyze_wavefront(zernike_astigmatism_45, "Astigmatism 45°")
    analyzer.analyze_wavefront(zernike_astigmatism_0_90, "Astigmatism 0/90°")
    analyzer.analyze_wavefront(zernike_trefoil_30, "Trefoil 30°")
    analyzer.analyze_wavefront(zernike_trefoil_0, "Trefoil 0°")
    
    plt.show()
    # Save the control matrix
    #analyzer.save_reconstructor("RtomoSingleNoTTF.mr")

if __name__ == "__main__":
    main()