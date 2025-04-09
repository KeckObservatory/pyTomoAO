# %%

import numpy as np
import matplotlib.pyplot as plt
import numpy as np; from matplotlib.gridspec import GridSpec
import yaml
from pyTomoAO.dmParametersClass import dmParameters

def cart2pol(x, y):
    """Convert Cartesian to polar coordinates"""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def zernike_defocus(rho, phi):
    """
    Zernike polynomial Z(2,0) - Defocus
    Formula: sqrt(3) * (2*rho^2 - 1)
    """
    return np.sqrt(3) * (2 * rho**2 - 1)

def zernike_astigmatism_45(rho, phi):
    """
    Zernike polynomial Z(2,-2) - Astigmatism at 45/135 degrees
    Formula: sqrt(6) * rho^2 * sin(2*phi)
    """
    return np.sqrt(6) * (rho**2) * np.sin(2*phi)


def zernike_astigmatism_0_90(rho, phi):
    """
    Zernike polynomial Z(2,2) - Astigmatism at 0/90 degrees
    Formula: sqrt(6) * rho^2 * cos(2*phi)
    """
    return np.sqrt(6) * (rho**2) * np.cos(2*phi)

def zernike_trefoil_0(rho, phi):
    """
    Zernike polynomial Z(3,3) - Trefoil at 0 degrees
    Formula: sqrt(8) * rho^3 * cos(3*phi)
    """
    return np.sqrt(8) * (rho**3) * np.cos(3*phi)

def zernike_trefoil_30(rho, phi):
    """
    Zernike polynomial Z(3,-3) - Trefoil at 30 degrees
    Formula: sqrt(8) * rho^3 * sin(3*phi)
    """
    return np.sqrt(8) * (rho**3) * np.sin(3*phi)

def spatial_derivatives(array, pixel_size=1.0):
    """
    Calculate spatial derivatives (slopes) in x and y directions of a 2D array.
    
    Parameters:
    array (numpy.ndarray): Input 2D array
    pixel_size (float): Size of each pixel (for scaling)
    
    Returns:
    tuple: (slope_x, slope_y) - Arrays containing derivatives in x and y directions
    """
    # Get array dimensions
    rows, cols = array.shape
    
    # Initialize derivative arrays
    slope_x = np.zeros_like(array)
    slope_y = np.zeros_like(array)
    
    # Calculate x derivatives (using central difference for interior points)
    # For the interior points
    slope_x[:, 1:-1] = (array[:, 2:] - array[:, :-2]) / (2 * pixel_size)
    # For the edge points (forward/backward difference)
    slope_x[:, 0] = (array[:, 1] - array[:, 0]) / pixel_size
    slope_x[:, -1] = (array[:, -1] - array[:, -2]) / pixel_size
    
    # Calculate y derivatives (using central difference for interior points)
    # For the interior points
    slope_y[1:-1, :] = (array[2:, :] - array[:-2, :]) / (2 * pixel_size)
    # For the edge points (forward/backward difference)
    slope_y[0, :] = (array[1, :] - array[0, :]) / pixel_size
    slope_y[-1, :] = (array[-1, :] - array[-2, :]) / pixel_size
    
    return slope_x, slope_y

# %%
# Build the reconstructor and fitting matrix
from pyTomoAO.fitting import fitting
from pyTomoAO.tomographicReconstructor import tomographicReconstructor
reconstructor = tomographicReconstructor("../examples/benchmark/tomography_config_kapa_single_channel.yaml")

# Create a fitting instance
print("\nInitializing fitting object...")
fit = fitting(reconstructor.dmParams)

modes = fit.set_influence_function(resolution=49, display=False, sigma1=0.5*2, sigma2=0.85*2)
modes = modes[reconstructor.gridMask.flatten(), :]
fit.modes = modes
print(f"Modes shape after applying grid mask: {modes.shape}")

# Generate a fitting matrix (pseudo-inverse of the influence functions)
print("\nRecalculating fitting matrix...")
fit.F = np.linalg.pinv(modes)
print(f"Fitting matrix shape: {fit.F.shape}")


R = -fit.F @ reconstructor.reconstructor
R = R[:,:608]
R_svd = np.load("../../Downloads/reconstructor_svd.npy")
R_keck = np.load("../../Downloads/reconstructor.npy")
R_carlos = np.load("../../Downloads/reconstructor_carlos.npy")
R_carlos = -R_carlos[:,:608]

# %%
# Create a meshgrid for the wavefront
x, y = np.meshgrid(np.linspace(-1, 1, 24), np.linspace(-1, 1, 24))
# Convert to polar coordinates
rho, phi = cart2pol(x, y)

# Apply mask
cmd_mask = np.array(fit.dmParams.validActuatorsSupport*1, dtype=np.float64)
ones_indices = np.where(cmd_mask == 1)

wfs_mask = np.array(reconstructor.lgsWfsParams.validLLMapSupport*1, dtype=np.float64)
ones_indices_wfs = np.where(wfs_mask == 1)
# %%
# tip-tilt wavefront
TipTilt = np.zeros(reconstructor.lgsWfsParams.nValidSubap*2)
TipTilt[0:reconstructor.lgsWfsParams.nValidSubap-1]= 1
TipTilt[reconstructor.lgsWfsParams.nValidSubap::]= -1
TipTilt = np.tile(TipTilt,reconstructor.nLGS)

TT = TipTilt[:608]

slopes_TT_keck = np.zeros(608)
slopes_TT_keck[::2] = np.squeeze(TT[:304])    # Set even indices to 4
slopes_TT_keck[1::2] = np.squeeze(TT[304:])  # Set odd indices to -4

slopes_TT = np.concatenate((np.squeeze(TT[:304]),np.squeeze(TT[304:])))
TipTilt = reconstructor.reconstruct_wavefront(np.tile(slopes_TT,4))


fig = plt.figure(figsize=(25, 5))
gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1, 1, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])
ax5 = fig.add_subplot(gs[0, 4])
im1 = ax1.imshow(TipTilt, cmap='gray')
ax1.set_title('Tip-tilt Wavefront')
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')
ax1.set_aspect('auto')
cmd_mask[ones_indices] = -R_svd@slopes_TT_keck
im2 = ax2.imshow(cmd_mask, cmap='gray')
ax2.set_title('DM commands (-R_svd)')
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')
plt.colorbar(im2, ax=ax2)
cmd_mask[ones_indices] = R_keck[:349,:]@slopes_TT_keck
im3 = ax3.imshow(cmd_mask, cmap='gray')
ax3.set_title('DM commands (R_bayes)')
ax3.set_xlabel('X (pixels)')
ax3.set_ylabel('Y (pixels)')
plt.colorbar(im3, ax=ax3)
cmd_mask[ones_indices] = R@slopes_TT
cmd_mask[12,12]=0
im4 = ax4.imshow(cmd_mask, cmap='gray')
ax4.set_title('DM commands (R_tomo)')
ax4.set_xlabel('X (pixels)')
ax4.set_ylabel('Y (pixels)')
plt.colorbar(im4, ax=ax4)
cmd_mask[ones_indices] = R_carlos@slopes_TT
cmd_mask[12,12]=0
im5 = ax5.imshow(cmd_mask, cmap='gray')
ax5.set_title('DM commands (R_tomo (Carlos))')
ax5.set_xlabel('X (pixels)')
ax5.set_ylabel('Y (pixels)')
plt.colorbar(im5, ax=ax5)
plt.tight_layout()



# %%
# Defocus wavefront
defocus = zernike_defocus(rho, phi)
slopes_x, slopes_y = spatial_derivatives(defocus)
slopes_x = slopes_x.flatten()
slopes_y = slopes_y.flatten()

fig = plt.figure(figsize=(10, 5)) 
gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1]) 
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
wfs_mask[ones_indices_wfs] = slopes_x[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()]
im1 = ax1.imshow(wfs_mask, cmap='gray')
ax1.set_title('Defocus Wavefront X Slopes')
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')
ax1.set_aspect('auto')
wfs_mask[ones_indices_wfs] = slopes_y[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()]
im2 = ax2.imshow(wfs_mask, cmap='gray')
ax2.set_title('Defocus Wavefront Y Slopes')
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')
ax2.set_aspect('auto')
plt.tight_layout()


slopes_defocus = np.concatenate((slopes_x[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()],slopes_y[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()]))

slopes_defocus_keck = np.zeros(608)
slopes_defocus_keck[::2] = np.squeeze(slopes_defocus[:304])   
slopes_defocus_keck[1::2] = np.squeeze(slopes_defocus[304:])

slopes_defocus = np.concatenate((slopes_y[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()],slopes_x[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()]))

fig = plt.figure(figsize=(25, 5)) 
gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1, 1, 1]) 
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])
ax5 = fig.add_subplot(gs[0, 4])
im1 = ax1.imshow(defocus*reconstructor.lgsWfsParams.validLLMapSupport, cmap='gray')
ax1.set_title('Defocus Wavefront')
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')
ax1.set_aspect('auto')

cmd_mask[ones_indices] = -R_svd@slopes_defocus_keck 
im2 = ax2.imshow(cmd_mask, cmap='gray')
ax2.set_title('DM commands (-R_svd)')
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')
plt.colorbar(im2, ax=ax2)

cmd_mask[ones_indices] = R_keck[:349,:]@slopes_defocus_keck 
im3 = ax3.imshow(cmd_mask, cmap='gray')
ax3.set_title('DM commands (R_bayes)')
ax3.set_xlabel('X (pixels)')
ax3.set_ylabel('Y (pixels)')
plt.colorbar(im3, ax=ax3)

cmd_mask[ones_indices] = R@slopes_defocus
cmd_mask[12,12]=0 
im4 = ax4.imshow(cmd_mask, cmap='gray')
ax4.set_title('DM commands (R_tomo)')
ax4.set_xlabel('X (pixels)')
ax4.set_ylabel('Y (pixels)')
plt.colorbar(im4, ax=ax4)

cmd_mask[ones_indices] = R_carlos@slopes_defocus 
cmd_mask[12,12]=0
im5 = ax5.imshow(cmd_mask, cmap='gray')
ax5.set_title('DM commands (R_tomo (Carlos))')
ax5.set_xlabel('X (pixels)')
ax5.set_ylabel('Y (pixels)')
plt.colorbar(im5, ax=ax5)

plt.tight_layout() 
#plt.savefig('reconstruction_comparison.png') 
#print('Plot saved as reconstruction_comparison.png')


# %%
# Astigmatism 45°  wavefront
astigmat = zernike_astigmatism_45(rho, phi)
slopes_x, slopes_y = spatial_derivatives(astigmat)
slopes_x = slopes_x.flatten()
slopes_y = slopes_y.flatten()

fig = plt.figure(figsize=(10, 5)) 
gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1]) 
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
wfs_mask[ones_indices_wfs] = slopes_x[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()]
im1 = ax1.imshow(wfs_mask, cmap='gray')
ax1.set_title('Astigmatism 45° Wavefront X Slopes')
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')
ax1.set_aspect('auto')
wfs_mask[ones_indices_wfs] = slopes_y[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()]
im2 = ax2.imshow(wfs_mask, cmap='gray')
ax2.set_title('Astigmatism 45° Wavefront Y Slopes')
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')
ax2.set_aspect('auto')
plt.tight_layout()

slopes_astigmat = np.concatenate((slopes_x[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()],slopes_y[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()]))
slopes_astigmat_keck = np.zeros(608)
slopes_astigmat_keck[::2] = np.squeeze(slopes_astigmat[:304])
slopes_astigmat_keck[1::2] = np.squeeze(slopes_astigmat[304:])

slopes_astigmat = np.concatenate((slopes_y[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()],slopes_x[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()]))

fig = plt.figure(figsize=(25, 5)) 
gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1, 1, 1]) 
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])
ax5 = fig.add_subplot(gs[0, 4])
im1 = ax1.imshow(astigmat*reconstructor.lgsWfsParams.validLLMapSupport, cmap='gray')
ax1.set_title('Astigmatism Wavefront')
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')
ax1.set_aspect('auto')

cmd_mask[ones_indices] = -R_svd@slopes_astigmat_keck 
im2 = ax2.imshow(cmd_mask, cmap='gray')
ax2.set_title('DM commands (-R_svd)')
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')
plt.colorbar(im2, ax=ax2)

cmd_mask[ones_indices] = R_keck[:349,:]@slopes_astigmat_keck 
im3 = ax3.imshow(cmd_mask, cmap='gray')
ax3.set_title('DM commands (R_bayes)')
ax3.set_xlabel('X (pixels)')
ax3.set_ylabel('Y (pixels)')
plt.colorbar(im3, ax=ax3)

cmd_mask[ones_indices] = R@slopes_astigmat 

im4 = ax4.imshow(cmd_mask, cmap='gray')
ax4.set_title('DM commands (R_tomo)')
ax4.set_xlabel('X (pixels)')
ax4.set_ylabel('Y (pixels)')
plt.colorbar(im4, ax=ax4)

cmd_mask[ones_indices] = R_carlos@slopes_astigmat 
cmd_mask[12,12]=0
im5 = ax5.imshow(cmd_mask, cmap='gray')
ax5.set_title('DM commands (R_tomo (Carlos))')
ax5.set_xlabel('X (pixels)')
ax5.set_ylabel('Y (pixels)')
plt.colorbar(im5, ax=ax5)

plt.tight_layout() 
#plt.savefig('reconstruction_comparison.png') 
#print('Plot saved as reconstruction_comparison.png')

# %%
# Astigmatism 90° wavefront
astigmat = zernike_astigmatism_0_90(rho, phi)
slopes_x, slopes_y = spatial_derivatives(astigmat)
slopes_x = slopes_x.flatten()
slopes_y = slopes_y.flatten()

fig = plt.figure(figsize=(10, 5)) 
gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1]) 
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
wfs_mask[ones_indices_wfs] = slopes_x[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()]
im1 = ax1.imshow(wfs_mask, cmap='gray')
ax1.set_title('Astigmatism 90° Wavefront X Slopes')
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')
ax1.set_aspect('auto')
wfs_mask[ones_indices_wfs] = slopes_y[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()]
im2 = ax2.imshow(wfs_mask, cmap='gray')
ax2.set_title('Astigmatism 90° Wavefront Y Slopes')
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')
ax2.set_aspect('auto')
plt.tight_layout()

slopes_astigmat = np.concatenate((slopes_x[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()],slopes_y[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()]))
slopes_astigmat_keck = np.zeros(608)
slopes_astigmat_keck[::2] = np.squeeze(slopes_astigmat[:304])
slopes_astigmat_keck[1::2] = np.squeeze(slopes_astigmat[304:])

slopes_astigmat = np.concatenate((slopes_y[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()],slopes_x[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()]))

fig = plt.figure(figsize=(25, 5)) 
gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1, 1, 1]) 
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])
ax5 = fig.add_subplot(gs[0, 4])
im1 = ax1.imshow(astigmat*reconstructor.lgsWfsParams.validLLMapSupport, cmap='gray')
ax1.set_title('Astigmatism Wavefront')
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')
ax1.set_aspect('auto')

cmd_mask[ones_indices] = -R_svd@slopes_astigmat_keck 
im2 = ax2.imshow(cmd_mask, cmap='gray')
ax2.set_title('DM commands (-R_svd)')
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')
plt.colorbar(im2, ax=ax2)

cmd_mask[ones_indices] = R_keck[:349,:]@slopes_astigmat_keck 
im3 = ax3.imshow(cmd_mask, cmap='gray')
ax3.set_title('DM commands (R_bayes)')
ax3.set_xlabel('X (pixels)')
ax3.set_ylabel('Y (pixels)')
plt.colorbar(im3, ax=ax3)

cmd_mask[ones_indices] = R@slopes_astigmat 

im4 = ax4.imshow(cmd_mask, cmap='gray')
ax4.set_title('DM commands (R_tomo)')
ax4.set_xlabel('X (pixels)')
ax4.set_ylabel('Y (pixels)')
plt.colorbar(im4, ax=ax4)

cmd_mask[ones_indices] = R_carlos@slopes_astigmat 
cmd_mask[12,12]=0
im5 = ax5.imshow(cmd_mask, cmap='gray')
ax5.set_title('DM commands (R_tomo (Carlos))')
ax5.set_xlabel('X (pixels)')
ax5.set_ylabel('Y (pixels)')
plt.colorbar(im5, ax=ax5)

plt.tight_layout() 
#plt.savefig('reconstruction_comparison.png') 
#print('Plot saved as reconstruction_comparison.png')


# %%
# Trefoil 30° wavefront
trefoil = zernike_trefoil_30(rho, phi)
slopes_x, slopes_y = spatial_derivatives(trefoil)
slopes_x = slopes_x.flatten()
slopes_y = slopes_y.flatten()
slopes_trefoil = np.concatenate((slopes_x[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()],slopes_y[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()]))

fig = plt.figure(figsize=(10, 5)) 
gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1]) 
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
wfs_mask[ones_indices_wfs] = slopes_x[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()]
im1 = ax1.imshow(wfs_mask, cmap='gray')
ax1.set_title('Trefoil 30° Wavefront X Slopes')
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')
ax1.set_aspect('auto')
wfs_mask[ones_indices_wfs] = slopes_y[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()]
im2 = ax2.imshow(wfs_mask, cmap='gray')
ax2.set_title('Trefoil 30° Wavefront Y Slopes')
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')
ax2.set_aspect('auto')
plt.tight_layout()

slopes_trefoil_keck = np.zeros(608)
slopes_trefoil_keck[::2] = np.squeeze(slopes_trefoil[:304])
slopes_trefoil_keck[1::2] = np.squeeze(slopes_trefoil[304:])

slopes_trefoil = np.concatenate((slopes_y[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()],slopes_x[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()]))

fig = plt.figure(figsize=(25, 5))
gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1, 1, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])
ax5 = fig.add_subplot(gs[0, 4])
im1 = ax1.imshow(trefoil*reconstructor.lgsWfsParams.validLLMapSupport, cmap='gray')
ax1.set_title('Trefoil Wavefront')
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')
ax1.set_aspect('auto')
cmd_mask[ones_indices] = -R_svd@slopes_trefoil_keck
im2 = ax2.imshow(cmd_mask, cmap='gray')
ax2.set_title('DM commands (-R_svd)')
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')
plt.colorbar(im2, ax=ax2)
cmd_mask[ones_indices] = R_keck[:349,:]@slopes_trefoil_keck
im3 = ax3.imshow(cmd_mask, cmap='gray')
ax3.set_title('DM commands (R_bayes)')
ax3.set_xlabel('X (pixels)')
ax3.set_ylabel('Y (pixels)')
plt.colorbar(im3, ax=ax3)
cmd_mask[ones_indices] = R@slopes_trefoil
#cmd_mask[12,12]=0
im4 = ax4.imshow(cmd_mask, cmap='gray')
ax4.set_title('DM commands (R_tomo)')
ax4.set_xlabel('X (pixels)')
ax4.set_ylabel('Y (pixels)')
plt.colorbar(im4, ax=ax4)   
cmd_mask[ones_indices] = R_carlos@slopes_trefoil
#cmd_mask[12,12]=0
im5 = ax5.imshow(cmd_mask, cmap='gray')
ax5.set_title('DM commands (R_tomo (Carlos))')
ax5.set_xlabel('X (pixels)')
ax5.set_ylabel('Y (pixels)')
plt.colorbar(im5, ax=ax5)
plt.tight_layout()
#plt.savefig('reconstruction_comparison.png')
#print('Plot saved as reconstruction_comparison.png')
# %%
# Trefoil 0° wavefront
trefoil = zernike_trefoil_0(rho, phi)
slopes_x, slopes_y = spatial_derivatives(trefoil)
slopes_x = slopes_x.flatten()
slopes_y = slopes_y.flatten()
slopes_trefoil = np.concatenate((slopes_x[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()],slopes_y[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()]))

fig = plt.figure(figsize=(10, 5)) 
gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1]) 
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
wfs_mask[ones_indices_wfs] = slopes_x[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()]
im1 = ax1.imshow(wfs_mask, cmap='gray')
ax1.set_title('Trefoil 0° Wavefront X Slopes')
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')
ax1.set_aspect('auto')
wfs_mask[ones_indices_wfs] = slopes_y[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()]
im2 = ax2.imshow(wfs_mask, cmap='gray')
ax2.set_title('Trefoil 0° Wavefront Y Slopes')
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')
ax2.set_aspect('auto')
plt.tight_layout()


slopes_trefoil_keck = np.zeros(608)
slopes_trefoil_keck[::2] = np.squeeze(slopes_trefoil[:304])
slopes_trefoil_keck[1::2] = np.squeeze(slopes_trefoil[304:])

slopes_trefoil = np.concatenate((slopes_y[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()],slopes_x[reconstructor.lgsWfsParams.validLLMapSupport
.flatten()]))


fig = plt.figure(figsize=(25, 5))
gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1, 1, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])
ax5 = fig.add_subplot(gs[0, 4])
im1 = ax1.imshow(trefoil*reconstructor.lgsWfsParams.validLLMapSupport, cmap='gray')
ax1.set_title('Trefoil Wavefront')
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')
ax1.set_aspect('auto')
cmd_mask[ones_indices] = -R_svd@slopes_trefoil_keck
im2 = ax2.imshow(cmd_mask, cmap='gray')
ax2.set_title('DM commands (-R_svd)')
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')
plt.colorbar(im2, ax=ax2)
cmd_mask[ones_indices] = R_keck[:349,:]@slopes_trefoil_keck
im3 = ax3.imshow(cmd_mask, cmap='gray')
ax3.set_title('DM commands (R_bayes)')
ax3.set_xlabel('X (pixels)')
ax3.set_ylabel('Y (pixels)')
plt.colorbar(im3, ax=ax3)
cmd_mask[ones_indices] = R@slopes_trefoil
#cmd_mask[12,12]=0
im4 = ax4.imshow(cmd_mask, cmap='gray')
ax4.set_title('DM commands (R_tomo)')
ax4.set_xlabel('X (pixels)')
ax4.set_ylabel('Y (pixels)')
plt.colorbar(im4, ax=ax4)   
cmd_mask[ones_indices] = R_carlos@slopes_trefoil
#cmd_mask[12,12]=0
im5 = ax5.imshow(cmd_mask, cmap='gray')
ax5.set_title('DM commands (R_tomo (Carlos))')
ax5.set_xlabel('X (pixels)')
ax5.set_ylabel('Y (pixels)')
plt.colorbar(im5, ax=ax5)
plt.tight_layout()



# %%
