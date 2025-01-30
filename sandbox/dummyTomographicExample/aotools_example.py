#%%
import numpy as np
from aotools.functions.pupil import circle
from aotools.turbulence.slopecovariance import CovarianceMatrix
import matplotlib.pyplot as plt
#%%
# Define system parameters
n_wfs = 5  # Number of wavefront sensors
telescope_diameter = 10.0  # Telescope diameter in meters

# Define sub-aperture parameters for each WFS
n_subaps = 20  # Number of sub-apertures along one dimension
subap_diameter = telescope_diameter / n_subaps  # Sub-aperture diameter in meters

# Create pupil masks for each WFS (1 if sub-aperture is active, 0 if not)
pupil_mask = circle(n_subaps/2, n_subaps)
pupil_masks = np.stack([pupil_mask]*n_wfs)

# Define guide star parameters
altitudes = np.zeros(n_wfs) + np.inf
# Altitudes of guide stars (inf for natural guide stars)
gs_altitudes = 1 / altitudes
# Replace infinity with 0
gs_altitudes[gs_altitudes == np.inf] = 0  

gs_positions = np.array([
    [0, 0],       # On-axis guide star
    [10, 0],      # 10 arcsec offset in X
    [0, 10],      # 10 arcsec offset in Y
    [10, 10],      # 10 arcsec offset in both X and Y
    [-10, -10]      # 10 arcsec offset in both X and Y
])  # Positions in arcseconds

# Wavelengths each WFS observes (in meters)
wfs_wavelengths = np.full(n_wfs, 589e-9)  # 500 nm for all WFS

# Define atmospheric layer parameters
n_layers = 1  # Number of atmospheric turbulence layers
layer_altitudes = np.array([0])  # Altitudes in meters
layer_r0s = np.array([0.2])  # Fried parameters in meters
layer_L0s = np.full(n_layers, 25.0)  # Outer scale in meters
# layer_altitudes = np.array([0, 5000, 10000])  # Altitudes in meters
# layer_r0s = np.array([0.2, 0.15, 0.1])  # Fried parameters in meters
# layer_L0s = np.full(n_layers, 25.0)  # Outer scale in meters

# Initialize the CovarianceMatrix object
cov_matrix = CovarianceMatrix(
    n_wfs=n_wfs,
    pupil_masks=pupil_masks,
    telescope_diameter=telescope_diameter,
    subap_diameters=np.full(n_wfs, subap_diameter),
    gs_altitudes=gs_altitudes,
    gs_positions=gs_positions,
    wfs_wavelengths=wfs_wavelengths,
    n_layers=n_layers,
    layer_altitudes=layer_altitudes,
    layer_r0s=layer_r0s,
    layer_L0s=layer_L0s
)

# Compute the covariance matrix
covariance_matrix = cov_matrix.make_covariance_matrix()

# Generate the tomographic reconstructor
svd_conditioning = 0.01  # SVD conditioning parameter
tomographic_reconstructor = cov_matrix.make_tomographic_reconstructor(svd_conditioning=svd_conditioning)

# The tomographic_reconstructor can now be used to reconstruct on-axis slopes
#%%
print(tomographic_reconstructor.shape)
print(2*np.count_nonzero(pupil_masks[1:]))
plt.imshow(tomographic_reconstructor)
plt.show()
# %%
