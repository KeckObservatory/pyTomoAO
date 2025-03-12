# tomographicReconstructor.py
"""
    This script computes a tomographic reconstructor to estimate the directional phase 
    in the science direction(s) from multiple Shack-Hartmann wavefront sensord based 
    on the turbulence model given by atmospheric parameters. This tomographic 
    reconstructor is compatible with super resolution. 
"""
# %%
import yaml
import numpy as np
import math
import time
import cProfile
import matplotlib.pyplot as plt
from scipy.sparse import block_diag
from dmParametersClass import dmParameters
from atmosphereParametersClass import atmosphereParameters
from lgsAsterismParametersClass import lgsAsterismParameters
from lgsWfsParametersClass import lgsWfsParameters 
from tomographyParametersClass import tomographyParameters
from scipy.io import loadmat

# Import your utility functions from tomography_utils

from tomography_utils import (
    sparseGradientMatrixAmplitudeWeighted,
    auto_correlation,
    cross_correlation
)

#%%
# Load parameters from config.yaml
# ---------------------------------------
#with open("tomography_config.yaml", "r") as f:
#    config = yaml.safe_load(f)

with open("tomography_config_kapa.yaml", "r") as f:
    config = yaml.safe_load(f)

#%%
# ===== DM PARAMETERS =====
try:
    dmParams = dmParameters(config)
    print("Successfully initialized DM parameters.")
    print(dmParams)
except (ValueError, TypeError) as e:
    print(f"Configuration Error: {e}")
    
#%%
# ===== ATMOSPHERE PARAMETERS =====
try:
    atmParams = atmosphereParameters(config)
    print("Successfully initialized Atmosphere parameters.")
    print(atmParams)
except (ValueError, TypeError) as e:
    print(f"Configuration Error: {e}")

#%%
# ===== LGS ASTERISM =====
try:
    lgsAsterismParams = lgsAsterismParameters(config, atmParams)
    print("Successfully initialized LGS atserism parameters.")
    print(lgsAsterismParams) 
except (ValueError, TypeError) as e:
    print(f"Configuration Error: {e}")

#%%
# ===== LGS WFS PARAMETERS =====
try:
    lgsWfsParams = lgsWfsParameters(config, lgsAsterismParams)
    print("Successfully initialized LGS WFS parameters.")
    print(lgsWfsParams)
except (ValueError, TypeError) as e:
    print(f"Configuration Error: {e}")

#%% 
# ===== TOMOGRAPHY PARAMETERS =====
# For example, the code uses iNoiseVar = 1 / 1e-14. 
iNoiseVar = 1.0 / float(config["noise_parameters"]["iNoiseVar"])

try:
    tomoParams = tomographyParameters(config)
    print("Successfully initialized Tomography parameters.")
    print(tomoParams) 
except (ValueError, TypeError) as e:
    print(f"Configuration Error: {e}")

#%%
# ===== LTAO SPATIO-ANGULAR RECONSTRUCTOR (LINEAR MMSE) SUPPORTING SR =====
start_time = time.perf_counter()

Gamma, gridMask = sparseGradientMatrixAmplitudeWeighted(lgsWfsParams.validLLMapSupport,
                                                        amplMask=None, overSampling=2)
GammaBeta = Gamma/(2*math.pi)

Gamma_list = []
for kGs in range(lgsAsterismParams.nLGS):
    Gamma_list.append(Gamma)

Gamma = block_diag(Gamma_list)

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time : {execution_time:.4f} seconds")

# %%
# ===== AUTO-COVARIANCE MATRIX =====
start_time = time.perf_counter()


tomoParams.sampling = gridMask.shape[0]

# Updates classes properties for Super Resolution
# To be moved as default parameters into the tomographyParametersClass
#lgsWfsParams.wfs_lenslets_rotation = [0,0,0,0]
lgsWfsParams.wfsLensletsOffset = np.array([[0.23/24,-0.23/24,-0.23/24,0.23/24],\
                                            [0.23/24,0.23/24,-0.23/24,-0.23/24]])

# Profile the function
#profiler = cProfile.Profile()
#profiler.enable()
Cxx = auto_correlation(tomoParams,lgsWfsParams, atmParams,lgsAsterismParams,gridMask)
#profiler.disable()
#profiler.print_stats(sort='time')
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time : {execution_time:.4f} seconds")

# %% 
# ===== CROSS-COVARIANCE MATRIX =====
start_time = time.perf_counter()

# Update the tomographyParametersClass to include the fitting weight for each sources
tomoParams.fitSrcWeight = np.ones(tomoParams.nFitSrc**2)/tomoParams.nFitSrc**2

Cox = cross_correlation(tomoParams,lgsWfsParams, atmParams,lgsAsterismParams)

CoxOut = 0
for i in range(tomoParams.nFitSrc**2):
    CoxOut = CoxOut + Cox[i,:,:]*tomoParams.fitSrcWeight[i]

row_mask = gridMask.ravel().astype(bool)
col_mask = np.tile(gridMask.ravel().astype(bool), 4)

# Select submatrix using boolean masks with np.ix_ for correct indexing
Cox = CoxOut[np.ix_(row_mask, col_mask)]

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time : {execution_time:.4f} seconds")

# %%
# ==== Compute the reconstructor =====
CnZ = np.eye(Gamma.shape[0]) * 1/10 * np.mean(np.diag(Gamma @ Cxx @ Gamma.T))
invCss = np.linalg.inv(Gamma @ Cxx @ Gamma.T + CnZ)

RecStatSA = Cox @ Gamma.T @ invCss

# LGS WFS subapertures diameter
d = lgsWfsParams.DSupport/lgsWfsParams.validLLMapSupport.shape[0]

# Size of the pixel at Shannon sampling
wavefront2Meter = lgsAsterismParams.LGSwavelength/d/2

RecStatSA_ = d * wavefront2Meter * RecStatSA

# %%
# ==== Run Reconstructor tests ====

# Test Gamma matrix
mat_data = loadmat('/Users/urielconod/tomographyDataTest/Gamma.mat')
Gamma_matlab = mat_data['Gamma']
gamma_test = np.allclose(Gamma_matlab.toarray(), Gamma.toarray())
print("Gamma matrix test passed:", gamma_test)

# Test auto-correlation matrix
mat_data = loadmat('/Users/urielconod/tomographyDataTest/Cxx.mat')
Cxx_matlab = mat_data['Cxx']
cxx_test = np.allclose(Cxx_matlab, Cxx, rtol=5e-4)
print("Auto-correlation matrix test passed:", cxx_test)

# Test cross-correlation matrix
mat_data = loadmat('/Users/urielconod/tomographyDataTest/Cox.mat')
Cox_matlab = mat_data['Cox']
cox_test = np.allclose(Cox_matlab, Cox, rtol=5e-4)
print("Cross-correlation matrix test passed:", cox_test)

# Test CnZ matrix
mat_data = loadmat('/Users/urielconod/tomographyDataTest/CnZ.mat')
CnZ_matlab = mat_data['CnZ']
cnz_test = np.allclose(CnZ_matlab, CnZ, rtol=5e-4)
print("CnZ test passed:", cnz_test)

# Test invCss matrix
mat_data = loadmat('/Users/urielconod/tomographyDataTest/invCss.mat')
invCss_matlab = mat_data['invCss']
invCss_test = np.allclose(invCss_matlab, invCss, atol=5e-3)
print("invCss test passed:", invCss_test)

# Test reconstructor matrix
mat_data = loadmat('/Users/urielconod/tomographyDataTest/RecStatSAsuperRes.mat')
RecStatSA_matlab = mat_data['RecStatSAsuperRes']
rec_test = np.allclose(RecStatSA_matlab, RecStatSA, atol=5e-3)
print("Reconstructor matrix test passed:", rec_test)

# %%
# ==== Test with slopes generated with Matlab ====
mat_data = loadmat('/Users/urielconod/tomographyDataTest/slopes_3.mat')
slopes_3 = mat_data['slopes_3']

# Load reconstructed phase from Matlab
mat_data = loadmat('/Users/urielconod/tomographyDataTest/wavefront_3.mat')
phase_3 = mat_data['wavefront_3']

# Reconstruct the phase
mask = np.array(gridMask*1, dtype=np.float64)
phase = RecStatSA_ @ slopes_3
phase = phase.flatten()

ones_indices = np.where(mask == 1)
mask[ones_indices] = phase

mask[mask==0]=np.nan

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

img1 = ax1.imshow(mask.T, origin='lower')
fig.colorbar(img1, ax=ax1, fraction=0.047)
ax1.set_aspect('equal')
ax1.set_title(f'Reconstructed OPD (Python, Fast version)\nMean value: {np.nanmean(mask)*1e9:.2f} [nm]')

img2 = ax2.imshow(phase_3, origin='lower')
fig.colorbar(img2, ax=ax2, fraction=0.047)
ax2.set_aspect('equal')
ax2.set_title(f'Reconstructed OPD (Matlab)\nMean value: {np.nanmean(phase_3)*1e9:.2f} [nm]')

diff = phase_3-mask.T
img3 = ax3.imshow(diff, origin='lower')
fig.colorbar(img3, ax=ax3, fraction=0.047)
ax3.set_aspect('equal')
ax3.set_title(f'Difference (Matlab-Python Fast version)\nMean difference: {np.nanmean(diff)*1e9:.2f} [nm]')

plt.tight_layout()

# %%
