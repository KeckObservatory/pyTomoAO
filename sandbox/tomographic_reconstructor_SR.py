# tomographic_reconstructor_SR.py
"""This scrip computes a tomographic reconstructor to estimate the directional phase 
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

# Import your utility functions from tomography_utils

from tomography_utils import (
    sparseGradientMatrixAmplitudeWeighted,
    auto_correlation,
    cross_correlation
)

#%%
# Load parameters from config.yaml
# ---------------------------------------
with open("tomography_config.yaml", "r") as f:
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
# LTAO SPATIO-ANGULAR RECONSTRUCTOR (LINEAR MMSE) SUPPORTING SR
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
# AUTO-COVARIANCE MATRIX
start_time = time.perf_counter()

""" L0_r0_ratio = (atmParams.L0/atmParams.r0)**(5./3.)
cst = (24.*math.gamma(6./5)/5)**(5./6.) * \
         (math.gamma(11./6.)/(2.**(5./6.)*math.pi**(8./3.))) * L0_r0_ratio

cst_L0 = (24.*math.gamma(6./5)/5)**(5./6.) * \
         (math.gamma(11./6.)*math.gamma(5./6.)/(2.*math.pi**(8./3.))) * L0_r0_ratio

cst_r0 = (24.*math.gamma(6./5)/5)**(5./6.) * \
         (math.gamma(11./6.)**2./(2.*math.pi**(11./3.))) * atmParams.r0**(-5./3.)
"""

# Updates classes properties for Super Resolution
# To be moved as default parameters into the tomographyParametersClass
tomoParams.sampling = gridMask.shape[0]
lgsWfsParams.wfs_lenslets_rotation = [0,0,0,0]
lgsWfsParams.wfs_lenslets_offset = np.array([[0.0096,-0.0096,-0.0096,0.0096],\
                                            [0.0096,0.0096,-0.0096,-0.0096]])

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
# CROSS-COVARIANCE MATRIX
start_time = time.perf_counter()

# Update the tomographyParametersClass to include the fitting weight for each sources
tomoParams.fitSrcWeight = np.ones(tomoParams.nFitSrc**2)

Cox = cross_correlation(tomoParams,lgsWfsParams, atmParams,lgsAsterismParams)

CoxOut = 0
for i in range(tomoParams.nFitSrc**2):
    CoxOut = CoxOut + Cox[i,:,:]*tomoParams.fitSrcWeight[i]

Cox_tmp = CoxOut[gridMask.flatten(),np.tile(gridMask.flatten(), 4)]


end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time : {execution_time:.4f} seconds")

# %%

