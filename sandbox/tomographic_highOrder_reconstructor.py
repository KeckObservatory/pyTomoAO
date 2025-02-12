# %%
import yaml
import numpy as np
import math
import time
from scipy.sparse import coo_matrix, block_diag, csr_matrix, eye, lil_matrix, hstack
from scipy.interpolate import splrep, splev 
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from numbers import Number
from lgsWfsParametersClass import lgsWfsParameters 
from dmParametersClass import dmParameters
from atmosphereParametersClass import atmosphereParameters
from lgsAsterismParametersClass import lgsAsterismParameters
from tomographyParametersClass import tomographyParameters

# Import your utility functions from tomography_utils
from tomography_utils import (
    p_bilinearSplineInterp,
    cart2pol,
    pol2cart,
    make_biharm_operator,
    sparseGradientMatrixAmplitudeWeighted,
    sparseGradientMatrix3x3Stencil,
    create_atm_grid,
    auto_correlation,
    rotateDM,
    covariance_matrix
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

Gamma, gridMask = sparseGradientMatrixAmplitudeWeighted(lgsWfsParams.validLLMapSupport, amplMask=None, overSampling=2)
GammaBeta = Gamma/(2*math.pi)

Gamma_list = []
for kGs in range(lgsAsterismParams.nLGS):
    Gamma_list.append(Gamma)

Gamma = block_diag(Gamma_list)

# %%
# AUTO-COVARIANCE MATRIX
#Cxx = spatioAngularIrregularCovarianceMatrix(2*nLenslet+1,tel.D, wfs,atm,lgsGs,'mask',gridMask);
##### TO BE COVERTTED TO A FUNCTION ######### 

#def spatioAngularIrregularCovarianceMatrix(sampling, range,wfs,atm,srcAC,varargin)
# computes the spatio-Angular auto-correlation of the

L0_r0_ratio = (atmParams.L0/atmParams.r0)**(5./3.)
cst = (24.*math.gamma(6./5)/5)**(5./6.) * \
         (math.gamma(11./6.)/(2.**(5./6.)*math.pi**(8./3.))) * L0_r0_ratio

cst_L0 = (24.*math.gamma(6./5)/5)**(5./6.) * \
         (math.gamma(11./6.)*math.gamma(5./6.)/(2.*math.pi**(8./3.))) * L0_r0_ratio

cst_r0 = (24.*math.gamma(6./5)/5)**(5./6.) * \
         (math.gamma(11./6.)**2./(2.*math.pi**(11./3.))) * atmParams.r0**(-5./3.)

windVx, windVy = pol2cart(atmParams.windDirection,atmParams.windSpeed)


sampling = gridMask.shape[0]
wfs_lenslets_rotation = [0,0,0,0]
wfs_lenslets_offset = np.array([[0.0096,-0.0096,-0.0096,0.0096],[0.0096,0.0096,-0.0096,-0.0096]])


S = auto_correlation(sampling,lgsWfsParams.D,wfs_lenslets_rotation,wfs_lenslets_offset,gridMask,lgsWfsParams.nLGS,lgsAsterismParams.directionVectorLGS,\
                    lgsAsterismParams.LGSheight,atmParams.nLayer,atmParams.altitude,atmParams.fractionnalR0,atmParams.r0,atmParams.L0)


# %%
