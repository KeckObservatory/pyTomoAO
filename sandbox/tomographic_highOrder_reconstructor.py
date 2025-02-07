#%%

import yaml
import numpy as np
import math
import time
from scipy.sparse import coo_matrix, block_diag, csr_matrix, eye, lil_matrix, hstack
from scipy.interpolate import splrep, splev 
from numpy.linalg import pinv
import matplotlib.pyplot as plt

# Import your utility functions from tomography_utils
from tomography_utils import (
    p_bilinearSplineInterp,
    cart2pol,
    make_biharm_operator,
    sparseGradientMatrixAmplitudeWeighted,
    sparseGradientMatrix3x3Stencil,
    create_atm_grid
)


#%%
# Load parameters from config.yaml
# ---------------------------------------
with open("tomography_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ===== LGS WFS PARAMETERS =====
D             = config["lgs_wfs_parameters"]["D"]
nLenslet      = config["lgs_wfs_parameters"]["nLenslet"]
nPx           = config["lgs_wfs_parameters"]["nPx"]
fieldStopSize = config["lgs_wfs_parameters"]["fieldStopSize"]
nLGS          = config["lgs_wfs_parameters"]["nLGS"]

validLLMap_list        = config["lgs_wfs_parameters"]["validLLMap"]
validActuatorMap_list  = config["lgs_wfs_parameters"]["validActuatorMap"]

# Convert these to boolean NumPy arrays
validLLMap       = np.array(validLLMap_list, dtype=bool)
validActuatorMap = np.array(validActuatorMap_list, dtype=bool)

# ===== DM PARAMETERS =====
dmHeights        = np.array(config["dm_parameters"]["dmHeights"])
dmPitch          = np.array(config["dm_parameters"]["dmPitch"])
dmCrossCoupling  = config["dm_parameters"]["dmCrossCoupling"]
nActuators       = np.array(config["dm_parameters"]["nActuators"])
validActuators_3dlist = config["dm_parameters"]["validActuators"]
# In your original code, you only use one DM layer, so we take the first:
validActuators = []
validActuators.append(np.array(validActuators_3dlist, dtype=bool))

# ===== ATMOSPHERE PARAMETERS =====
nLayer        = config["atmosphere_parameters"]["nLayer"]
zenithAngleInDeg = config["atmosphere_parameters"]["zenithAngleInDeg"]
altitude      = np.array(config["atmosphere_parameters"]["altitude"]) * 1e3
L0            = config["atmosphere_parameters"]["L0"]
r0            = config["atmosphere_parameters"]["r0"]
fractionnalR0 = np.array(config["atmosphere_parameters"]["fractionnalR0"])
wavelength    = config["atmosphere_parameters"]["wavelength"]

# Compute airmass if you replicate the original logic
airmass  = 1.0 / math.cos(math.radians(zenithAngleInDeg))
altitude = altitude * airmass

# ===== LGS ASTERISM =====
radiusAst      = config["lgs_asterism"]["radiusAst"]
LGSwavelength  = config["lgs_asterism"]["LGSwavelength"]
baseLGSHeight  = config["lgs_asterism"].get("baseLGSHeight", 90000.0)
LGSheight      = baseLGSHeight * airmass
arcsec2radian  = math.pi/180.0/3600.0

# ===== NOISE PARAMETERS (OPTIONAL) =====
# For example, the code uses iNoiseVar = 1 / 1e-14. 
# Let's say config noise_parameters.iNoiseVar = 1e14, so iNoiseVar = 1 / 1e-14
iNoiseVar = 1.0 / float(config["noise_parameters"]["iNoiseVar"])

# ===== OPTMIZATION PARAMETERS (!!!!!!!!! TO BE MOVED in the tomography_config.yaml !!!!!!!) =====
fitSrcHeight = np.inf
nFitSrc = 7  # number of source for optimization  array of nFitSrc x nFitSrc
fovOptimizaton = 85 # optimization box size in arcseconds if nFitSrc > 1

d = D/nLenslet

#%%
# get reconstruction grid - used in caes where a phase is reconstructed where the actuators are located, or a oversampling is applied


#%%
# LTAO SPATIO-ANGULAR RECONSTRUCTOR (LINEAR MMSE) SUPPORTING SR

Gamma, gridMask = sparseGradientMatrixAmplitudeWeighted(validLLMap, amplMask=None, overSampling=2)
GammaBeta = Gamma/(2*math.pi)

Gamma_list = []
for kGs in range(nLGS):
    Gamma_list.append(Gamma)

Gamma = block_diag(Gamma_list)
# %%
# AUTO-COVARIANCE MATRIX
#Cxx = spatioAngularIrregularCovarianceMatrix(2*nLenslet+1,tel.D, wfs,atm,lgsGs,'mask',gridMask);

