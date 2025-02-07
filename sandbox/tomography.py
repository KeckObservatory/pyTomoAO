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
"""
This main function encapsulates the translation of the original MATLAB script
into Python, detailing the creation of various matrices (Gx, Cnn, iCxx, Ha, Hx)
for a GNAO MCAO reconstructor. It follows the same flow as the MATLAB code,
except that most configuration parameters come from a YAML file.
"""
# ---------------------------------------
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


#%%
# ------------------------------------------------------------
# Now the "Everything Else" part of the code
# ------------------------------------------------------------
tic = time.time()

# Basic derived parameters
dSub        = D / nLenslet
resolution  = nLenslet * nPx

# Build slopesMask: repeat validLLMap for both x,y slopes, then for nLGS
# slopesMask = np.tile(validLLMap.ravel(), (2, 1))  # 2 for x,y slopes
# slopesMask = np.tile(slopesMask, (1, nLGS))       # replicate horizontally
slopesMask = np.tile(validLLMap.ravel()[:, None], (2, nLGS))

# Phase mask construction
nMap = 2 * nLenslet + 1
phaseMask = np.zeros((nMap**2, nLGS), dtype=bool)
iMap0, jMap0 = np.mgrid[0:3, 0:3]

for jLenslet in range(nLenslet):
    for iLenslet in range(nLenslet):
        if validLLMap[iLenslet, jLenslet]:
            for kGs in range(nLGS):
                iOffset = 2 * iLenslet
                jOffset = 2 * jLenslet
                indices = (iMap0 + iOffset) * nMap + (jMap0 + jOffset)
                phaseMask[indices.flatten(), kGs] = True


# Count how many valid lenslets, for reference:
nValidLenslet = np.count_nonzero(validLLMap)

#%% ------------------------------------------------------------
# Build the DM parameters further
# ------------------------------------------------------------
nDmLayer = len(dmHeights)
nValidActuatorsArr = []
for i in range(nDmLayer):
    nValidActuatorsArr.append(np.count_nonzero(validActuators[i]))

# %% ------------------------------------------------------------
# LGS directions
# We create the 3D pointing vectors for each LGS
# ------------------------------------------------------------
LGSdirections = np.array([
    [arcsec2radian*radiusAst*math.sqrt(2), math.radians(225)],
    [arcsec2radian*radiusAst*math.sqrt(2), math.radians(135)],
    [arcsec2radian*radiusAst*math.sqrt(2), math.radians(315)],
    [arcsec2radian*radiusAst*math.sqrt(2), math.radians(45)]
])
directionVectorLGS = np.zeros((3, nLGS))
for i in range(nLGS):
    directionVectorLGS[0, i] = math.tan(LGSdirections[i,0]) * math.cos(LGSdirections[i,1])
    directionVectorLGS[1, i] = math.tan(LGSdirections[i,0]) * math.sin(LGSdirections[i,1])
    directionVectorLGS[2, i] = 1.0

# %% ------------------------------------------------------------
# Optimization Directions (for fitting sources)
# ------------------------------------------------------------

if nFitSrc==1:
    zenithOpt = np.array([0])
    azimuthOpt = np.array([0])
else:
    x = np.linspace(-fovOptimizaton/2, fovOptimizaton/2, nFitSrc)
    x, y = np.meshgrid(x, x)
    theta, rho = np.arctan2(y, x), np.sqrt(x**2 + y**2)
    zenithOpt = rho.flatten() * arcsec2radian
    azimuthOpt = theta.T.flatten() 

directionVectorFitSrc = np.zeros((3, nFitSrc**2))
for i in range(nFitSrc**2):
    directionVectorFitSrc[0, i] = math.tan(zenithOpt[i]) * math.cos(azimuthOpt[i])
    directionVectorFitSrc[1, i] = math.tan(zenithOpt[i]) * math.sin(azimuthOpt[i])
    directionVectorFitSrc[2, i] = 1.0

# %% ------------------------------------------------------------
# Define Atmosphere Grid for each layer
# ------------------------------------------------------------
atmGrid = create_atm_grid(directionVectorLGS, directionVectorFitSrc, nLayer, dSub, altitude, LGSheight, D)

# %% ------------------------------------------------------------
# Set the sparse gradient matrix (3x3 stencil)
# ------------------------------------------------------------
#p_Gamma, gridMask = sparseGradientMatrixAmplitudeWeighted(validLLMap, amplMask=None, overSampling=2)
p_Gamma, gridMask = sparseGradientMatrix3x3Stencil(validLLMap)
# extract the phaseMask from the grid mask (flatten and tiled for nLGS)
# phaseMask = np.tile(gridMask.flatten(order='F'),(nLGS,1)).T

# Scale the matrix
p_Gamma = p_Gamma / (2 * dSub)

# Build block-diagonal Gamma
# MATLAB: vL = repmat(validLLMap(:), 2, 1);
# Flatten validLLMap in column-major order and replicate
vL = np.tile(validLLMap.flatten(order='F'), 2)  # shape: 2*(nLenslet^2)

Gamma_list = []
for kGs in range(nLGS):
    # slopesMask(vL, kGs) => a boolean row selector
    # phaseMask(gridMask(:), kGs) => a boolean column selector
    row_mask = slopesMask[vL, kGs]
    col_mask = phaseMask[gridMask.flatten() , kGs]
    # Select row_mask rows and col_mask columns
    Gamma_kGs = p_Gamma[row_mask, :][:, col_mask]
    Gamma_list.append(Gamma_kGs)

Gamma = block_diag(Gamma_list)

# %% ------------------------------------------------------------
# Build H from GS to WFS
# ------------------------------------------------------------
x_tmp, y_tmp = np.meshgrid(np.linspace(-0.5, 0.5, nMap)*D,
                            np.linspace(-0.5, 0.5, nMap)*D)
grid_complex = x_tmp + 1j*y_tmp

p_H = []
overSampling = np.ones(nLayer)*2
for kGs in range(nLGS):
    Hl = []
    for kLayer in range(nLayer):
        pitchLayer = dSub / overSampling[kLayer]
        height     = altitude[kLayer]
        beta       = directionVectorLGS[:, kGs][:2] * height
        scale      = 1 - height / LGSheight
        pm         = phaseMask[:, kGs]
        idx_pm     = np.where(pm)[0]

        xi = grid_complex.ravel()[idx_pm].real*scale + beta[0]
        yi = grid_complex.ravel()[idx_pm].imag*scale + beta[1]

        x_grid_layer, y_grid_layer = atmGrid[kLayer]
        P = p_bilinearSplineInterp(x_grid_layer, y_grid_layer, pitchLayer, yi, xi)
        Hl.append(P)
    p_H.append(Hl)

H_list = []
for kGs in range(nLGS):
    combined_sparse = hstack(p_H[kGs], format="csr").toarray()
    H_list.append(combined_sparse)
    
H = np.concatenate(H_list, axis=0)

Gx  = Gamma @ H
GxT = Gx.transpose()

# %% ------------------------------------------------------------
# Build approximate iCxx (bi-harmonic operator) as block diag
# ------------------------------------------------------------
p_L2 = []
for kLayer in range(nLayer):
    x_grid_layer, y_grid_layer = atmGrid[kLayer]
    nPxLayerX = len(x_grid_layer)
    nPxLayerY = len(y_grid_layer)

    # This function is from tomography_utils
    p_L = make_biharm_operator(nPxLayerY, nPxLayerX)

    # Example scaling
    L0r0ratio = (L0 / r0)**(5.0/3.0)
    outVar = (24.*math.gamma(6./5)/5)**(5./6.) * \
                (math.gamma(11./6.)*math.gamma(5./6.)/(2.*math.pi**(8./3.))) * L0r0ratio
    outVar *= fractionnalR0[kLayer]

    scaling_cov = outVar * (wavelength/(2*math.pi))**2

    N_ly = nPxLayerX*nPxLayerY
    p_L2_layer = (N_ly / 1e-2) * p_L
    p_L2_layer = p_L2_layer.multiply(scaling_cov)

    p_L2.append(p_L2_layer)

iCxx = block_diag(p_L2)

# %% ------------------------------------------------------------
# Compute recon matrix with optional filtering
# ------------------------------------------------------------
# For simplicity, let Cnn = iNoiseVar * Identity
Cnn = iNoiseVar * eye(np.sum(slopesMask), format='csr')

Right = GxT @ Cnn
Left  = GxT @ Cnn @ Gx + iCxx
# Typically solve Left*x = Right, or do a pinv, etc.
