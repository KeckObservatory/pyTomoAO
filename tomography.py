#%%
import yaml
import numpy as np
import math
import time
from scipy.sparse import coo_matrix, block_diag, csr_matrix, eye, lil_matrix
from scipy.interpolate import splrep, splev 
from numpy.linalg import pinv
import matplotlib.pyplot as plt

# Import your utility functions from tomography_utils
from tomography_utils import (
    p_bilinearSplineInterp,
    cart2pol,
    make_biharm_operator
)

#%%
"""
This main function encapsulates the translation of the original MATLAB script
into Python, detailing the creation of various matrices (Gx, Cnn, iCxx, Ha, Hx)
for a GNAO MCAO reconstructor. It follows the same flow as the MATLAB code,
except that most configuration parameters come from a YAML file.
"""
# ---------------------------------------
# 1) Load parameters from config.yaml
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

# nMap for wavefront
nMap = 2*nLenslet + 1

# Build phaseMask array
iMap0, jMap0 = np.mgrid[0:3, 0:3]
iMap0 += 1  # match 1-based indexing from MATLAB
jMap0 += 1
phaseMask = np.zeros((nMap*nMap, nLGS), dtype=bool)

for jLenslet in range(nLenslet):
    jOffset = 2 * jLenslet
    for iLenslet in range(nLenslet):
        index1 = jLenslet + nLenslet*(iLenslet)
        iOffset = 2 * iLenslet
        # If valid in slopesMask, mark the 3x3 patch
        for kGs in range(nLGS):
            if slopesMask[index1, kGs]:
                index2 = (iMap0 + iOffset - 1) + (jMap0 + jOffset - 1)*nMap
                index2_flat = index2.ravel() - 1  # 0-based
                phaseMask[index2_flat, kGs] = True

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
nFitSrc      = 49
fitSrcHeight = np.inf

x_  = np.linspace(-85/2, 85/2, 7)
xx, yy = np.meshgrid(x_, x_)
theta, rho = cart2pol(xx, yy)  # from tomography_utils

zenithOpt  = rho.ravel() * arcsec2radian
azimuthOpt = theta.ravel()

directionVectorFitSrc = np.zeros((3, nFitSrc))
for i in range(nFitSrc):
    directionVectorFitSrc[0, i] = math.tan(zenithOpt[i]) * math.cos(azimuthOpt[i])
    directionVectorFitSrc[1, i] = math.tan(zenithOpt[i]) * math.sin(azimuthOpt[i])
    directionVectorFitSrc[2, i] = 1.0

# %% ------------------------------------------------------------
# Define Atmosphere Grid for each layer
# ------------------------------------------------------------
vg = directionVectorLGS[0:2, :]    # Just the x,y
vs = directionVectorFitSrc[0:2, :] # x,y for fitting sources

atmGrid = []
overSampling = np.ones(nLayer) * 2

for kLayer in range(nLayer):
    pitchLayer = dSub / overSampling[kLayer]
    height     = altitude[kLayer]
    mVal       = 1 - height / LGSheight

    dDirecG = vg * height
    dDirecS = vs * height

    # min/max
    dmin = np.min(np.concatenate([dDirecG - D/2*mVal, dDirecS - D/2], axis=1), axis=1)
    dmax = np.max(np.concatenate([dDirecG + D/2*mVal, dDirecS + D/2], axis=1), axis=1)

    nPxLayerX = int(math.floor((dmax[0] - dmin[0]) / pitchLayer)) + 2
    nPxLayerY = int(math.floor((dmax[1] - dmin[1]) / pitchLayer)) + 2

    Dx = (nPxLayerX - 1)*pitchLayer
    Dy = (nPxLayerY - 1)*pitchLayer

    sx = dmin[0] - (Dx - (dmax[0]-dmin[0]))/2
    sy = dmin[1] - (Dy - (dmax[1]-dmin[1]))/2

    x_grid = np.linspace(0, 1, nPxLayerX)*Dx + sx
    y_grid = np.linspace(0, 1, nPxLayerY)*Dy + sy

    atmGrid.append((x_grid, y_grid))

# %% ------------------------------------------------------------
# Set the sparse gradient matrix (3x3 stencil)
# ------------------------------------------------------------
# Stencil definitions, etc.
i0x = np.array([0,0,0,0,0,0]) + 1
j0x = np.array([0,1,2,0,1,2]) + 1
i0y = np.array([0,2,0,2,0,2]) + 1
j0y = np.array([0,0,1,1,2,2]) + 1

s0x = np.array([-1, -2, -1, 1, 2, 1], dtype=float)/2
s0y = -np.array([1, -1, 2, -2, 1, -1], dtype=float)/2

i_x = np.zeros(6*nValidLenslet, dtype=int)
j_x = np.zeros(6*nValidLenslet, dtype=int)
s_x = np.zeros(6*nValidLenslet, dtype=float)
i_y = np.zeros(6*nValidLenslet, dtype=int)
j_y = np.zeros(6*nValidLenslet, dtype=int)
s_y = np.zeros(6*nValidLenslet, dtype=float)

gridMask = np.zeros((nMap, nMap), dtype=bool)

idx_counter = 0
for jLenslet in range(nLenslet):
    jOffset = 2*jLenslet
    for iLenslet in range(nLenslet):
        if validLLMap[iLenslet, jLenslet]:
            iOffset = 2*iLenslet

            i_x[idx_counter:idx_counter+6] = i0x + iOffset
            j_x[idx_counter:idx_counter+6] = j0x + jOffset
            s_x[idx_counter:idx_counter+6] = s0x

            i_y[idx_counter:idx_counter+6] = i0y + iOffset
            j_y[idx_counter:idx_counter+6] = j0y + jOffset
            s_y[idx_counter:idx_counter+6] = s0y

            # Update gridMask
            for ii in range(3):
                for jj in range(3):
                    gridMask[iOffset + ii, jOffset + jj] = True

            idx_counter += 6

# Sub2ind utility
def sub2ind(shape, row, col):
    return row * shape[1] + col

i_x0 = i_x - 1
j_x0 = j_x - 1
i_y0 = i_y - 1
j_y0 = j_y - 1

indx = sub2ind((nMap, nMap), i_x0, j_x0)
indy = sub2ind((nMap, nMap), i_y0, j_y0)

v_ = np.arange(2*nValidLenslet)
v_repeated = np.repeat(v_, 6)

data_x = np.concatenate([s_x, s_y])
cols_x = np.concatenate([indx, indy])
rows_x = np.concatenate([v_repeated, v_repeated])

p_Gamma_coo = coo_matrix((data_x, (rows_x, cols_x)), shape=(2*nValidLenslet, nMap**2))
p_Gamma = p_Gamma_coo.tocsr()

# Remove columns not used
gridMask_flat = gridMask.ravel()
p_Gamma = p_Gamma[:, gridMask_flat]

p_Gamma = p_Gamma / (2*dSub)

# Build block-diagonal Gamma
Gamma_list = []
for kGs in range(nLGS):
    rowMask = slopesMask[:, kGs]
    phaseMaskSlice = phaseMask[:, kGs]
    colMask = np.logical_and(phaseMaskSlice, gridMask_flat)
    Gamma_kGs = p_Gamma[rowMask, :][:, colMask]
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
        P = p_bilinearSplineInterp(x_grid_layer, y_grid_layer, pitchLayer, xi, yi)
        Hl.append(P)
    p_H.append(Hl)

# For each LGS, horizontally stack across nLayer, then block diag across nLGS
H_list = []
for kGs in range(nLGS):
    H_kGs = csr_matrix(np.hstack(p_H[kGs]))
    H_list.append(H_kGs)
H = block_diag(H_list)

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

# %% ------------------------------------------------------------
# Influence matrix (Gaussian IF model) demonstration
# ------------------------------------------------------------
ratioTelDm = 1.0
offset     = np.zeros(2)

c   = 1.0/np.sqrt(np.log(1.0/dmCrossCoupling))
df  = 1e-10
mx  = np.sqrt(-np.log(df))*c
xvals = np.linspace(-mx, mx, 1001)
fvals = np.exp(-xvals**2/c**2)

dmInfFuncMatrix = None
iDmInfFuncMatrix = None
nValidActuatorTotal = 0
layersNPixel = []
D_m = []

# We'll store the final "modes" in a Python list, one entry per DM layer
modes = [None]*nDmLayer
iFCell = [None]*nDmLayer  # holds the inverse of F for each layer

nValidActuatorsArr = []
for i in range(nDmLayer):
    tmp = validActuators[i]
    nValidActuatorsArr.append(np.count_nonzero(tmp))
actCoordVector = np.zeros(nDmLayer)

# Loop over each DM layer
for kDmLayer in range(nDmLayer):
    # ------------------------------------------------------------
    # 1) Build the "gaussian" array and the spline
    #    (Equivalent to "gaussian(:,1) = x*dmPitch(kDmLayer); gaussian(:,2) = f; splineP = spline(...).")
    # ------------------------------------------------------------
    # In MATLAB, 'x' was multiplied by dmPitch, and 'f' was the Gaussian samples.
    # We'll store them if needed, but usually only the spline is essential:
    gaussian = np.column_stack([
        xvals * dmPitch[kDmLayer],  # "x * dmPitch(kDmLayer)"
        fvals                       # "f"
    ])
    # Build the spline (similar to "spline(x*dmPitch(kDmLayer), f)")
    splineP = splrep(xvals * dmPitch[kDmLayer], fvals, s=0)

    # ------------------------------------------------------------
    # 2) Build the DM actuator coordinate grid (xIF,yIF), then complex form
    # ------------------------------------------------------------
    nA = nActuators[kDmLayer]
    pitch = dmPitch[kDmLayer]

    # In MATLAB:
    #  xIF = linspace(-1,1,nA)*(nA-1)/2*dmPitch(kDmLayer) - offset(1)
    #  yIF = linspace(-1,1,nA)*(nA-1)/2*dmPitch(kDmLayer) - offset(2)
    xIF = np.linspace(-1, 1, nA) * ((nA - 1)/2) * pitch - offset[0]
    yIF = np.linspace(-1, 1, nA) * ((nA - 1)/2) * pitch - offset[1]

    # ndgrid => np.meshgrid
    xIF2, yIF2 = np.meshgrid(xIF, yIF)
    # In MATLAB: actCoordVector{kDmLayer} = yIF2 + 1i*flip(xIF2)
    # We'll follow that logic exactly:
    # 'flip(xIF2)' in the first dimension => np.flipud(xIF2)
    # The real part is from 'yIF2', the imaginary from 'flip'
    actCoord = yIF2 + 1j*np.flipud(xIF2)

    # Suppose you have a list to store 'actCoordVector':
    actCoordVector[kDmLayer] = actCoord

    # ------------------------------------------------------------
    # 3) Compute D_m (width in real space), layersNPixel, etc.
    # ------------------------------------------------------------
    D_m[kDmLayer] = np.max(actCoord.real) - np.min(actCoord.real)
    do = dSub
    layersNPixel[kDmLayer] = int(round(D_m[kDmLayer] / do)) + 1

    # Update a total count of valid actuators across DMs
    nValidActuatorTotal += nValidActuatorsArr[kDmLayer]

    # ------------------------------------------------------------
    # 4) Build the 1D coordinate 'u0' that samples the DM grid
    #    "u0 = ratioTelDm .* linspace(-1,1,layersNPixel(kDmLayer))*(nA-1)/2 * pitch"
    # ------------------------------------------------------------
    u0 = (ratioTelDm *
          np.linspace(-1, 1, layersNPixel[kDmLayer]) *
          ((nA - 1) / 2) * pitch)

    # ------------------------------------------------------------
    # 5) Build 'u' and 'v' by subtracting the actuator coords xIF,yIF
    #    Then fill in 'wu' and 'wv' via spline interpolation
    # ------------------------------------------------------------
    # Equivalent to: u = bsxfun(@minus, u0', xIF);
    # We'll do broadcasting in Python:
    u = u0[:, None] - xIF[None, :]
    wu = np.zeros_like(u)

    limit_val = gaussian[-1, 0]  # "gaussian(end,1)" in MATLAB
    index_v = (u >= -limit_val) & (u <= limit_val)
    # Evaluate the spline at those points
    wu[index_v] = splev(u[index_v], splineP)

    # Now for v and wv
    v = u0[:, None] - yIF[None, :]
    wv = np.zeros_like(v)
    index_v2 = (v >= -limit_val) & (v <= limit_val)
    wv[index_v2] = splev(v[index_v2], splineP)

    # ------------------------------------------------------------
    # 6) Build the "m_modes" sparse matrix
    #    in MATLAB: spalloc(layersNPixel^2, nValid, nu*nv)
    # ------------------------------------------------------------
    nPix2 = layersNPixel[kDmLayer]*layersNPixel[kDmLayer]
    nValid = nValidActuatorsArr[kDmLayer]
    # We'll use a LIL matrix for efficient assignment
    m_modes = lil_matrix((nPix2, nValid), dtype=float)

    # "indIF = 1:nActuators(kDmLayer)^2; indIF(~validActuators{kDmLayer}) = [];"
    indIF = np.arange(nA*nA)  # 0-based
    valid_map = validActuators[kDmLayer].ravel()  # flatten the boolean
    indIF = indIF[valid_map]

    # "iIF,jIF = ind2sub(...)"
    iIF, jIF = np.unravel_index(indIF, (nA, nA))

    print(f' @(influenceFunction)> Computing the 2D DM zonal modes... ({nValid:4d},    ')

    # In MATLAB: "for kIF = 1:nValid"
    # Here we do 0-based: "for idx in range(nValid):"
    for idxIF in range(nValid):
        # wv[:, iIF[idxIF]] and wu[:, jIF[idxIF]]
        row_wv = wv[:, iIF[idxIF]]  # shape (layersNPixel,)
        col_wu = wu[:, jIF[idxIF]]  # shape (layersNPixel,)

        # "buffer = wv(:,kIF)*wu(:,kIF)' => outer product
        buffer_2d = np.outer(row_wv, col_wu)  # shape: (layersNPixel, layersNPixel)

        # Flatten in column-major or row-major?  
        # MATLAB "buffer(:)" is column-major by default. If we want to replicate exactly,
        # we might do buffer_2d.T.ravel() or specify order='F'.
        # But typically row-major is standard in Python.  
        # We'll replicate MATLAB’s column-major with order='F':
        m_modes[:, idxIF] = buffer_2d.ravel(order='F')

        # Print progress (like MATLAB's "\b\b\b\b%4d"):
        print(f'\b\b\b\b{idxIF+1:4d}', end='')
    print('')  # newline

    # Convert to a more standard CSR format
    m_modes = m_modes.tocsr()

    # In MATLAB: "modes{kDmLayer} = m_modes;"
    modes[kDmLayer] = m_modes

    # "F = 2*modes{kDmLayer};"
    F = 2.0 * m_modes

    # "dmInfFuncMatrix = blkdiag(dmInfFuncMatrix,F);"
    if dmInfFuncMatrix is None:
        # first time
        dmInfFuncMatrix = F
    else:
        dmInfFuncMatrix = block_diag((dmInfFuncMatrix, F))

    # "iF = pinv(full(F));"
    # toarray() converts CSR -> dense for pinv
    iF = pinv(F.toarray())
    iFCell[kDmLayer] = iF

    # "iDmInfFuncMatrix = blkdiag(iDmInfFuncMatrix,iF);"
    if iDmInfFuncMatrix is None:
        iDmInfFuncMatrix = csr_matrix(iF)
    else:
        iDmInfFuncMatrix = block_diag((iDmInfFuncMatrix, csr_matrix(iF)))


# %% --------------------------------------------------------------------------
# Hx and Ha
# --------------------------------------------------------------------------
outputWavefrontMask = validActuatorMap  # presumably a boolean 2D map
# In MATLAB: "[x,y] = meshgrid(linspace(-1,1,nLenslet+1)*D/2);"
x_grid, y_grid = np.meshgrid(
    np.linspace(-1, 1, nLenslet+1)*(D/2),
    np.linspace(-1, 1, nLenslet+1)*(D/2)
)
outputPhaseGrid = x_grid[outputWavefrontMask] + 1j*y_grid[outputWavefrontMask]

nStar = nFitSrc

# We create Python lists-of-lists for Hx and Ha
Hx = [[None for _ in range(nLayer)]  for _ in range(nStar)]
Ha = [[None for _ in range(nDmLayer)] for _ in range(nStar)]

intHa = [None]*nStar
intHx = [None]*nStar

for kGs in range(nStar):
    # ---- Build Hx ----
    for kAtmLayer in range(nLayer):
        pitchAtmLayer = dSub / overSampling[kAtmLayer]
        height        = altitude[kAtmLayer]
        # pupil center in layer
        beta  = directionVectorFitSrc[:, kGs]*height   # shape (3,) in MATLAB, but we only use x,y
        scale = 1 - height/fitSrcHeight

        # Prepare the (x,y) coords for interpolation
        xi_ = outputPhaseGrid.real * scale + beta[0]
        yi_ = outputPhaseGrid.imag * scale + beta[1]

        # atmGrid[kAtmLayer] might be (xarray, yarray)
        x_atm = atmGrid[kAtmLayer][0]
        y_atm = atmGrid[kAtmLayer][1]

        # p_bilinearSplineInterp(...) is your Python equivalent to "p_bilinearSplineInterp.m"
        Hx[kGs][kAtmLayer] = p_bilinearSplineInterp(
            x_atm, y_atm, pitchAtmLayer,
            xi_, yi_
        )
    
    # ---- Build Ha (the DM "propagator") ----
    for kdL in range(nDmLayer):
        pitchDmLayer = dSub
        height       = dmHeights[kdL]
        beta  = directionVectorFitSrc[:, kGs]*height
        scale = 1 - height/fitSrcHeight

        # We retrieve the actuator coordinate array
        actCoord = actCoordVector[kdL]  # from above

        dmin_ = np.min(actCoord.real)
        dmax_ = np.max(actCoord.real)
        Dx = (layersNPixel[kdL]-1)*pitchDmLayer
        sx = dmin_ - (Dx - (dmax_ - dmin_))/2

        # Equivalent to "[x,y] = meshgrid(linspace(0,1,layersNPixel)*Dx + sx);"
        x_lin = np.linspace(0, 1, layersNPixel[kdL]) * Dx + sx
        y_lin = x_lin  # same size
        x_dm, y_dm = np.meshgrid(x_lin, y_lin)

        # Interpolate
        xi_ = outputPhaseGrid.real * scale + beta[0]
        yi_ = outputPhaseGrid.imag * scale + beta[1]

        Ha[kGs][kdL] = p_bilinearSplineInterp(
            x_dm, y_dm, pitchDmLayer,
            xi_, yi_
        )

    # "intHa{kGs} = [Ha{kGs,:}]*dmInfFuncMatrix;"
    # In Python, we horizontally stack each layer's Ha and then multiply by dmInfFuncMatrix
    Ha_concat = csr_matrix(np.hstack(Ha[kGs]))
    intHa[kGs] = Ha_concat @ dmInfFuncMatrix

    # "intHx{kGs} = [Hx{kGs,:}];"
    # Similarly, for Hx we might want a single large matrix horizontally stacked:
    Hx_concat = csr_matrix(np.hstack(Hx[kGs]))
    intHx[kGs] = Hx_concat

    # (Optional) If you want to accumulate LeftMean and RightMean:
    # LeftMean += intHa[kGs].T @ intHa[kGs]
    # RightMean += intHa[kGs].T @ intHx[kGs]

# You could then do:
# fittingMatrix = pinv(full(LeftMean),1)*RightMean;
# or in Python:
# fitMat = pinv(LeftMean.toarray()) @ RightMean.toarray()

# Done!

toc = time.time()
print(f"Total run time = {toc - tic:.2f} seconds.")

#%%