import numpy as np
import math
from scipy.sparse import spdiags, coo_matrix, csr_matrix
from scipy.linalg import triu
from scipy.special import kv  # kv is the Bessel function of the second kind

def cart2pol(x, y):
    """
    Convert Cartesian coordinates (x, y) to polar (theta, rho).
    theta is the angle in radians, rho is the radial distance.
    """
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return theta, rho

def pol2cart(theta, rho):
    """
    Convert polar coordinates (theta, rho) to Cartesian (x, y).
    theta is the angle in radians, rho is the radial distance.
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def rotateDM(px,py, rotAngleInRadians):
    """
    This function rotate the DM actuators positions.

    Args:
        px (1D array): The original DM X actuator position.
        py (1D array): The original DM Y actuator position.
        rotAngleInRadians (double): The rotation angle in radians.

    Returns:
        pxx (1D array): The new DM X actuator position after rotation.
        pyy (1D array): The new DM Y actuator position after rotation.
    """
    pxx = px * math.cos(rotAngleInRadians) - py * math.sin(rotAngleInRadians)
    pyy= py * math.cos(rotAngleInRadians) + px * math.sin(rotAngleInRadians)
    return pxx, pyy

def make_biharm_operator(n, m):
    """
    Creates a discrete Laplacian operator L^2 using 5-point or 9-point stencils
    in a matrix of size (n*m).
    This replicates the approach in the MATLAB code, but with
    boundary conditions replaced with large weights.
    """
    N = n*m
    e = np.ones(N)

    # adjacency patterns for spdiags:
    # For a 2D grid stored column-major: 
    # -m, -1, 0, +1, +m are main lines in the sparse matrix.

    ex1 = e.copy()
    ex1[-m:] = 2
    ex1 = np.roll(ex1, -m)

    ex2 = e.copy()
    ex2[:m] = 2
    ex2 = np.roll(ex2, m)

    ey1 = e.copy()
    ey1[np.where(np.mod(np.arange(N), m)==0)] = 2
    ey1[np.where(np.mod(np.arange(N), m)==1)] = 0
    ey1 = np.roll(ey1, -1)

    ey2 = e.copy()
    ey2[np.where(np.mod(np.arange(N), m)==0)] = 0
    ey2[np.where(np.mod(np.arange(N), m)==1)] = 2
    ey2 = np.roll(ey2, 1)

    diag_data = np.column_stack([ex1, ey1, -4*e, ey2, ex2])
    diag_pos = np.array([-m, -1, 0, 1, m])
    p_L = spdiags(diag_data.T, diag_pos, N, N)
    p_L2 = p_L.transpose().dot(p_L)

    return p_L2

def sparseGradientMatrixAmplitudeWeighted(validLenslet, amplMask=None, overSampling=2):
    """
    Computes the sparse gradient matrix (3x3 stencil) with amplitude mask.
    
    Parameters
    ----------
    validLenslet : 2D array
        Valid lenslet map
    amplMask : 2D array
        Amplitudes Weight Mask (default=None). 
    overSampling : int
        Oversampling factor for the gridMask. Can be either 2 or 4 (default=2).

    Returns
    -------
    Gamma : scipy.sparse.csr_matrix
        Sparse gradient matrix of size.
    gridMask : 2D array
        Mask of size (overSampling*validLenslet.shape[0]+1) used for the reconstructed phase.
    """
    nLenslet = validLenslet.shape[0] # Number of lenselt across the pupil
    osFactor = overSampling 

    if amplMask is None:
        amplMask = np.ones((osFactor * nLenslet + 1, osFactor * nLenslet + 1))

    nMap = osFactor * nLenslet + 1
    nValidLenslet_ = np.count_nonzero(validLenslet)
    dsa = 1

    if osFactor == 2:
        i0x = np.tile(np.arange(1, 4), 3) # x stencil row subscript
        j0x = np.repeat(np.arange(1, 4), 3) # x stencil col subscript
        i0y = np.tile(np.arange(1, 4), 3) # y stencil row subscript
        j0y = np.repeat(np.arange(1, 4), 3) # y stencil col subscript
        s0x = np.array([-1/4, -1/2, -1/4, 0, 0, 0, 1/4, 1/2, 1/4]) * (1/dsa) # x stencil weight
        s0y = -np.array([1/4, 0, -1/4, 1/2, 0, -1/2, 1/4, 0, -1/4]) * (1/dsa) # y stencil weight
        Gv = np.array([[-2, 2, -1, 1], [-2, 2, -1, 1], [-1, 1, -2, 2], [-1, 1, -2, 2]])
        i_x = np.zeros(9 * nValidLenslet_)
        j_x = np.zeros(9 * nValidLenslet_)
        s_x = np.zeros(9 * nValidLenslet_)
        i_y = np.zeros(9 * nValidLenslet_)
        j_y = np.zeros(9 * nValidLenslet_)
        s_y = np.zeros(9 * nValidLenslet_)
        iMap0, jMap0 = np.meshgrid(np.arange(1, 4), np.arange(1, 4))
        gridMask = np.zeros((nMap, nMap), dtype=bool)
        u = np.arange(1, 10)
    elif osFactor == 4:
        i0x = np.tile(np.arange(1, 6), 5) # x stencil row subscript
        j0x = np.repeat(np.arange(1, 6), 5) # x stencil col subscript
        i0y = np.tile(np.arange(1, 6), 5) # y stencil row subscript
        j0y = np.repeat(np.arange(1, 6), 5) # y stencil col subscript
        s0x = np.array([-1/16, -3/16, -1/2, -3/16, -1/16, \
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/16, 3/16, 1/2, 3/16, 1/16]) * (1/dsa) # x stencil weight
        s0y = s0x.reshape(5,5).T.flatten() # y stencil weight
        i_x = np.zeros(25 * nValidLenslet_)
        j_x = np.zeros(25 * nValidLenslet_)
        s_x = np.zeros(25 * nValidLenslet_)
        i_y = np.zeros(25 * nValidLenslet_)
        j_y = np.zeros(25 * nValidLenslet_)
        s_y = np.zeros(25 * nValidLenslet_)
        iMap0, jMap0 = np.meshgrid(np.arange(1, 6), np.arange(1, 6))
        gridMask = np.zeros((nMap, nMap), dtype=bool)
        u = np.arange(1, 26)

    # Perform accumulation of x and y stencil row and col subscript and weight
    for jLenslet in range(1, nLenslet + 1):
        jOffset = osFactor * (jLenslet - 1)
        for iLenslet in range(1, nLenslet + 1):
            if validLenslet[iLenslet - 1, jLenslet - 1]:
                I = (iLenslet - 1) * osFactor + 1
                J = (jLenslet - 1) * osFactor + 1

                a = amplMask[I - 1:I + osFactor, J - 1:J + osFactor]
                numIllum = np.sum(a)

                if numIllum == (osFactor + 1) ** 2:
                    iOffset = osFactor * (iLenslet - 1)
                    i_x[u - 1] = i0x + iOffset
                    j_x[u - 1] = j0x + jOffset
                    s_x[u - 1] = s0x
                    i_y[u - 1] = i0y + iOffset
                    j_y[u - 1] = j0y + jOffset
                    s_y[u - 1] = s0y
                    u = u + (osFactor + 1) ** 2
                    gridMask[iMap0 + iOffset - 1, jMap0 + jOffset - 1] = True
                elif numIllum != (osFactor + 1) ** 2:
                    # Perform calculations for numIllum != (osFactor+1)**2
                    # ...
                    pass

    indx = np.ravel_multi_index((i_x.astype(int) - 1, j_x.astype(int) - 1), (nMap, nMap), order='F')
    indy = np.ravel_multi_index((i_y.astype(int) - 1, j_y.astype(int) - 1), (nMap, nMap), order='F')
    v = np.tile(np.arange(1, 2 * nValidLenslet_ + 1), (u.size, 1)).T
    Gamma = csr_matrix((np.concatenate((s_x, s_y)), (v.flatten() - 1, np.concatenate((indx, indy)))),
                       shape=(2 * nValidLenslet_, nMap ** 2))
    Gamma = Gamma[:, gridMask.ravel()]

    return Gamma, gridMask

def sparseGradientMatrix3x3Stencil(validLenslet):
    """
    Computes the sparse gradient matrix (3x3 stencil).
    
    Parameters
    ----------
    validLenslet : 2D array
        Valid lenslet map

    Returns
    -------
    Gamma : scipy.sparse.csr_matrix
        Sparse gradient matrix of size.
    gridMask : 2D array
        Mask of size (overSampling*validLenslet.shape[0]+1) used for the reconstructed phase.
    """
    nLenslet = validLenslet.shape[0] # Number of lenselt across the pupil
    nMap = 2 * nLenslet + 1
    nValidLenslet_ = np.count_nonzero(validLenslet)

    i0x = np.tile(np.arange(1, 4), 2) # x stencil row subscript
    j0x = np.repeat(np.arange(1, 4), 2) # x stencil col subscript
    i0y = np.tile(np.arange(1, 4), 2) # y stencil row subscript
    j0y = np.repeat(np.arange(1, 4), 2) # y stencil col subscript
    s0x = np.array([-1/2, -1, -1/2, 1/2, 1, 1/2]) # x stencil weight
    s0y = -np.array([1/2, -1/2, 1, -1, 1/2, -1/2]) # y stencil weight
    i_x = np.zeros(6 * nValidLenslet_)
    j_x = np.zeros(6 * nValidLenslet_)
    s_x = np.zeros(6 * nValidLenslet_)
    i_y = np.zeros(6 * nValidLenslet_)
    j_y = np.zeros(6 * nValidLenslet_)
    s_y = np.zeros(6 * nValidLenslet_)
    iMap0, jMap0 = np.meshgrid(np.arange(1, 4), np.arange(1, 4))
    gridMask = np.zeros((nMap, nMap), dtype=bool)
    u = np.arange(1, 7)

    # Perform accumulation of x and y stencil row and col subscript and weight
    for jLenslet in range(1, nLenslet + 1):
        jOffset = 2 * (jLenslet - 1)
        for iLenslet in range(1, nLenslet + 1):
            if validLenslet[iLenslet - 1, jLenslet - 1]:
                    iOffset = 2 * (iLenslet - 1)
                    i_x[u - 1] = i0x + iOffset
                    j_x[u - 1] = j0x + jOffset
                    s_x[u - 1] = s0x
                    i_y[u - 1] = i0y + iOffset
                    j_y[u - 1] = j0y + jOffset
                    s_y[u - 1] = s0y
                    u = u + 6
                    gridMask[iMap0 + iOffset - 1, jMap0 + jOffset - 1] = True

    indx = np.ravel_multi_index((i_x.astype(int) - 1, j_x.astype(int) - 1), (nMap, nMap), order='F')
    indy = np.ravel_multi_index((i_y.astype(int) - 1, j_y.astype(int) - 1), (nMap, nMap), order='F')
    v = np.tile(np.arange(1, 2 * nValidLenslet_ + 1), (u.size, 1)).T
    Gamma = csr_matrix((np.concatenate((s_x, s_y)), (v.flatten() - 1, np.concatenate((indx, indy)))),
                       shape=(2 * nValidLenslet_, nMap ** 2))
    Gamma = Gamma[:, gridMask.ravel()]

    return Gamma, gridMask



def p_bilinearSplineInterp(xo, yo, do, xi, yi):
    """
    Sparse bilinear interpolation from a regular grid (xo, yo) with spacing 'do'
    to query points (xi, yi).

    Parameters
    ----------
    xo, yo : 1D arrays
        The original grid in x and y.  They define a 2D grid of shape (len(yo), len(xo))
        in row-major style if we do e.g. y for rows, x for columns.
    do : float
        The spacing between grid points (assumed uniform).
    xi, yi : 1D arrays
        Coordinates of the interpolation points.

    Returns
    -------
    P : scipy.sparse.csr_matrix
        A sparse matrix of shape (len(xi), len(xo)*len(yo)),
        such that for a field F (flattened as row-major),
        we can approximate F_i = P * F.
    """
    ni = len(xi)
    nxo = len(xo)
    nyo = len(yo)
    no = nxo * nyo

    mask = (xi >= xo[0]) & (yi >= yo[0]) & (xi <= xo[-1]) & (yi <= yo[-1])
    index = np.flatnonzero(mask)
    xi_masked = xi[mask]
    yi_masked = yi[mask]

    ox = np.floor((xi_masked - xo[0]) / do).astype(int) + 1
    oy = np.floor((yi_masked - yo[0]) / do).astype(int) + 1

    fxo = np.abs(xi_masked - (xo[0] + do * (ox - 1))) / do
    fyo = np.abs(yi_masked - (yo[0] + do * (oy - 1))) / do

    s1 = (1 - fxo) * (1 - fyo)
    s2 = fxo * (1 - fyo)
    s3 = (1 - fxo) * fyo
    s4 = fxo * fyo

    ox_python = ox - 1
    oy_python = oy - 1

    o1 = ox_python * nyo + oy_python
    o2 = (ox_python + 1) * nyo + oy_python
    o3 = ox_python * nyo + (oy_python + 1)
    o4 = (ox_python + 1) * nyo + (oy_python + 1)

    masks = [s1 != 0, s2 != 0, s3 != 0, s4 != 0]
    os = [o1, o2, o3, o4]
    ss = [s1, s2, s3, s4]

    rows, cols, data = [], [], []
    for i in range(4):
        mask_i = masks[i]
        s = ss[i][mask_i]
        o = os[i][mask_i]
        idx = index[mask_i]
        rows.append(idx)
        cols.append(o)
        data.append(s)

    if rows:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
    else:
        rows = np.array([], dtype=int)
        cols = np.array([], dtype=int)
        data = np.array([], dtype=float)

    P = coo_matrix((data, (rows, cols)), shape=(ni, no))
    return P.tocsr()

def create_atm_grid(directionVectorLGS, directionVectorFitSrc, nLayer, dSub, altitude, LGSheight, D):
    """
    Create an atmospheric grid for each layer based on the given parameters.

    Parameters:
    - directionVectorLGS: Direction vector for the LGS (Laser Guide Star).
    - directionVectorFitSrc: Direction vector for the fitting sources.
    - nLayer: Number of atmospheric layers.
    - dSub: Sub-aperture size.
    - altitude: Array of altitudes for each layer.
    - LGSheight: Height of the Laser Guide Star.
    - D: Diameter of the telescope.

    Returns:
    - atmGrid: List of tuples containing the x and y grids for each layer.
    """
    
    # Extract the x, y components from the direction vectors
    vg = directionVectorLGS[0:2, :]    # Just the x, y components for LGS
    vs = directionVectorFitSrc[0:2, :] # x, y components for fitting sources

    atmGrid = []  # Initialize the atmospheric grid list
    overSampling = np.ones(nLayer) * 2  # Oversampling factor for each layer

    for kLayer in range(nLayer):
        pitchLayer = dSub / overSampling[kLayer]  # Pitch for the current layer
        height = altitude[kLayer]  # Altitude of the current layer
        mVal = 1 - height / LGSheight  # Scaling factor based on height

        # Calculate the displacement vectors for LGS and fitting sources
        dDirecG = vg * height
        dDirecS = vs * height

        # Determine the min and max bounds for the grid
        dmin = np.min(np.concatenate([dDirecG - D/2*mVal, dDirecS - D/2], axis=1), axis=1)
        dmax = np.max(np.concatenate([dDirecG + D/2*mVal, dDirecS + D/2], axis=1), axis=1)

        # Calculate the number of pixels in the x and y directions
        nPxLayerX = int(math.floor((dmax[0] - dmin[0]) / pitchLayer)) + 2
        nPxLayerY = int(math.floor((dmax[1] - dmin[1]) / pitchLayer)) + 2

        # Calculate the total size of the grid in x and y directions
        Dx = (nPxLayerX - 1) * pitchLayer
        Dy = (nPxLayerY - 1) * pitchLayer

        # Calculate the starting points for the grid
        sx = dmin[0] - (Dx - (dmax[0] - dmin[0])) / 2
        sy = dmin[1] - (Dy - (dmax[1] - dmin[1])) / 2

        # Create the x and y grids
        x_grid = np.linspace(0, 1, nPxLayerX) * Dx + sx
        y_grid = np.linspace(0, 1, nPxLayerY) * Dy + sy

        # Append the grid for the current layer to the atmGrid list
        atmGrid.append((x_grid, y_grid))

    return atmGrid

def auto_correlation(sampling, D, wfs_lenslets_rotation, wfs_lenslets_offset, \
                    mask, nGs, srcACdirectionVector, srcACheight, nLayer, altitude, \
                    fractionnalR0,r0,L0):
    """
    Python implementation of the MATLAB autoCorrelation function.
    
    Parameters:
    -----------
    sampling : int
        Sampling rate for the grid.
    range : float
        Range of the grid.
    wfs : object
        Wavefront sensor object with lenslets' rotation and offset attributes.
    mask : ndarray
        Mask for filtering the covariance matrix.
    nGs : int
        Number of guide stars.
    srcACdirectionVector : ndarray
        Direction vector for the guide stars.
    srcACheight : ndarray
        Heights of the guide stars.
    nLayer : int
        Number of atmospheric layers.
    altitude : ndarray
        Altitudes of the atmospheric layers.
    fr0 : float
        Reference frequency.
    L0 : float
        Outer scale of turbulence.

    
    Returns:
    --------
    S : ndarray
        Auto-correlation meta-matrix.
    """
    
    print("-->> Auto-correlation meta-matrix!\n")
    
    # Generate indices for the upper triangular part of the matrix
    kGs = np.triu(np.arange(1, nGs**2 + 1).reshape(nGs, nGs).T, 1).T.reshape(nGs**2)
    kGs[0] = 1
    kGs = kGs[kGs != 0]
    
    # Initialize a list of zero matrices based on the mask
    S = [np.zeros((np.sum(mask),np.sum(mask))) for _ in range(len(kGs))]
    
    for k in range(len(kGs)):
        # Get the indices iGs and jGs from the index kGs(k)
        jGs, iGs = np.unravel_index(kGs[k] - 1, (nGs, nGs))  # Adjust for 0-based index in Python
        
        buf = 0
        
        # Create a grid for the first guide star
        x1, y1 = np.meshgrid(np.linspace(-1, 1, sampling) * D/2,
                             np.linspace(-1, 1, sampling) * D/2)
        x1, y1 = rotateDM(x1.flatten(), y1.flatten(), wfs_lenslets_rotation[iGs] * 180/np.pi)
        x1 = x1 - wfs_lenslets_offset[0, iGs] * D
        y1 = y1 - wfs_lenslets_offset[1, iGs] * D
        x1 = x1.reshape(sampling, sampling)
        y1 = y1.reshape(sampling, sampling)
        
        # Create a grid for the second guide star
        x2, y2 = np.meshgrid(np.linspace(-1, 1, sampling) * D/2,
                             np.linspace(-1, 1, sampling) * D/2)
        x2, y2 = rotateDM(x2.flatten(), y2.flatten(), wfs_lenslets_rotation[jGs] * 180/np.pi)
        x2 = x2 - wfs_lenslets_offset[0, jGs] * D
        y2 = y2 - wfs_lenslets_offset[1, jGs] * D
        x2 = x2.reshape(sampling, sampling)
        y2 = y2.reshape(sampling, sampling)
        
        for kLayer in range(nLayer):
            # Calculate the scaled and shifted coordinates for the first guide star
            beta = srcACdirectionVector[:, iGs] * altitude[kLayer]
            scale = 1 - altitude[kLayer] / srcACheight
            iZ = x1 * scale + beta[0] + 1j * (y1 * scale + beta[1])
            
            # Calculate the scaled and shifted coordinates for the second guide star
            beta = srcACdirectionVector[:, jGs] * altitude[kLayer]
            scale = 1 - altitude[kLayer] / srcACheight
            jZ = x2 * scale + beta[0] + 1j * (y2 * scale + beta[1])
            
            # Compute the covariance matrix
            #out = covariance_matrix(iZ, jZ, slab(atm, kLayer))
            out = covariance_matrix(iZ, jZ, r0, L0, fractionnalR0[kLayer])
            #out[~mask, :] = 0
            #out[:, ~mask] = 0
            out = out[mask.flatten(),:]
            out = out[:,mask.flatten()]
            # Accumulate the results
            buf += out
        
        S[k] = buf
    
    # Rearrange the results into a full nGs x nGs matrix
    buf = S
    S = [np.zeros(np.sum(mask)) for _ in range(nGs**2)]
    for idx, val in zip(kGs, buf):
        S[idx - 1] = val  # Adjust for 0-based index in Python
    S = np.array(S).reshape(nGs, nGs)
    
    # Fill the diagonal with the first element
    np.fill_diagonal(S, S[0, 0])
    
    # Make the matrix symmetric
    S = triu(S, 1) + triu(S).T
    
    return S

def covariance_matrix(*args):
    """
    COVARIANCEMATRIX Phase covariance matrix

    Computes the phase auto-covariance matrix from the vector rho1 and r0, L0, fractionnalR0
    or the phase cross-covariance matrix from the vectors rho1 and rho2 and r0, L0, fractionnalR0.

    Parameters:
    -----------
    *args : tuple
        - If four arguments are provided: rho1 (complex array), r0, L0, fractionnalR0
        - If five arguments are provided: rho1 (complex array), 
        rho2 (complex array), r0, L0, fractionnalR0

    Returns:
    --------
    out : numpy.ndarray
        The covariance matrix

    Example:
    --------
    # covariance matrix on a 1 metre square grid sampled on 16 pixels
    x, y = np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 16))
    rho1 = x + 1j * y
    g = covariance_matrix(rho1, r0, L0, fractionnalR0)
    plt.imshow(g)
    plt.colorbar()
    plt.show()

    # covariance matrix on a 1 metre square grid sampled on 16 pixels with the 
    # same grid but displaced of 1 metre
    x, y = np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 16))
    rho1 = x + 1j * y
    rho2 = rho1 + 1
    g = covariance_matrix(rho1, rho2, r0, L0, fractionnalR0)
    plt.imshow(g)
    plt.colorbar()
    plt.show()
    """
    
    if len(args) < 4 or len(args) > 5:
        raise ValueError("Number of arguments must be 4 or 5")
    
    rho1 = args[0].flatten()
    if len(args) == 4:
        rho = np.abs(rho1[:, np.newaxis] - rho1)
        r0 = args[1]
        L0 = args[2]
        fractionnalR0 = args[3]
    else:
        rho2 = args[1].flatten()
        rho = np.abs(rho1[:, np.newaxis] - rho2)
        r0 = args[2]
        L0 = args[3]
        fractionnalR0 = args[4]
    
    nRho, mRho = rho.shape
    blockSize = 5000
    
    if max(nRho, mRho) > blockSize:  # Memory gentle
        l = nRho // blockSize
        #le = nRho - l * blockSize
        p = mRho // blockSize
        #pe = mRho - p * blockSize
        
        print(f" @(covariance_matrix)> Memory gentle! ({(l+1)}X{(p+1)} blocks)")
        
        # Split the matrix into blocks
        rho_blocks = [rho[i * blockSize:(i + 1) * blockSize, j * blockSize:(j + 1) * blockSize] \
            for i in range(l + 1) for j in range(p + 1)]
        
        out_blocks = []
        for block in rho_blocks:
            out_blocks.append(_compute_covariance(block, r0, L0, fractionnalR0))
        
        # Reconstruct the full matrix from blocks
        out = np.block([[out_blocks[i * (p + 1) + j] for j in range(p + 1)] for i in range(l + 1)])
        
    else:  # Memory intensive
        out = _compute_covariance(rho, r0, L0, fractionnalR0)
    
    return out


def _compute_covariance(rho, r0, L0, fractionnalR0):
    """
    Helper function to compute the covariance matrix for a given rho and 
    atmosphere parameters: r0, L0 and fractionnal r0.

    Parameters:
    -----------
    rho : numpy.ndarray
        The distance matrix
    r0 : float
        The atmosphere coherence length (internal scale)
    L0 : float
        The atmosphere coherence length (external scale)
    fractionnalR0: numpy.ndarray
        The atmosphere fractionnal r0 profile (Cn2 profile)

    Returns:
    --------
    out : numpy.ndarray
        The covariance matrix
    """
    
    L0r0ratio = (L0/r0)**(5./3.)
    cst = (24.*math.gamma(6./5.)/5)**(5./6.)* \
        (math.gamma(11./6.)/(2.**(5./6.)*np.pi**(8./3.)))*L0r0ratio
    
    out = np.ones_like(rho)*(24.*math.gamma(6./5.)/5.)**(5./6.)* \
        (math.gamma(11./6.)*math.gamma(5./6.)/(2*np.pi**(8./3.)))*L0r0ratio
    
    index = rho != 0
    u = 2*np.pi*rho[index]/L0
    out[index] = cst*u**(5./6.)*kv(5./6., u)
    
    out = np.sum(fractionnalR0) * out
    
    return out