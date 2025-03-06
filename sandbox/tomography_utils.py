import numpy as np
import math
from scipy.sparse import spdiags, coo_matrix, csr_matrix
from scipy.special import kv, gamma  # kv is the Bessel function of the second kind
from numpy import triu, tril

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

def create_guide_star_grid(sampling, D, rotation_angle, offset_x, offset_y):
    # Create a grid
    x, y = np.meshgrid(np.linspace(-1, 1, sampling) * D/2,
                        np.linspace(-1, 1, sampling) * D/2)
    
    # Flatten, rotate, and apply offset
    x, y = rotateDM(x.flatten(), y.flatten(), rotation_angle * 180/np.pi)
    x = x - offset_x * D
    y = y - offset_y * D
    
    # Reshape back to the original grid
    return x.reshape(sampling, sampling), y.reshape(sampling, sampling)

def calculate_scaled_shifted_coords(x, y, srcACdirectionVector, gs_index, 
                                    altitude, kLayer, srcACheight):
    beta = srcACdirectionVector[:, gs_index] * altitude[kLayer]
    scale = 1 - altitude[kLayer] / srcACheight
    return x * scale + beta[0] + 1j * (y * scale + beta[1])


def pinv_matlab(A):
    """
    Compute the pseudoinverse of matrix A similar to MATLAB's pinv.
    
    Parameters
    ----------
    A : array_like, shape (M, N)
        Input matrix.
    
    Returns
    -------
    X : ndarray, shape (N, M)
        Pseudoinverse of A.
    """
    # Compute the Singular Value Decomposition of A
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Calculate tolerance similar to MATLAB's implementation:
    # tol = max(size(A)) * max(s) * eps
    tol = max(A.shape) * np.max(s) * np.finfo(s.dtype).eps
    
    # Invert singular values greater than tolerance, set others to zero.
    s_inv = np.array([1/si if si > tol else 0 for si in s])
    
    # Reconstruct the pseudoinverse
    return Vt.T @ np.diag(s_inv) @ U.T

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
    Computes the sparse gradient matrix (3x3 or 5x5 stencil) with amplitude mask.
    
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
        Sparse gradient matrix.
    gridMask : 2D array
        Mask used for the reconstructed phase.
    """
    print("-->> Computing sparse gradient matrix <<--\n")
    
    # Get dimensions and counts
    nLenslet = validLenslet.shape[0]  # Size of lenslet array
    nMap = overSampling * nLenslet + 1  # Size of oversampled grid
    nValidLenslet_ = np.count_nonzero(validLenslet)  # Number of valid lenslets
    
    # Create default amplitude mask if none provided
    if amplMask is None:
        amplMask = np.ones((nMap, nMap))

    # Set up stencil parameters based on oversampling factor
    if overSampling == 2:
        # 3x3 stencil for 2x oversampling
        stencil_size = 3
        s0x = np.array([-1/4, -1/2, -1/4, 0, 0, 0, 1/4, 1/2, 1/4])  # x-gradient weights
        s0y = -np.array([1/4, 0, -1/4, 1/2, 0, -1/2, 1/4, 0, -1/4])  # y-gradient weights
        num_points = 9
    elif overSampling == 4:
        # 5x5 stencil for 4x oversampling
        stencil_size = 5
        s0x = np.array([-1/16, -3/16, -1/2, -3/16, -1/16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                        0, 0, 0, 0, 0, 1/16, 3/16, 1/2, 3/16, 1/16])  # x-gradient weights
        s0y = s0x.reshape(5,5).T.flatten()  # y-gradient weights (transpose of x)
        num_points = 25
    else:
        raise ValueError("overSampling must be 2 or 4")

    # Initialize stencil position arrays
    i0x = np.tile(np.arange(1, stencil_size+1), stencil_size)  # Row indices
    j0x = np.repeat(np.arange(1, stencil_size+1), stencil_size)  # Column indices
    i0y = i0x.copy()  # Same pattern for y-gradient
    j0y = j0x.copy()
    
    # Initialize arrays to store sparse matrix entries
    i_x = np.zeros(num_points * nValidLenslet_)  # Row indices for x-gradient
    j_x = np.zeros(num_points * nValidLenslet_)  # Column indices for x-gradient
    s_x = np.zeros(num_points * nValidLenslet_)  # Values for x-gradient
    i_y = np.zeros(num_points * nValidLenslet_)  # Row indices for y-gradient
    j_y = np.zeros(num_points * nValidLenslet_)  # Column indices for y-gradient
    s_y = np.zeros(num_points * nValidLenslet_)  # Values for y-gradient
    
    # Create grid for mask
    iMap0, jMap0 = np.meshgrid(np.arange(1, stencil_size+1), np.arange(1, stencil_size+1))
    gridMask = np.zeros((nMap, nMap), dtype=bool)
    u = np.arange(1, num_points+1)  # Counter for filling arrays

    # Build sparse matrix by iterating over lenslets
    for jLenslet in range(1, nLenslet + 1):
        jOffset = overSampling * (jLenslet - 1)  # Column offset in oversampled grid
        for iLenslet in range(1, nLenslet + 1):
            if validLenslet[iLenslet - 1, jLenslet - 1]:  # Only process valid lenslets
                # Calculate indices in amplitude mask
                I = (iLenslet - 1) * overSampling + 1
                J = (jLenslet - 1) * overSampling + 1
                
                # Check if amplitude mask is valid for this lenslet
                if np.sum(amplMask[I-1:I+overSampling, J-1:J+overSampling]) == (overSampling + 1) ** 2:
                    iOffset = overSampling * (iLenslet - 1)  # Row offset in oversampled grid
                    # Fill in gradient arrays
                    i_x[u - 1] = i0x + iOffset
                    j_x[u - 1] = j0x + jOffset
                    s_x[u - 1] = s0x
                    i_y[u - 1] = i0y + iOffset
                    j_y[u - 1] = j0y + jOffset
                    s_y[u - 1] = s0y
                    u = u + num_points
                    gridMask[iMap0 + iOffset - 1, jMap0 + jOffset - 1] = True

    # Create sparse matrix in CSR format
    # Convert indices to linear indices
    indx = np.ravel_multi_index((i_x.astype(int) - 1, j_x.astype(int) - 1), (nMap, nMap), order='F')
    indy = np.ravel_multi_index((i_y.astype(int) - 1, j_y.astype(int) - 1), (nMap, nMap), order='F')
    v = np.tile(np.arange(1, 2 * nValidLenslet_ + 1), (u.size, 1)).T
    
    # Construct final sparse gradient matrix
    Gamma = csr_matrix((np.concatenate((s_x, s_y)), (v.flatten() - 1, np.concatenate((indx, indy)))),
                       shape=(2 * nValidLenslet_, nMap ** 2))
    Gamma = Gamma[:, gridMask.ravel()]  # Apply mask to reduce matrix size

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

def auto_correlation(tomoParams,lgsWfsParams, atmParams,lgsAsterismParams,gridMask):
    """
    Computes the auto-correlation meta-matrix for tomographic atmospheric reconstruction.
    
    Parameters:
    -----------
    tomoParams : object
        Contains tomography parameters:
        - sampling (int): Number of grid samples per axis
        - mask (ndarray): 2D boolean grid mask
    
    lgsWfsParams : object
        LGS WFS parameters:
        - D (float): Telescope diameter [m]
        - wfs_lenslets_rotation (ndarray): Lenslet rotations [rad]
        - wfs_lenslets_offset (ndarray): Lenslet offsets [normalized]
    
    atmParams : object
        Atmospheric parameters:
        - nLayer (int): Number of turbulence layers
        - altitude (ndarray): Layer altitudes [m]
        - r0 (float): Fried parameter [m]
        - L0 (float): Outer scale [m]
        - fractionnalR0 (ndarray): Turbulence strength per layer
    
    lgsAsterismParams : object
        LGS constellation parameters:
        - nLGS (int): Number of LGS
        - directionVectorLGS (ndarray): Direction vectors
        - LGSheight (ndarray): LGS heights [m]

    gridMask : ndarray
        2D boolean mask for valid grid points

    Returns:
    --------
    S : ndarray
        Auto-correlation meta-matrix of shape (nGs*valid_pts, nGs*valid_pts)
    """
    
    print("-->> Computing auto-correlation meta-matrix <<--\n")
    # ======================================================================
    # Parameter Extraction
    # ======================================================================
    # Tomography parameters
    sampling = tomoParams.sampling
    mask = gridMask
    
    # LGS constellation parameters
    nGs = lgsAsterismParams.nLGS
    srcACdirectionVector = lgsAsterismParams.directionVectorLGS
    srcACheight  = lgsAsterismParams.LGSheight
    
    # WFS parameters
    D = lgsWfsParams.DSupport  
    wfsLensletsRotation = lgsWfsParams.wfsLensletsRotation
    wfs_lenslets_offset = lgsWfsParams.wfs_lenslets_offset
    
    # Atmospheric parameters
    nLayer = atmParams.nLayer
    altitude = atmParams.altitude
    r0 = atmParams.r0
    L0 = atmParams.L0
    fractionnalR0 = atmParams.fractionnalR0
    
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
        
        # Create grids for the first and second guide stars
        x1, y1 = create_guide_star_grid(sampling, D, wfsLensletsRotation[iGs], 
                                        wfs_lenslets_offset[0, iGs], wfs_lenslets_offset[1, iGs])
        x2, y2 = create_guide_star_grid(sampling, D, wfsLensletsRotation[jGs], 
                                        wfs_lenslets_offset[0, jGs], wfs_lenslets_offset[1, jGs])
        
        for kLayer in range(nLayer):
            # Calculate the scaled and shifted coordinates for the first and second guide stars
            iZ = calculate_scaled_shifted_coords(x1, y1, srcACdirectionVector, iGs, altitude, kLayer, srcACheight)
            jZ = calculate_scaled_shifted_coords(x2, y2, srcACdirectionVector, jGs, altitude, kLayer, srcACheight)
            
            # Compute the covariance matrix
            out = covariance_matrix(iZ.T, jZ.T, r0, L0, fractionnalR0[kLayer])
            out = out[mask.flatten(),:]
            out = out[:,mask.flatten()]
            # Accumulate the results
            buf += out
        
        S[k] = buf.T
    
    # Rearrange the results into a full nGs x nGs matrix
    buf = S
    S_tmp = [np.zeros((np.sum(mask), np.sum(mask))) for _ in range(nGs**2)]
    for c, i in enumerate(kGs):
        S_tmp[i-1] = buf[c]
    
    # index = [5, 10, 15]
    # diagonal_indices = np.diag_indices(nGs)
    # If you want these as a 1D array of indices    
    diagonal_indices_1d = np.diag_indices(nGs)[0] * nGs + np.diag_indices(nGs)[1]
    
    for i in diagonal_indices_1d:
        S_tmp[i] = S_tmp[0]   
    
    S_tmp = np.stack(S_tmp, axis=0)
    S = S_tmp.reshape(nGs, nGs, np.sum(mask), np.sum(mask))\
        .transpose(0, 2, 1, 3).reshape(nGs*np.sum(mask), nGs*np.sum(mask))
        
    # Make the matrix symmetric
    S = tril(S) + triu(S.T, 1)
    
    return S

def covariance_matrix(*args):
    """
    Optimized phase covariance matrix calculation using Von Karman turbulence model
    
    Parameters:
        *args: (rho1, [rho2], r0, L0, fractionalR0)
            rho1, rho2: Complex coordinate arrays (x + iy)
            r0: Fried parameter (m)
            L0: Outer scale (m)
            fractionalR0: Turbulence layer weighting factor
    
    Returns:
        Covariance matrix with same dimensions as input coordinates
    """
    
    # Validate input arguments
    if len(args) not in {4, 5}:
        raise ValueError("Expected 4 or 5 arguments: (rho1, [rho2], r0, L0, fractionalR0)")
    
    # Parse arguments and flatten coordinates
    rho1 = args[0].flatten()
    auto_covariance = len(args) == 4
    if auto_covariance:
        r0, L0, fractionalR0 = args[1:]
        rho2 = rho1
    else:
        rho2, r0, L0, fractionalR0 = args[1], args[2], args[3], args[4]
        rho2 = rho2.flatten()

    # ==================================================================
    # Precompute constants (critical performance improvement)
    # ==================================================================
    # Gamma function values precomputed for numerical stability
    GAMMA_6_5 = gamma(6/5)
    GAMMA_11_6 = gamma(11/6)
    GAMMA_5_6 = gamma(5/6)
    
    # Base constant components
    BASE_CONST = (24 * GAMMA_6_5 / 5) ** (5/6)
    SCALE_FACTOR = (GAMMA_11_6 / (2**(5/6) * np.pi**(8/3)))
    
    # L0/r0 ratio raised to 5/3 power
    L0_r0_ratio = (L0 / r0) ** (5/3)
    
    # Final constant for non-zero distances
    cst = BASE_CONST * SCALE_FACTOR * L0_r0_ratio
    
    # Variance term for zero distances (r=0 case)
    var_term = (BASE_CONST * GAMMA_11_6 * GAMMA_5_6 / 
               (2 * np.pi**(8/3))) * L0_r0_ratio

    # ==================================================================
    # Calculate pairwise distances
    # ==================================================================
    # Vectorized distance calculation using broadcasting
    rho = np.abs(rho1[:, np.newaxis] - rho2)
    n, m = rho.shape

    # ==================================================================
    # Block processing for large matrices (>5000 elements per dimension)
    # ==================================================================
    block_size = 5000
    if max(n, m) > block_size:
        # Preallocate output array for memory efficiency
        out = np.empty((n, m), dtype=np.float64)
        
        # Process row blocks
        for i in range(0, n, block_size):
            i_end = min(i + block_size, n)
            
            # Process column blocks
            for j in range(0, m, block_size):
                j_end = min(j + block_size, m)
                
                # Process current block
                block = rho[i:i_end, j:j_end]
                out[i:i_end, j:j_end] = _compute_block(
                    block, L0, cst, var_term
                )
        
        # Apply fractional weighting
        out *= fractionalR0
        return out

    # Single block processing for smaller matrices
    out = _compute_block(rho, L0, cst, var_term)
    return out * fractionalR0

import numpy as np
import numba as nb

# Precomputed Gamma function values for v=5/6
gamma_1_6 = 5.56631600178  # Gamma(1/6)
gamma_11_6 = 0.94065585824  # Gamma(11/6)

@nb.njit(nb.complex128(nb.complex128), cache=True)
def _kv56_scalar(z):
    """Scalar implementation used as kernel for array version"""
    v = 5.0 / 6.0
    z_abs = np.abs(z)
    
    if z_abs < 2.0:
        # Series expansion for small |z|
        sum_a = 0.0j
        sum_b = 0.0j
        
        term_a = (0.5 * z)**v / gamma_11_6
        term_b = (0.5 * z)**-v / gamma_1_6
        sum_a += term_a
        sum_b += term_b
        
        z_sq_over_4 = (0.5 * z)**2
        k = 1
        tol = 1e-15
        max_iter = 1000
        
        for _ in range(max_iter):
            factor_a = z_sq_over_4 / (k * (k + v))
            term_a *= factor_a
            sum_a += term_a
            
            factor_b = z_sq_over_4 / (k * (k - v))
            term_b *= factor_b
            sum_b += term_b
            
            if abs(term_a) < tol * abs(sum_a) and abs(term_b) < tol * abs(sum_b):
                break
            k += 1
        
        K = np.pi * (sum_b - sum_a)
    else:
        # Asymptotic expansion for large |z|
        z_inv = 1.0 / z
        sum_terms = 1.0 + (2.0/9.0)*z_inv + (-7.0/81.0)*z_inv**2 + \
                    (175.0/2187.0)*z_inv**3 + (-2275.0/19683.0)*z_inv**4 + \
                    (5005.0/177147.0)*z_inv**5 #+ (-2662660.0/4782969.0)*z_inv**6
        prefactor = np.sqrt(np.pi/(2.0*z)) * np.exp(-z)
        K = prefactor * sum_terms
    
    return K

# Vectorized version with parallel execution
@nb.vectorize([nb.complex128(nb.complex128),  # Complex input
                nb.complex128(nb.float64)],    # Real input
                nopython=True, target='parallel')
def kv56(z):
    """
    Modified Bessel function K_{5/6}(z) for numpy arrays
    Handles both real and complex inputs efficiently
    """
    return _kv56_scalar(z)

def _compute_block(rho_block, L0, cst, var_term):
    """
    Vectorized computation of covariance values for a matrix block
    """
    # Initialize output with variance term
    out = np.full(rho_block.shape, var_term, dtype=np.float64)
    
    # Find non-zero distances and compute covariance
    mask = rho_block != 0
    u = (2 * np.pi * rho_block[mask]) / L0
    
    # Vectorized Bessel function calculation
    # u = np.round(u,2)
    out[mask] = cst * u**(5/6) * kv56(u.astype(np.complex128)) # kv56(u.astype(np.complex128)) # kv(5/6, u)
    
    return out

#import cupy as cp
# CuPy kernel for real-valued K_{5/6}
# kv56_gpu_kernel = cp.ElementwiseKernel(
#     'float64 z',
#     'float64 K',
#     '''
#     double v = 5.0 / 6.0;
#     double z_abs = fabs(z);
#     if (z_abs < 2.0) {
#         double sum_a = 0.0;
#         double sum_b = 0.0;
#         double term_a = pow(0.5 * z, v) / gamma_11_6;
#         double term_b = pow(0.5 * z, -v) / gamma_1_6;
#         sum_a = term_a;
#         sum_b = term_b;
#         double z_sq_over_4 = pow(0.5 * z, 2);
#         int k = 1;
#         double tol = 1e-15;
#         int max_iter = 1000;
#         for (int i = 0; i < max_iter; ++i) {
#             double factor_a = z_sq_over_4 / (k * (k + v));
#             term_a *= factor_a;
#             sum_a += term_a;
#             double factor_b = z_sq_over_4 / (k * (k - v));
#             term_b *= factor_b;
#             sum_b += term_b;
#             if (fabs(term_a) < tol * fabs(sum_a) && fabs(term_b) < tol * fabs(sum_b)) {
#                 break;
#             }
#             k += 1;
#         }
#         K = M_PI * (sum_b - sum_a);
#     } else {
#         double z_inv = 1.0 / z;
#         double sum_terms = 1.0 + z_inv * (2.0/9.0 + z_inv * (
#                     -7.0/81.0 + z_inv * (175.0/2187.0 + z_inv * (-980.0/6561.0)))); # TO DO: update corrected terms up to power 5 
#         double prefactor = sqrt(M_PI / (2.0 * z)) * exp(-z);
#         K = prefactor * sum_terms;
#     }
#     ''',
#     name='kv56_gpu_kernel',
#     preamble=f'''
#     const double gamma_1_6 = {gamma_1_6};
#     const double gamma_11_6 = {gamma_11_6};
#     '''
# )

def _compute_block_gpu(rho_block, L0, cst, var_term):
    rho_block_gpu = cp.asarray(rho_block)
    out_gpu = cp.full(rho_block_gpu.shape, var_term, dtype=np.float64)
    mask = rho_block_gpu != 0
    u = (2 * cp.pi * rho_block_gpu[mask]) / L0
    if u.size == 0:
        return out_gpu.get()
    K_gpu = kv56_gpu_kernel(u)
    out_gpu[mask] = cst * cp.power(u, 5.0/6.0) * K_gpu
    return out_gpu.get()

def cross_correlation(tomoParams,lgsWfsParams, atmParams,lgsAsterismParams,gridMask=None):
    """
    Computes the cross-correlation meta-matrix for tomographic atmospheric reconstruction.
    
    Parameters:
    -----------
    tomoParams : object
        Contains tomography parameters:
        - sampling (int): Number of grid samples per axis
        - mask (ndarray): 2D boolean grid mask
    
    lgsWfsParams : object
        LGS WFS parameters:
        - D (float): Telescope diameter [m]
        - wfs_lenslets_rotation (ndarray): Lenslet rotations [rad]
        - wfs_lenslets_offset (ndarray): Lenslet offsets [normalized]
    
    atmParams : object
        Atmospheric parameters:
        - nLayer (int): Number of turbulence layers
        - altitude (ndarray): Layer altitudes [m]
        - r0 (float): Fried parameter [m]
        - L0 (float): Outer scale [m]
        - fractionnalR0 (ndarray): Turbulence strength per layer
    
    lgsAsterismParams : object
        LGS constellation parameters:
        - nLGS (int): Number of LGS
        - directionVectorLGS (ndarray): Direction vectors
        - LGSheight (ndarray): LGS heights [m]
    
    gridMask : ndarray
        2D boolean mask for valid grid points
    
    Returns:
    --------
    S : ndarray
        Cross-correlation meta-matrix of shape (nGs*valid_pts, nGs*valid_pts)
    """
    
    print("-->> Computing cross-correlation meta-matrix <<--\n")
    # ======================================================================
    # Parameter Extraction
    # ======================================================================
    # Tomography parameters
    sampling = tomoParams.sampling
    
    if gridMask is None:
        mask = np.ones((sampling,sampling),dtype=bool)
    else:
        mask = gridMask
        
    
    nSs  = tomoParams.nFitSrc**2
    srcCCdirectionVector = tomoParams.directionVectorSrc
    srcCCheight = tomoParams.fitSrcHeight
    
    # LGS constellation parameters
    nGs = lgsAsterismParams.nLGS
    srcACdirectionVector = lgsAsterismParams.directionVectorLGS
    srcACheight  = lgsAsterismParams.LGSheight
    
    # WFS parameters
    D = lgsWfsParams.DSupport  
    wfsLensletsRotation = lgsWfsParams.wfsLensletsRotation
    wfs_lenslets_offset = lgsWfsParams.wfs_lenslets_offset
    
    # Atmospheric parameters
    nLayer = atmParams.nLayer
    altitude = atmParams.altitude
    r0 = atmParams.r0
    L0 = atmParams.L0
    fractionnalR0 = atmParams.fractionnalR0
    
    # Initialize a 2d list (nSs,nGs) of zero matrices of size (sampling**2,sampling**2)
    C = [[np.zeros((np.sum(sampling**2),np.sum(sampling**2))) for _ in range(nGs)] for _ in range(nSs)]
    
    for k in range(nSs*nGs):
        # Get the indices kGs and jGs 
        kGs, iGs = np.unravel_index(k, (nSs, nGs)) 
        
        buf = 0
        
        # Create grids for the first and second guide stars
        x1, y1 = create_guide_star_grid(sampling, D, wfsLensletsRotation[iGs], 
                                        wfs_lenslets_offset[0, iGs], wfs_lenslets_offset[1, iGs])
        
        x2, y2 = np.meshgrid(np.linspace(-1, 1, sampling) * D/2,
                            np.linspace(-1, 1, sampling) * D/2)
        
        for kLayer in range(nLayer):
            # Calculate the scaled and shifted coordinates for the first and second guide stars
            iZ = calculate_scaled_shifted_coords(x1, y1, srcACdirectionVector, 
                                                iGs, altitude, kLayer, srcACheight)
            jZ = calculate_scaled_shifted_coords(x2, y2, srcCCdirectionVector, 
                                                kGs, altitude, kLayer, srcCCheight)
            
            # Compute the covariance matrix
            out = covariance_matrix(iZ.T, jZ.T, r0, L0, fractionnalR0[kLayer])
            out = out[mask.flatten(),:]
            out = out[:,mask.flatten()]
            # Accumulate the results
            buf += out
        
        C[kGs][iGs] = buf.T
    
    # Rearrange the results into a single array
    C = np.array([np.concatenate(row, axis=1) for row in C])
    
    return C

