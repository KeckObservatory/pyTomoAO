import numpy as np
from scipy.sparse import spdiags, coo_matrix

# Sub2ind utility
def sub2ind(shape, row, col):
    return row * shape[1] + col

# ------------------------------------------------------------
# iCxx: Set bi-harmonic operator (approx to inverse phase covariance matrix)
# ------------------------------------------------------------
# We'll create a Laplacian^2 for each layer, store in p_L2, then block diagonal them.
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
    no = nxo*nyo

    # Build a mask to remove points out of range
    mask = (xi >= xo[0]) & (yi >= yo[0]) & (xi <= xo[-1]) & (yi <= yo[-1])
    valid_idx = np.where(mask)[0]

    # local integer coordinate in the 'xo' and 'yo' sense
    ox = np.floor((xi[valid_idx] - xo[0]) / do).astype(int)
    oy = np.floor((yi[valid_idx] - yo[0]) / do).astype(int)

    # local fractional
    fxo = np.abs(xi[valid_idx] - (xo[0] + do*ox)) / do
    fyo = np.abs(yi[valid_idx] - (yo[0] + do*oy)) / do

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
    wfs_lenslets_rotation = lgsWfsParams.wfs_lenslets_rotation
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
        x1, y1 = create_guide_star_grid(sampling, D, wfs_lenslets_rotation[iGs], 
                                        wfs_lenslets_offset[0, iGs], wfs_lenslets_offset[1, iGs])
        x2, y2 = create_guide_star_grid(sampling, D, wfs_lenslets_rotation[jGs], 
                                        wfs_lenslets_offset[0, jGs], wfs_lenslets_offset[1, jGs])
        
        for kLayer in range(nLayer):
            # Calculate the scaled and shifted coordinates for the first and second guide stars
            iZ = calculate_scaled_shifted_coords(x1, y1, srcACdirectionVector, iGs, altitude, kLayer, srcACheight)
            jZ = calculate_scaled_shifted_coords(x2, y2, srcACdirectionVector, jGs, altitude, kLayer, srcACheight)
            
            # Compute the covariance matrix
            out = covariance_matrix(iZ, jZ, r0, L0, fractionnalR0[kLayer])
            out = out[mask.flatten(),:]
            out = out[:,mask.flatten()]
            # Accumulate the results
            buf += out
        
        S[k] = buf
    
    # Rearrange the results into a full nGs x nGs matrix
    buf = S
    S_tmp = [np.zeros((np.sum(mask), np.sum(mask))) for _ in range(nGs**2)]
    for c, i in enumerate(kGs):
        S_tmp[i-1] = buf[c]
    
    index = [5, 10, 15]
    for i in index:
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
                    (175.0/2187.0)*z_inv**3 + (-980.0/6561.0)*z_inv**4
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
    out[mask] = cst * u**(5/6) * kv56(u.astype(np.complex128)) # kv(5/6, u)
    
    return out

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
    wfs_lenslets_rotation = lgsWfsParams.wfs_lenslets_rotation
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
        x1, y1 = create_guide_star_grid(sampling, D, wfs_lenslets_rotation[iGs], 
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
            out = covariance_matrix(iZ, jZ, r0, L0, fractionnalR0[kLayer])
            out = out[mask.flatten(),:]
            out = out[:,mask.flatten()]
            # Accumulate the results
            buf += out
        
        C[kGs][iGs] = buf
    
    # Rearrange the results into a single array
    C = np.array([np.concatenate(row, axis=1) for row in C])
    
    return C

