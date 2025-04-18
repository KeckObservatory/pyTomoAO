import numpy as np
import math
import numba as nb
from scipy.special import gamma
from scipy.sparse import block_diag

@nb.njit(nb.complex128(nb.complex128), cache=False)
def _kv56_scalar(z):
    """Scalar implementation used as kernel for array version"""
    # Precomputed Gamma function values for v=5/6
    gamma_1_6 = 5.56631600178  # Gamma(1/6)
    gamma_11_6 = 0.94065585824  # Gamma(11/6)
    # Precompute constants for numerical stability
    # Constants for the series expansion and asymptotic approximation
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
                    (5005.0/177147.0)*z_inv**5  #+ (-2662660.0/4782969.0)*z_inv**6
        prefactor = np.sqrt(np.pi/(2.0*z)) * np.exp(-z)
        K = prefactor * sum_terms
    return K

# Vectorized version outside the class
@nb.vectorize([nb.complex128(nb.complex128),  # Complex input
            nb.complex128(nb.float64)],    # Real input
            nopython=True, target='parallel')
def _kv56(z):
    """
    Modified Bessel function K_{5/6}(z) for numpy arrays
    Handles both real and complex inputs efficiently
    """
    return _kv56_scalar(z)

def _rotateWFS(px,py, rotAngleInRadians):
    """
    This function rotate the WFS subapertures positions.
    
    Parameters:
    -----------
        px (1D array): The original WFS X subaperture position.
        py (1D array): The original WFS Y subaperture position.
        rotAngleInRadians (double): The rotation angle in radians.
    
    Returns:
    --------
        pxx (1D array): The new WFS X subaperture position after rotation.
        pyy (1D array): The new WFS Y subapertuer position after rotation.
    """
    pxx = px * math.cos(rotAngleInRadians) - py * math.sin(rotAngleInRadians)
    pyy= py * math.cos(rotAngleInRadians) + px * math.sin(rotAngleInRadians)
    return pxx, pyy

def _create_guide_star_grid(sampling, D, rotation_angle, offset_x, offset_y):
    """
    Create a grid of guide star positions based on the specified parameters.

    Parameters:
    -----------
        sampling (int): Number of samples in each dimension for the grid.
        D (float): Diameter of the telescope, used to scale the grid.
        rotation_angle (float): Angle to rotate the grid in degrees.
        offset_x (float): Offset in the x-direction to apply to the grid.
        offset_y (float): Offset in the y-direction to apply to the grid.

    Returns:
    --------
        tuple: Two 2D arrays representing the x and y coordinates of the guide stars.
    """
    
    # Create a grid of points in Cartesian coordinates
    x, y = np.meshgrid(np.linspace(-1, 1, sampling) * D/2,
                        np.linspace(-1, 1, sampling) * D/2)
    
    # Flatten the grid, rotate the positions, and apply the specified offsets
    x, y = _rotateWFS(x.flatten(), y.flatten(), rotation_angle * 180/np.pi)
    x = x - offset_x * D  # Apply x offset
    y = y - offset_y * D  # Apply y offset
    
    # Reshape the modified coordinates back to the original grid shape
    return x.reshape(sampling, sampling), y.reshape(sampling, sampling)

def _calculate_scaled_shifted_coords(x, y, srcACdirectionVector, gs_index, 
                                    altitude, kLayer, srcACheight):
    """
    Calculate the scaled and shifted coordinates for a guide star.

    Parameters:
    -----------
        x (ndarray): The x-coordinates in Cartesian space.
        y (ndarray): The y-coordinates in Cartesian space.
        srcACdirectionVector (ndarray): Direction vectors for the guide stars.
        gs_index (int): Index of the guide star being processed.
        altitude (ndarray): Altitudes of the turbulence layers.
        kLayer (int): Index of the current turbulence layer.
        srcACheight (float): Height of the source guide star.

    Returns:
    --------
        complex: The scaled and shifted coordinates as a complex number,
                where the real part is the x-coordinate and the imaginary
                part is the y-coordinate.
    """
    # Calculate the beta shift based on the direction vector and altitude
    beta = srcACdirectionVector[:, gs_index] * altitude[kLayer]
    
    # Calculate the scaling factor based on the altitude and source height
    scale = 1 - altitude[kLayer] / srcACheight
    
    # Return the scaled and shifted coordinates as a complex number
    return x * scale + beta[0] + 1j * (y * scale + beta[1])

def _compute_block(rho_block, L0, cst, var_term):
    """
    Vectorized computation of covariance values for a matrix block
    """
    # Initialize output with variance term
    out = np.full(rho_block.shape, var_term, dtype=np.float64)
    # Find non-zero distances and compute covariance
    mask = rho_block != 0
    u = (2 * np.pi * rho_block[mask]) / L0
    # Vectorized Bessel function calculation with explicit conversion to real
    out[mask] = cst * u**(5/6) * np.real(_kv56(u.astype(np.complex128)))
    return out

def _covariance_matrix(*args):
    """
    Optimized phase covariance matrix calculation using Von Karman turbulence model
    
    Parameters:
    -----------
        *args: (rho1, [rho2], r0, L0, fractionalR0)
            rho1, rho2: Complex coordinate arrays (x + iy)
            r0: Fried parameter (m)
            L0: Outer scale (m)
            fractionalR0: Turbulence layer weighting factor
    
    Returns:
    --------
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


def _auto_correlation(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, gridMask):
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
    #print("-->> Computing auto-correlation meta-matrix <<--\n")
    # ======================================================================
    # Parameter Extraction
    # ======================================================================
    # Tomography parameters
    tomoParams.sampling = gridMask.shape[0]
    sampling = tomoParams.sampling
    mask = gridMask
    
    # LGS constellation parameters
    nGs = lgsAsterismParams.nLGS
    srcACdirectionVector = lgsAsterismParams.directionVectorLGS
    srcACheight  = lgsAsterismParams.LGSheight
    
    # WFS parameters
    D = lgsWfsParams.DSupport  
    wfsLensletsRotation = lgsWfsParams.wfsLensletsRotation
    wfsLensletsOffset = lgsWfsParams.wfsLensletsOffset
    
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
        x1, y1 = _create_guide_star_grid(sampling, D, wfsLensletsRotation[iGs], 
                                        wfsLensletsOffset[0, iGs], wfsLensletsOffset[1, iGs])
        x2, y2 = _create_guide_star_grid(sampling, D, wfsLensletsRotation[jGs], 
                                        wfsLensletsOffset[0, jGs], wfsLensletsOffset[1, jGs])
        
        for kLayer in range(nLayer):
            # Calculate the scaled and shifted coordinates for the first and second guide stars
            iZ = _calculate_scaled_shifted_coords(x1, y1, srcACdirectionVector, iGs, altitude, kLayer, srcACheight)
            jZ = _calculate_scaled_shifted_coords(x2, y2, srcACdirectionVector, jGs, altitude, kLayer, srcACheight)
            
            # Compute the covariance matrix
            out = _covariance_matrix(iZ.T, jZ.T, r0, L0, fractionnalR0[kLayer])
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
    
    # If you want these as a 1D array of indices    
    diagonal_indices_1d = np.diag_indices(nGs)[0] * nGs + np.diag_indices(nGs)[1]
    
    for i in diagonal_indices_1d:
        S_tmp[i] = S_tmp[0]   
    
    S_tmp = np.stack(S_tmp, axis=0)
    S = S_tmp.reshape(nGs, nGs, np.sum(mask), np.sum(mask))\
        .transpose(0, 2, 1, 3).reshape(nGs*np.sum(mask), nGs*np.sum(mask))
        
    # Make the matrix symmetric
    S = np.tril(S) + np.triu(S.T, 1)
    
    return S

def _cross_correlation(tomoParams,lgsWfsParams, atmParams,lgsAsterismParams,gridMask=None):
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
    #print("-->> Computing cross-correlation meta-matrix <<--\n")
    # ======================================================================
    # Parameter Extraction
    # ======================================================================
    # Tomography parameters
    try:    
        tomoParams.sampling = gridMask.shape[0]
    except:
        tomoParams.sampling = 49

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
    wfsLensletsOffset = lgsWfsParams.wfsLensletsOffset
    
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
        x1, y1 = _create_guide_star_grid(sampling, D, wfsLensletsRotation[iGs], 
                                        wfsLensletsOffset[0, iGs], wfsLensletsOffset[1, iGs])
        
        x2, y2 = np.meshgrid(np.linspace(-1, 1, sampling) * D/2,
                            np.linspace(-1, 1, sampling) * D/2)
        
        for kLayer in range(nLayer):
            # Calculate the scaled and shifted coordinates for the first and second guide stars
            iZ = _calculate_scaled_shifted_coords(x1, y1, srcACdirectionVector, 
                                                iGs, altitude, kLayer, srcACheight)
            jZ = _calculate_scaled_shifted_coords(x2, y2, srcCCdirectionVector, 
                                                kGs, altitude, kLayer, srcCCheight)
            
            # Compute the covariance matrix
            out = _covariance_matrix(iZ.T, jZ.T, r0, L0, fractionnalR0[kLayer])
            out = out[mask.flatten(),:]
            out = out[:,mask.flatten()]
            # Accumulate the results
            buf += out
        
        C[kGs][iGs] = buf.T
    
    # Rearrange the results into a single array
    C = np.array([np.concatenate(row, axis=1) for row in C])
    
    return C

def _sparseGradientMatrixAmplitudeWeighted(validLenslet, amplMask=None, overSampling=2, stencilSize=3):
    """
    Computes the sparse gradient matrix (3x3 or 5x5 stencil) with amplitude mask.
    
    Parameters:
    ----------
    validLenslet : 2D array
        Valid lenslet map
    amplMask : 2D array
        Amplitudes Weight Mask (default=None). 
    overSampling : int
        Oversampling factor for the gridMask. Can be either 2 or 4 (default=2).
    
    Returns:
    -------
    Gamma : scipy.sparse.csr_matrix
        Sparse gradient matrix.
    gridMask : 2D array
        Mask used for the reconstructed phase.
    """
    #print("-->> Computing sparse gradient matrix <<--\n")
    
    import numpy as np
    # Get dimensions and counts
    nLenslet = validLenslet.shape[0]  # Size of lenslet array
    nMap = overSampling * nLenslet + 1  # Size of oversampled grid
    nValidLenslet_ = np.count_nonzero(validLenslet)  # Number of valid lenslets
    
    # Create default amplitude mask if none provided
    if amplMask is None:
        amplMask = np.ones((nMap, nMap))

    # Set up stencil parameters based on oversampling factor
    if stencilSize == 3:
        # 3x3 stencil for 2x oversampling
        
        s0x = np.array([-1/4, -1/2, -1/4, 0, 0, 0, 1/4, 1/2, 1/4])  # x-gradient weights
        s0y = -np.array([1/4, 0, -1/4, 1/2, 0, -1/2, 1/4, 0, -1/4])  # y-gradient weights
        num_points = 9
    elif stencilSize == 5:
        # 5x5 stencil for 4x oversampling
        
        s0x = np.array([-1/16, -3/16, -1/2, -3/16, -1/16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                        0, 0, 0, 0, 0, 1/16, 3/16, 1/2, 3/16, 1/16])  # x-gradient weights
        s0y = s0x.reshape(5,5).T.flatten()  # y-gradient weights (transpose of x)
        num_points = 25
    else:
        raise ValueError("overSampling must be 2 or 4")

    # Initialize stencil position arrays
    i0x = np.tile(np.arange(1, stencilSize+1), stencilSize)  # Row indices
    j0x = np.repeat(np.arange(1, stencilSize+1), stencilSize)  # Column indices
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
    iMap0, jMap0 = np.meshgrid(np.arange(1, stencilSize+1), np.arange(1, stencilSize+1))
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
    from scipy.sparse import csr_matrix
    import numpy as np
    
    indx = np.ravel_multi_index((i_x.astype(int) - 1, j_x.astype(int) - 1), (nMap, nMap), order='F')
    indy = np.ravel_multi_index((i_y.astype(int) - 1, j_y.astype(int) - 1), (nMap, nMap), order='F')
    v = np.tile(np.arange(1, 2 * nValidLenslet_ + 1), (u.size, 1)).T
    
    # Construct final sparse gradient matrix
    Gamma = csr_matrix((np.concatenate((s_x, s_y)), (v.flatten() - 1, np.concatenate((indx, indy)))),
                    shape=(2 * nValidLenslet_, nMap ** 2))
    Gamma = Gamma[:, gridMask.ravel()]  # Apply mask to reduce matrix size

    return Gamma, gridMask

def _build_reconstructor_model(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams):
        
    Gamma, gridMask = _sparseGradientMatrixAmplitudeWeighted(
        lgsWfsParams.validLLMapSupport,
        amplMask=None, 
        overSampling=2
    )
    GammaBeta = Gamma/(2*math.pi)

    Gamma_list = []
    for kGs in range(lgsAsterismParams.nLGS):
        Gamma_list.append(Gamma)

    Gamma = block_diag(Gamma_list)

    # Update sampling parameter for Super Resolution
    tomoParams.sampling = gridMask.shape[0]

    Cxx = _auto_correlation(
        tomoParams,
        lgsWfsParams, 
        atmParams,
        lgsAsterismParams,
        gridMask
    )

    # Update the tomography parameters to include the fitting weight for each source
    tomoParams.fitSrcWeight = np.ones(tomoParams.nFitSrc**2)/tomoParams.nFitSrc**2

    Cox = _cross_correlation(
        tomoParams,
        lgsWfsParams, 
        atmParams,
        lgsAsterismParams
    )

    CoxOut = 0
    for i in range(tomoParams.nFitSrc**2):
        CoxOut = CoxOut + Cox[i,:,:]*tomoParams.fitSrcWeight[i]

    row_mask = gridMask.ravel().astype(bool)
    col_mask = np.tile(gridMask.ravel().astype(bool), lgsAsterismParams.nLGS)

    # Select submatrix using boolean masks with np.ix_ for correct indexing
    Cox = CoxOut[np.ix_(row_mask, col_mask)]

    CnZ = np.eye(Gamma.shape[0]) * 1/10 * np.mean(np.diag(Gamma @ Cxx @ Gamma.T))
    invCss = np.linalg.inv(Gamma @ Cxx @ Gamma.T + CnZ)

    RecStatSA = Cox @ Gamma.T @ invCss

    # LGS WFS subapertures diameter
    d = lgsWfsParams.DSupport/lgsWfsParams.validLLMapSupport.shape[0]

    # Size of the pixel at Shannon sampling
    _wavefront2Meter = lgsAsterismParams.LGSwavelength/d/2

    # Compute final scaled reconstructor
    _reconstructor = d * _wavefront2Meter * RecStatSA

    return _reconstructor, Gamma, gridMask, Cxx, Cox, CnZ, RecStatSA

def _build_reconstructor_im(IM, tomoParams, lgsWfsParams, atmParams, lgsAsterismParams):
        
    Gamma, gridMask = _sparseGradientMatrixAmplitudeWeighted(
        lgsWfsParams.validLLMapSupport,
        amplMask=None, 
        overSampling=1
    )
    GammaBeta = Gamma/(2*math.pi)

    Gamma_list = []
    for kGs in range(lgsAsterismParams.nLGS):
        Gamma_list.append(Gamma)

    Gamma = block_diag(Gamma_list)

    # Update sampling parameter for Super Resolution
    tomoParams.sampling = gridMask.shape[0]

    Cxx = _auto_correlation(
        tomoParams,
        lgsWfsParams, 
        atmParams,
        lgsAsterismParams,
        gridMask
    )

    # Update the tomography parameters to include the fitting weight for each source
    tomoParams.fitSrcWeight = np.ones(tomoParams.nFitSrc**2)/tomoParams.nFitSrc**2

    Cox = _cross_correlation(
        tomoParams,
        lgsWfsParams, 
        atmParams,
        lgsAsterismParams
    )

    CoxOut = 0
    for i in range(tomoParams.nFitSrc**2):
        CoxOut = CoxOut + Cox[i,:,:]*tomoParams.fitSrcWeight[i]

    row_mask = gridMask.ravel().astype(bool)
    col_mask = np.tile(gridMask.ravel().astype(bool), lgsAsterismParams.nLGS)

    # Select submatrix using boolean masks with np.ix_ for correct indexing
    Cox = CoxOut[np.ix_(row_mask, col_mask)]

    CnZ = np.eye(Gamma.shape[0]) * 1/10 * np.mean(np.diag(Gamma @ Cxx @ Gamma.T))
    invCss = np.linalg.inv(Gamma @ Cxx @ Gamma.T + CnZ)

    RecStatSA = Cox @ Gamma.T @ invCss

    # LGS WFS subapertures diameter
    d = lgsWfsParams.DSupport/lgsWfsParams.validLLMapSupport.shape[0]

    # Size of the pixel at Shannon sampling
    _wavefront2Meter = lgsAsterismParams.LGSwavelength/d/2

    # Compute final scaled reconstructor
    _reconstructor = d * _wavefront2Meter * RecStatSA

    return _reconstructor, Gamma, gridMask, Cxx, Cox, CnZ, RecStatSA
