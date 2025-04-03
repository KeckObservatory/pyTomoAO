#%%
import numpy as np
import cupy as cp
from scipy.special import kv, gamma
import time 
import numba as nb
import math
# Define the scalar function outside the class so Numba can compile it properly
@nb.njit(nb.complex128(nb.complex128), cache=True)
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

def compute_block(rho_block, L0, cst, var_term):
    return _compute_block_cpu(rho_block, L0, cst, var_term)

def _compute_block_cpu(rho_block, L0, cst, var_term):
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

def rotateWFS(px,py, rotAngleInRadians):
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

def covariance_matrix(*args):
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
                out[i:i_end, j:j_end] = compute_block(
                    block, L0, cst, var_term
                )
        
        # Apply fractional weighting
        out *= fractionalR0
        return out

    # Single block processing for smaller matrices
    out = compute_block(rho, L0, cst, var_term)
    return out * fractionalR0

def create_guide_star_grid(sampling, D, rotation_angle, offset_x, offset_y):
    # Create a grid of points in Cartesian coordinates
    x, y = np.meshgrid(np.linspace(-1, 1, sampling) * D/2,
                        np.linspace(-1, 1, sampling) * D/2)
    
    # Flatten the grid, rotate the positions, and apply the specified offsets
    x, y = rotateWFS(x.flatten(), y.flatten(), rotation_angle * 180/np.pi)
    x = x - offset_x * D  # Apply x offset
    y = y - offset_y * D  # Apply y offset
    
    # Reshape the modified coordinates back to the original grid shape
    return x.reshape(sampling, sampling), y.reshape(sampling, sampling)

def calculate_scaled_shifted_coords(x, y, srcACdirectionVector, gs_index, 
                                    altitude, kLayer, srcACheight):
        # Calculate the beta shift based on the direction vector and altitude
        beta = srcACdirectionVector[:, gs_index] * altitude[kLayer]
        
        # Calculate the scaling factor based on the altitude and source height
        scale = 1 - altitude[kLayer] / srcACheight
        
        # Return the scaled and shifted coordinates as a complex number
        return x * scale + beta[0] + 1j * (y * scale + beta[1])

def auto_correlation(tomoParams, lgsWfsParams, atmParams,lgsAsterismParams,gridMask):
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
        x1, y1 = create_guide_star_grid(sampling, D, wfsLensletsRotation[iGs], 
                                        wfsLensletsOffset[0, iGs], wfsLensletsOffset[1, iGs])
        x2, y2 = create_guide_star_grid(sampling, D, wfsLensletsRotation[jGs], 
                                        wfsLensletsOffset[0, jGs], wfsLensletsOffset[1, jGs])
        
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

# Fake classes to simulate the required parameters
class TomoParams:
    def __init__(self, sampling):
        self.sampling = sampling

class LgsWfsParams:
    def __init__(self, DSupport, wfsLensletsRotation, wfsLensletsOffset):
        self.DSupport = DSupport
        self.wfsLensletsRotation = wfsLensletsRotation
        self.wfsLensletsOffset = wfsLensletsOffset

class AtmParams:
    def __init__(self, nLayer, altitude, r0, L0, fractionnalR0):
        self.nLayer = nLayer
        self.altitude = altitude
        self.r0 = r0
        self.L0 = L0
        self.fractionnalR0 = fractionnalR0

class LgsAsterismParams:
    def __init__(self, nLGS, directionVectorLGS, LGSheight):
        self.nLGS = nLGS
        self.directionVectorLGS = directionVectorLGS
        self.LGSheight = LGSheight

# %%
# Fake parameters
if __name__ == "__main__":
    DSupport = 8.0
    wfsLensletsRotation = np.zeros(4)
    wfsLensletsOffset = np.zeros((2, 4))
    nLayer = 2
    altitude = np.array([    0.  ,         577.35026919,  1154.70053838,  2309.40107676,
    4618.80215352,  9237.60430703, 18475.20861407])
    r0 = 0.171
    L0 = 30.0
    fractionnalR0 = np.array([0.46, 0.13, 0.04, 0.05, 0.12, 0.09, 0.11])
    nLGS = 4
    directionVectorLGS = np.array([[ 3.68458398e-05,  2.25615699e-21, -3.68458398e-05, -6.76847096e-21],
                                    [ 0.00000000e+00, 3.68458398e-05,  4.51231397e-21, -3.68458398e-05],
                                    [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00]])
    LGSheight = 103923.04845413263
    gridMask = np.load("gridMask.npy")  # Load the grid mask from a file
    sampling = gridMask.shape[0]  # Assuming square grid mask

    # Create parameter objects
    tomoParams = TomoParams(sampling)
    lgsWfsParams = LgsWfsParams(DSupport, wfsLensletsRotation, wfsLensletsOffset)
    atmParams = AtmParams(nLayer, altitude, r0, L0, fractionnalR0)
    lgsAsterismParams = LgsAsterismParams(nLGS, directionVectorLGS, LGSheight)

    # Call and benchmark the function
    start_time = time.time()
    S = auto_correlation(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, gridMask)
    end_time = time.time()

    print(f"Auto-correlation matrix shape: {S.shape}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")

# %%
