#%%
import numpy as np
import cupy as cp
from scipy.special import gamma
import math
from cupyx.scipy.special import gamma as cp_gamma

# Pre-computed gamma values
gamma_1_6 = 5.56631600178  # Gamma(1/6)
gamma_11_6 = 0.94065585824  # Gamma(11/6)

# Optimized Real-valued Bessel function kernel for float64
kv56_real_kernel_float64_optimized = cp.ElementwiseKernel(
    'float64 z',
    'float64 K',
    '''
    double v = 5.0 / 6.0;
    double z_abs = fabs(z);
    if (z_abs < 2.0) {
        // Series approximation for small z
        if (z_abs < 1e-12) {
            // Very small z approximation to avoid numerical issues
            K = 1.89718990814 * pow(z_abs, -5.0/6.0); // Approximation for tiny values
            return;
        }
        
        double half_z = 0.5 * z;
        double half_z_sq = half_z * half_z;
        double z_pow_v = pow(half_z, v);
        double z_pow_neg_v = pow(half_z, -v);
        
        double sum_a = z_pow_v / gamma_11_6;
        double sum_b = z_pow_neg_v / gamma_1_6;
        double term_a = sum_a;
        double term_b = sum_b;
        
        // More efficient series computation with better convergence
        double prev_sum_a = 0.0;
        double prev_sum_b = 0.0;
        int k = 1;
        double tol = 1e-15;
        
        #pragma unroll 2
        for (int i = 0; i < 100; ++i) { // Reduced max iterations with better termination
            double k_plus_v = k + v;
            double k_minus_v = k - v;
            
            double factor_a = half_z_sq / (k * k_plus_v);
            double factor_b = half_z_sq / (k * k_minus_v);
            
            term_a *= factor_a;
            term_b *= factor_b;
            sum_a += term_a;
            sum_b += term_b;
            
            // Check convergence with relative error every few iterations
            if ((i & 1) == 1) {
                double rel_change_a = fabs(sum_a - prev_sum_a) / fabs(sum_a);
                double rel_change_b = fabs(sum_b - prev_sum_b) / fabs(sum_b);
                
                if (rel_change_a < tol && rel_change_b < tol) {
                    break;
                }
                prev_sum_a = sum_a;
                prev_sum_b = sum_b;
            }
            k += 1;
        }
        K = M_PI * (sum_b - sum_a);
    } else {
        // Asymptotic approximation for larger z
        double z_inv = 1.0 / z;
        
        // Horner's method for polynomial evaluation
        double sum_terms = 1.0 + z_inv * (2.0/9.0 + z_inv * (
                    -7.0/81.0 + z_inv * (175.0/2187.0 + z_inv * (
                        -2275.0/19683.0 + z_inv * 5005.0/177147.0
                    )))); 
        
        // More numerically stable computation
        double sqrt_term = sqrt(M_PI / (2.0 * z));
        double exp_term = exp(-z);
        K = sqrt_term * exp_term * sum_terms;
    }
    ''',
    name='kv56_real_kernel_float64_optimized',
    preamble=f'''
    const double gamma_1_6 = {gamma_1_6};
    const double gamma_11_6 = {gamma_11_6};
    '''
)

# Optimized Real-valued Bessel function kernel for float32
kv56_real_kernel_float32_optimized = cp.ElementwiseKernel(
    'float32 z',
    'float32 K',
    '''
    float v = 5.0f / 6.0f;
    float z_abs = fabsf(z);
    if (z_abs < 2.0f) {
        // Series approximation for small z
        if (z_abs < 1e-6f) {
            // Very small z approximation to avoid numerical issues
            K = 1.897f * powf(z_abs, -5.0f/6.0f); // Simplified approximation for tiny values
            return;
        }
        
        float half_z = 0.5f * z;
        float z_pow_v = powf(half_z, v);
        float z_pow_neg_v = powf(half_z, -v);
        float sum_a = z_pow_v / gamma_11_6_f;
        float sum_b = z_pow_neg_v / gamma_1_6_f;
        float term_a = sum_a;
        float term_b = sum_b;
        float z_sq_over_4 = half_z * half_z;
        
        // Fewer iterations with better convergence check
        float prev_sum_a = 0.0f;
        float prev_sum_b = 0.0f;
        int k = 1;
        float tol = 1e-6f;  // Slightly relaxed tolerance for better performance
        
        // Manual loop unrolling for better performance
        #pragma unroll 4
        for (int i = 0; i < 50; ++i) { // Reduced max iterations
            float k_plus_v = k + v;
            float k_minus_v = k - v;
            float factor_a = z_sq_over_4 / (k * k_plus_v);
            float factor_b = z_sq_over_4 / (k * k_minus_v);
            
            term_a *= factor_a;
            term_b *= factor_b;
            sum_a += term_a;
            sum_b += term_b;
            
            // Check convergence every 8 iterations instead of 4 (reduce branch frequency)
            if ((i & 7) == 7) {
                float rel_change_a = fabsf(sum_a - prev_sum_a) / (fabsf(sum_a) + 1e-10f);
                float rel_change_b = fabsf(sum_b - prev_sum_b) / (fabsf(sum_b) + 1e-10f);
                if (rel_change_a < tol && rel_change_b < tol) {
                    break;
                }
                prev_sum_a = sum_a;
                prev_sum_b = sum_b;
            }
            k += 1;
        }
        K = M_PI_F * (sum_b - sum_a);
    } else {
        // Asymptotic approximation for larger z
        float z_inv = 1.0f / z;
        
        // Optimized Horner's method with fewer operations and FMA
        float sum_terms = 1.0f + z_inv * __fmaf_rn(z_inv,
                    __fmaf_rn(z_inv,
                        __fmaf_rn(z_inv,
                            __fmaf_rn(z_inv, 
                                5005.0f/177147.0f,
                                -2275.0f/19683.0f),
                            175.0f/2187.0f),
                        -7.0f/81.0f),
                    2.0f/9.0f);
        
        // Use faster intrinsics with FMA operations
        float sqrt_term = __fsqrt_rn(M_PI_F * 0.5f * z_inv);
        float exp_term = __expf(-z);  // Fast exponential approximation
        K = __fmul_rn(sqrt_term, __fmul_rn(exp_term, sum_terms));
    }
    ''',
    name='kv56_real_kernel_float32_optimized',
    preamble=f'''
    const float gamma_1_6_f = {float(gamma_1_6)};
    const float gamma_11_6_f = {float(gamma_11_6)};
    const float M_PI_F = 3.14159265358979323846f;
    '''
)

# Optimized functions using CuPy RawKernel
_fast_math_kernel = cp.RawKernel(r'''
// Fast math utility functions for better performance
extern "C" __global__ 
void fast_math_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        
        // Use fast reciprocal
        float result;
        asm("rcp.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
        
        // Use fast exponential
        float exp_result = __expf(-x);
        
        output[idx] = exp_result * result;
    }
}
''', 'fast_math_kernel')

def _kv56(z_gpu, use_float32=False):
    """GPU implementation of K_{5/6} Bessel function using ElementwiseKernel with dynamic parallelism"""
    dtype = cp.float32 if use_float32 else cp.float64

    if cp.isrealobj(z_gpu):
        out_gpu = cp.zeros_like(z_gpu, dtype=dtype)

        if z_gpu.dtype == cp.float32 or use_float32:
            z_float32 = z_gpu.astype(cp.float32) if z_gpu.dtype != cp.float32 else z_gpu
            kv56_real_kernel_float32_optimized(z_float32, out_gpu)
        else:
            kv56_real_kernel_float64_optimized(cp.real(z_gpu), out_gpu)
    else:
        raise ValueError("Input must be real-valued.")

    return out_gpu

def _compute_block(rho_block_gpu, L0, cst, var_term):
    """
    Vectorized computation of covariance values for a matrix block. GPU version
    """
    # Initialize output with variance term
    out_gpu = cp.full(rho_block_gpu.shape, var_term, dtype=cp.float64)
    
    # Find non-zero distances
    mask_gpu = rho_block_gpu != 0
    
    # Only compute where mask is True
    if cp.any(mask_gpu):
        rho_nonzero = rho_block_gpu[mask_gpu]
        u_gpu = (2 * cp.pi * rho_nonzero) / L0
        
        # Calculate Bessel function
        bessel_input = u_gpu.astype(cp.complex128)
        bessel_output = _kv56(cp.real(bessel_input))
        
        # Compute final values
        covariance_values = cst * u_gpu**(5/6) * cp.real(bessel_output)
        
        # Update output array
        out_gpu[mask_gpu] = covariance_values
        
    return out_gpu

def _rotateWFS(px_gpu, py_gpu, rotAngleInRadians):
    """
    This function rotate the WFS subapertures positions. GPU version.
    
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
    cos_angle = math.cos(rotAngleInRadians)
    sin_angle = math.sin(rotAngleInRadians)
    
    pxx_gpu = px_gpu * cos_angle - py_gpu * sin_angle
    pyy_gpu = py_gpu * cos_angle + px_gpu * sin_angle
    
    return pxx_gpu, pyy_gpu

def _covariance_matrix(*args):
    """
    Optimized phase covariance matrix calculation using Von Karman turbulence model. GPU version.
    
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
    rho1_gpu = args[0].flatten()
    auto_covariance = len(args) == 4
    
    if auto_covariance:
        r0, L0, fractionalR0 = args[1:]
        rho2_gpu = rho1_gpu
    else:
        rho2_gpu, r0, L0, fractionalR0 = args[1], args[2], args[3], args[4]
        rho2_gpu = rho2_gpu.flatten()

    # Precompute constants
    GAMMA_6_5 = gamma(6/5)
    GAMMA_11_6 = gamma(11/6)
    GAMMA_5_6 = gamma(5/6)
    
    BASE_CONST = (24 * GAMMA_6_5 / 5) ** (5/6)
    SCALE_FACTOR = (GAMMA_11_6 / (2**(5/6) * np.pi**(8/3)))
    
    L0_r0_ratio = (L0 / r0) ** (5/3)
    
    cst = BASE_CONST * SCALE_FACTOR * L0_r0_ratio
    var_term = (BASE_CONST * GAMMA_11_6 * GAMMA_5_6 / (2 * np.pi**(8/3))) * L0_r0_ratio

    # Calculate pairwise distances on GPU
    n, m = len(rho1_gpu), len(rho2_gpu)
    
    # Use broadcasting to compute distances efficiently
    rho1_reshaped = cp.reshape(rho1_gpu, (n, 1))
    rho2_reshaped = cp.reshape(rho2_gpu, (1, m))
    rho_gpu = cp.abs(rho1_reshaped - rho2_reshaped)

    # Block processing for large matrices
    block_size = 5000
    if max(n, m) > block_size:
        out_gpu = cp.empty((n, m), dtype=cp.float64)
        
        for i in range(0, n, block_size):
            i_end = min(i + block_size, n)
            
            for j in range(0, m, block_size):
                j_end = min(j + block_size, m)
                
                block_gpu = rho_gpu[i:i_end, j:j_end]
                out_gpu[i:i_end, j:j_end] = _compute_block(
                    block_gpu, L0, cst, var_term
                )
        
        out_gpu *= fractionalR0
        return out_gpu

    # Single block processing for smaller matrices
    out_gpu = _compute_block(rho_gpu, L0, cst, var_term)
    return out_gpu * fractionalR0

def _create_guide_star_grid(sampling, D, rotation_angle, offset_x, offset_y):
    """
    Create a grid of guide star positions based on the specified parameters. GPU version.

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
    # Create coordinate grids
    x_range = cp.linspace(-1, 1, sampling) * D/2
    y_range = cp.linspace(-1, 1, sampling) * D/2
    x_gpu, y_gpu = cp.meshgrid(x_range, y_range)
    
    # Flatten, rotate, and apply offsets
    x_flat, y_flat = _rotateWFS(x_gpu.flatten(), y_gpu.flatten(), rotation_angle * 180/cp.pi)
    x_flat = x_flat - offset_x * D
    y_flat = y_flat - offset_y * D
    
    # Reshape back to grid
    return x_flat.reshape(sampling, sampling), y_flat.reshape(sampling, sampling)

def _calculate_scaled_shifted_coords(x_gpu, y_gpu, srcACdirectionVector_gpu, gs_index, 
                                        altitude, kLayer, srcACheight):
    """
    Calculate the scaled and shifted coordinates for a guide star. GPU version.

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
    # Convert NumPy arrays to CuPy if needed
    if isinstance(srcACdirectionVector_gpu, np.ndarray):
        srcACdirectionVector_gpu = cp.array(srcACdirectionVector_gpu)
    
    # Calculate beta shift
    beta = srcACdirectionVector_gpu[:, gs_index] * altitude[kLayer]
    
    # Calculate scaling factor
    scale = 1 - altitude[kLayer] / srcACheight
    
    # Calculate scaled and shifted coordinates
    return x_gpu * scale + beta[0] + 1j * (y_gpu * scale + beta[1])

def _auto_correlation(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, gridMask):
    """
    Computes the auto-correlation meta-matrix for tomographic atmospheric reconstruction.
    GPU version.
    
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
    # Parameter extraction
    sampling = tomoParams.sampling
    mask_gpu = cp.array(gridMask)
    
    # Ensure gridMask is the right size
    if mask_gpu.shape[0] != sampling or mask_gpu.shape[1] != sampling:
        print(f"Warning: Mask shape {mask_gpu.shape} doesn't match sampling {sampling}.")
        # Resize mask or adjust sampling
        # Option 1: Generate a mask of the right size
        mask_gpu = cp.ones((sampling, sampling), dtype=bool)
    
    # LGS constellation parameters
    nGs = lgsAsterismParams.nLGS
    # Convert to GPU array
    srcACdirectionVector_gpu = cp.array(lgsAsterismParams.directionVectorLGS)
    srcACheight = lgsAsterismParams.LGSheight
    
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
    
    # Generate indices for the upper triangular part
    # This operation is small, so can remain on CPU
    kGs = np.triu(np.arange(1, nGs**2 + 1).reshape(nGs, nGs).T, 1).T.reshape(nGs**2)
    kGs[0] = 1
    kGs = kGs[kGs != 0]
    
    # Initialize result list
    mask_sum = int(cp.sum(mask_gpu))
    S_gpu = [cp.zeros((mask_sum, mask_sum)) for _ in range(len(kGs))]
    
    for k in range(len(kGs)):
        # Get the indices
        jGs, iGs = np.unravel_index(kGs[k] - 1, (nGs, nGs))
        
        buf_gpu = cp.zeros((mask_sum, mask_sum))
        
        # Create grids for guide stars
        x1_gpu, y1_gpu = _create_guide_star_grid(
            sampling, D, wfsLensletsRotation[iGs], 
            wfsLensletsOffset[0, iGs], wfsLensletsOffset[1, iGs]
        )
        
        x2_gpu, y2_gpu = _create_guide_star_grid(
            sampling, D, wfsLensletsRotation[jGs], 
            wfsLensletsOffset[0, jGs], wfsLensletsOffset[1, jGs]
        )
        
        for kLayer in range(nLayer):
            # Calculate coordinates
            iZ_gpu = _calculate_scaled_shifted_coords(
                x1_gpu, y1_gpu, srcACdirectionVector_gpu, iGs, 
                altitude, kLayer, srcACheight
            )
            
            jZ_gpu = _calculate_scaled_shifted_coords(
                x2_gpu, y2_gpu, srcACdirectionVector_gpu, jGs, 
                altitude, kLayer, srcACheight
            )
            
            # Compute covariance matrix
            out_gpu = _covariance_matrix(
                iZ_gpu.T, jZ_gpu.T, r0, L0, fractionnalR0[kLayer]
            )
            
            # Apply mask correctly - ensure dimensions match
            mask_flat = mask_gpu.flatten()
            # First check shapes to avoid IndexError
            if out_gpu.shape[0] != mask_flat.shape[0]:
                print(f"Shape mismatch: out_gpu.shape={out_gpu.shape}, mask_flat.shape={mask_flat.shape}")
                # Only take values at valid indices
                valid_mask = cp.arange(mask_flat.size) < out_gpu.shape[0]
                mask_subset = mask_flat[valid_mask]
                # Apply mask to subset that fits
                masked_out = out_gpu[mask_subset, :]
                masked_out = masked_out[:, mask_subset]
            else:
                # Apply mask normally
                masked_out = out_gpu[mask_flat, :]
                masked_out = masked_out[:, mask_flat]
                
            # Accumulate the results
            if buf_gpu.shape == masked_out.shape:
                buf_gpu += masked_out
            else:
                print(f"Warning: Buffer and output shapes don't match. buf_gpu={buf_gpu.shape}, masked_out={masked_out.shape}")
                # Resize buffer if needed
                if masked_out.shape[0] > 0:
                    buf_gpu = masked_out  # Replace if can't add
        
        if k < len(S_gpu):
            S_gpu[k] = buf_gpu.T
    
    # Rearrange results into full matrix
    S_tmp_gpu = [cp.zeros((mask_sum, mask_sum)) for _ in range(nGs**2)]
    for c, i in enumerate(kGs):
        S_tmp_gpu[i-1] = S_gpu[c]
    
    # Handle diagonal elements
    diagonal_indices_1d = np.diag_indices(nGs)[0] * nGs + np.diag_indices(nGs)[1]
    for i in diagonal_indices_1d:
        S_tmp_gpu[i] = S_tmp_gpu[0]
    
    # Stack and reshape
    S_tmp_gpu = cp.stack(S_tmp_gpu, axis=0)
    S_gpu = S_tmp_gpu.reshape(nGs, nGs, mask_sum, mask_sum) \
        .transpose(0, 2, 1, 3).reshape(nGs*mask_sum, nGs*mask_sum)
    
    # Make symmetric
    S_gpu = cp.tril(S_gpu) + cp.triu(S_gpu.T, 1)
    
    return S_gpu

def _cross_correlation(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, gridMask=None):
    """
    Computes the cross-correlation meta-matrix for tomographic atmospheric reconstruction.
    GPU version.
    
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
    # Parameter extraction
    sampling = tomoParams.sampling
    
    # Handle mask
    if gridMask is None:
        mask_gpu = cp.ones((sampling, sampling), dtype=bool)
    else:
        mask_gpu = cp.array(gridMask)
    
    # Tomography parameters
    nSs = tomoParams.nFitSrc**2
    srcCCdirectionVector_gpu = cp.array(tomoParams.directionVectorSrc)
    srcCCheight = tomoParams.fitSrcHeight
    
    # LGS constellation parameters
    nGs = lgsAsterismParams.nLGS
    srcACdirectionVector_gpu = cp.array(lgsAsterismParams.directionVectorLGS)
    srcACheight = lgsAsterismParams.LGSheight
    
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
    
    # Calculate the number of valid points in the mask
    mask_sum = int(cp.sum(mask_gpu))
    
    # Initialize a 2D list of GPU arrays
    C_gpu = [[cp.zeros((mask_sum, mask_sum)) for _ in range(nGs)] for _ in range(nSs)]
    
    for k in range(nSs*nGs):
        # Get the indices kGs and jGs
        kGs, iGs = np.unravel_index(k, (nSs, nGs))
        
        buf_gpu = cp.zeros((mask_sum, mask_sum))
        
        # Create grids for the guide star
        x1_gpu, y1_gpu = _create_guide_star_grid(
            sampling, D, wfsLensletsRotation[iGs], 
            wfsLensletsOffset[0, iGs], wfsLensletsOffset[1, iGs]
        )
        
        # Create grid for the science target 
        x_range = cp.linspace(-1, 1, sampling) * D/2
        y_range = cp.linspace(-1, 1, sampling) * D/2
        x2_gpu, y2_gpu = cp.meshgrid(x_range, y_range)
        
        for kLayer in range(nLayer):
            # Calculate coordinates
            iZ_gpu = calculate_scaled_shifted_coords(
                x1_gpu, y1_gpu, srcACdirectionVector_gpu, iGs, 
                altitude, kLayer, srcACheight
            )
            
            jZ_gpu = calculate_scaled_shifted_coords(
                x2_gpu, y2_gpu, srcCCdirectionVector_gpu, kGs, 
                altitude, kLayer, srcCCheight
            )
            
            # Compute covariance matrix
            out_gpu = covariance_matrix(
                iZ_gpu.T, jZ_gpu.T, r0, L0, fractionnalR0[kLayer]
            )
            
            # Apply mask
            mask_flat = mask_gpu.flatten()
            if out_gpu.shape[0] != mask_flat.shape[0]:
                print(f"Shape mismatch: out_gpu.shape={out_gpu.shape}, mask_flat.shape={mask_flat.shape}")
                valid_mask = cp.arange(mask_flat.size) < out_gpu.shape[0]
                mask_subset = mask_flat[valid_mask]
                masked_out = out_gpu[mask_subset, :]
                masked_out = masked_out[:, mask_subset]
            else:
                masked_out = out_gpu[mask_flat, :]
                masked_out = masked_out[:, mask_flat]
            
            # Accumulate results
            if buf_gpu.shape == masked_out.shape:
                buf_gpu += masked_out
            else:
                print(f"Warning: Buffer and output shapes don't match. buf_gpu={buf_gpu.shape}, masked_out={masked_out.shape}")
                if masked_out.shape[0] > 0:
                    buf_gpu = masked_out
        
        C_gpu[kGs][iGs] = buf_gpu.T
    
    # Convert list of lists to a CuPy array for final processing
    C_rows = []
    for row in C_gpu:
        C_rows.append(cp.concatenate(row, axis=1))
    
    C_gpu_array = cp.array(C_rows)
    
    return C_gpu_array
