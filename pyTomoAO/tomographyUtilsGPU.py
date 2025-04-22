#%%
import numpy as np
import cupy as cp
from scipy.special import gamma
import math
from cupyx.scipy.special import gamma
from scipy.sparse import block_diag
#from test_auto import sparseGradientMatrixAmplitudeWeighted

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

# Modify `_kv56_gpu` to use dynamic parallelism for recursive computations
# This allows threads to launch additional kernels for deeper levels of computation

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

def _compute_block(rho_block_gpu, L0, cst, var_term, use_float32=False):
    """GPU implementation of block computation"""
    dtype = cp.float32 if use_float32 else cp.float64

    var_term_gpu = cp.asarray(var_term, dtype=dtype)
    out_gpu = cp.full(rho_block_gpu.shape, dtype=dtype, fill_value=var_term_gpu)
    mask_gpu = rho_block_gpu != 0

    if cp.any(mask_gpu):
        rho_nonzero = rho_block_gpu[mask_gpu]
        u_gpu = (2 * cp.pi * rho_nonzero) / L0

        bessel_output = _kv56(cp.real(u_gpu.astype(cp.complex64 if use_float32 else cp.complex128)), use_float32)
        covariance_values = cst * u_gpu**(5/6) * cp.real(bessel_output)
        out_gpu[mask_gpu] = covariance_values

    return out_gpu

def _rotateWFS(px_gpu, py_gpu, rotAngleInRadians, use_float32=False):
    """GPU version of the WFS rotation function"""
    dtype = cp.float32 if use_float32 else cp.float64
    
    cos_angle = math.cos(rotAngleInRadians)
    sin_angle = math.sin(rotAngleInRadians)
    
    pxx_gpu = px_gpu * cos_angle - py_gpu * sin_angle
    pyy_gpu = py_gpu * cos_angle + px_gpu * sin_angle
    
    return pxx_gpu, pyy_gpu

def _covariance_matrix(*args, use_float32=False):
    """GPU implementation of covariance matrix calculation"""
    # Set computation dtype
    dtype = cp.float32 if use_float32 else cp.float64
    
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
        out_gpu = cp.empty((n, m), dtype=dtype)
        
        for i in range(0, n, block_size):
            i_end = min(i + block_size, n)
            
            for j in range(0, m, block_size):
                j_end = min(j + block_size, m)
                
                block_gpu = rho_gpu[i:i_end, j:j_end]
                out_gpu[i:i_end, j:j_end] = _compute_block(
                    block_gpu, L0, cst, var_term, use_float32
                )
        
        out_gpu *= fractionalR0
        return out_gpu

    # Single block processing for smaller matrices
    out_gpu = _compute_block(rho_gpu, L0, cst, var_term, use_float32)
    return out_gpu * fractionalR0

def _create_guide_star_grid(sampling, D, rotation_angle, offset_x, offset_y, use_float32=False):
    """GPU version of guide star grid creation"""
    # Set computation dtype
    dtype = cp.float32 if use_float32 else cp.float64
    
    # Create coordinate grids
    x_range = cp.linspace(-1, 1, sampling, dtype=dtype) * D/2
    y_range = cp.linspace(-1, 1, sampling, dtype=dtype) * D/2
    x_gpu, y_gpu = cp.meshgrid(x_range, y_range)
    
    # Flatten, rotate, and apply offsets
    x_flat, y_flat = _rotateWFS(x_gpu.flatten(), y_gpu.flatten(), rotation_angle * 180/cp.pi, use_float32)
    x_flat = x_flat - offset_x * D
    y_flat = y_flat - offset_y * D
    
    # Reshape back to grid
    return x_flat.reshape(sampling, sampling), y_flat.reshape(sampling, sampling)

def _calculate_scaled_shifted_coords(x_gpu, y_gpu, srcACdirectionVector_gpu, gs_index, 
                                        altitude, kLayer, srcACheight, use_float32=False):
    """GPU version of coordinate calculation"""
    # Set computation dtype
    dtype = cp.float32 if use_float32 else cp.float64
    
    # Convert NumPy arrays to CuPy if needed
    if isinstance(srcACdirectionVector_gpu, np.ndarray):
        srcACdirectionVector_gpu = cp.array(srcACdirectionVector_gpu, dtype=dtype)
    
    # Calculate beta shift
    beta = srcACdirectionVector_gpu[:, gs_index] * altitude[kLayer]
    
    # Calculate scaling factor
    scale = 1 - altitude[kLayer] / srcACheight
    
    # Calculate scaled and shifted coordinates
    return x_gpu * scale + beta[0] + 1j * (y_gpu * scale + beta[1])

def _auto_correlation(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, gridMask, use_float32=False):
    """GPU-optimized auto-correlation function"""
    #print("-->> Computing auto-correlation matrix <<--\n")
    # Set computation dtype
    dtype = cp.float32 if use_float32 else cp.float64
    
    # Parameter extraction
    tomoParams.sampling = gridMask.shape[0]
    sampling = tomoParams.sampling
    mask_gpu = cp.array(gridMask, dtype=bool)
    
    # Ensure gridMask is the right size
    if mask_gpu.shape[0] != sampling or mask_gpu.shape[1] != sampling:
        print(f"Warning: Mask shape {mask_gpu.shape} doesn't match sampling {sampling}.")
        # Generate a mask of the right size
        mask_gpu = cp.ones((sampling, sampling), dtype=bool)
    
    # LGS constellation parameters
    nGs = lgsAsterismParams.nLGS
    # Convert to GPU array
    srcACdirectionVector_gpu = cp.array(lgsAsterismParams.directionVectorLGS, dtype=dtype)
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
    mask_flat = mask_gpu.flatten()
    mask_indices = cp.where(mask_flat)[0]
    mask_sum = len(mask_indices)
    S_gpu = [cp.zeros((mask_sum, mask_sum), dtype=dtype) for _ in range(len(kGs))]
    
    for k in range(len(kGs)):
        # Get the indices
        jGs, iGs = np.unravel_index(kGs[k] - 1, (nGs, nGs))
        
        buf_gpu = cp.zeros((mask_sum, mask_sum), dtype=dtype)
        
        # Create grids for guide stars
        x1_gpu, y1_gpu = _create_guide_star_grid(
            sampling, D, wfsLensletsRotation[iGs], 
            wfsLensletsOffset[0, iGs], wfsLensletsOffset[1, iGs],
            use_float32
        )
        
        x2_gpu, y2_gpu = _create_guide_star_grid(
            sampling, D, wfsLensletsRotation[jGs], 
            wfsLensletsOffset[0, jGs], wfsLensletsOffset[1, jGs],
            use_float32
        )
        
        for kLayer in range(nLayer):
            # Calculate coordinates
            iZ_gpu = _calculate_scaled_shifted_coords(
                x1_gpu, y1_gpu, srcACdirectionVector_gpu, iGs, 
                altitude, kLayer, srcACheight, use_float32
            )
            
            jZ_gpu = _calculate_scaled_shifted_coords(
                x2_gpu, y2_gpu, srcACdirectionVector_gpu, jGs, 
                altitude, kLayer, srcACheight, use_float32
            )
            
            # Compute covariance matrix
            out_gpu = _covariance_matrix(
                iZ_gpu.T, jZ_gpu.T, r0, L0, fractionnalR0[kLayer], use_float32=use_float32
            )
            
            # Directly apply mask using precomputed indices
            if out_gpu.shape[0] >= len(mask_flat):
                # When output is large enough to accommodate all mask points
                masked_out = out_gpu[mask_indices, :][:, mask_indices]
                
                # Accumulate results without shape checking
                buf_gpu += masked_out
            else:
                # Only when output is smaller than expected (should be rare)
                valid_limit = min(out_gpu.shape[0], len(mask_indices))
                valid_indices = mask_indices[mask_indices < out_gpu.shape[0]]
                
                if len(valid_indices) > 0:
                    masked_out = out_gpu[valid_indices, :][:, valid_indices]
                    
                    # Resize buffer if needed (first encounter only)
                    if buf_gpu.shape != masked_out.shape:
                        buf_gpu = cp.zeros_like(masked_out)
                    
                    buf_gpu += masked_out
        
        if k < len(S_gpu):
            S_gpu[k] = buf_gpu.T
    
    # Rearrange results into full matrix
    S_tmp_gpu = [cp.zeros((mask_sum, mask_sum), dtype=dtype) for _ in range(nGs**2)]
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

def _cross_correlation(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, gridMask=None, use_float32=False):
    """GPU-optimized cross-correlation function"""
    #print("-->> Computing cross-correlation matrix <<--\n")
    # Set computation dtype
    dtype = cp.float32 if use_float32 else cp.float64
    
    # Parameter extraction
    sampling = tomoParams.sampling
    
    # Handle mask
    if gridMask is None:
        mask_gpu = cp.ones((sampling, sampling), dtype=bool)
    else:
        mask_gpu = cp.array(gridMask, dtype=bool)
    
    # Precompute mask indices for faster indexing
    mask_flat = mask_gpu.flatten()
    mask_indices = cp.where(mask_flat)[0]
    mask_sum = len(mask_indices)
    
    # Tomography parameters
    nSs = tomoParams.nFitSrc**2
    srcCCdirectionVector_gpu = cp.array(tomoParams.directionVectorSrc, dtype=dtype)
    srcCCheight = tomoParams.fitSrcHeight
    
    # LGS constellation parameters
    nGs = lgsAsterismParams.nLGS
    srcACdirectionVector_gpu = cp.array(lgsAsterismParams.directionVectorLGS, dtype=dtype)
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
    
    # Initialize a 2D list of GPU arrays
    C_gpu = [[cp.zeros((mask_sum, mask_sum), dtype=dtype) for _ in range(nGs)] for _ in range(nSs)]
    
    for k in range(nSs*nGs):
        # Get the indices kGs and jGs
        kGs, iGs = np.unravel_index(k, (nSs, nGs))
        
        buf_gpu = cp.zeros((mask_sum, mask_sum), dtype=dtype)
        
        # Create grids for the guide star
        x1_gpu, y1_gpu = _create_guide_star_grid(
            sampling, D, wfsLensletsRotation[iGs], 
            wfsLensletsOffset[0, iGs], wfsLensletsOffset[1, iGs],
            use_float32
        )
        
        # Create grid for the science target 
        x_range = cp.linspace(-1, 1, sampling, dtype=dtype) * D/2
        y_range = cp.linspace(-1, 1, sampling, dtype=dtype) * D/2
        x2_gpu, y2_gpu = cp.meshgrid(x_range, y_range)
        
        for kLayer in range(nLayer):
            # Calculate coordinates
            iZ_gpu = _calculate_scaled_shifted_coords(
                x1_gpu, y1_gpu, srcACdirectionVector_gpu, iGs, 
                altitude, kLayer, srcACheight, use_float32
            )
            
            jZ_gpu = _calculate_scaled_shifted_coords(
                x2_gpu, y2_gpu, srcCCdirectionVector_gpu, kGs, 
                altitude, kLayer, srcCCheight, use_float32
            )
            
            # Compute covariance matrix
            out_gpu = _covariance_matrix(
                iZ_gpu.T, jZ_gpu.T, r0, L0, fractionnalR0[kLayer], use_float32=use_float32
            )
            
            # Directly apply mask using precomputed indices
            if out_gpu.shape[0] >= len(mask_flat):
                # When output is large enough for all mask points
                masked_out = out_gpu[mask_indices, :][:, mask_indices]
                
                # Accumulate without shape checking
                buf_gpu += masked_out
            else:
                # Only when output is smaller than expected (rare case)
                valid_indices = mask_indices[mask_indices < out_gpu.shape[0]]
                
                if len(valid_indices) > 0:
                    masked_out = out_gpu[valid_indices, :][:, valid_indices]
                    
                    # Resize buffer if needed (first encounter only)
                    if buf_gpu.shape != masked_out.shape:
                        buf_gpu = cp.zeros_like(masked_out)
                    
                    buf_gpu += masked_out
        
        C_gpu[kGs][iGs] = buf_gpu.T
    
    # Convert list of lists to a CuPy array for final processing
    C_rows = []
    for row in C_gpu:
        C_rows.append(cp.concatenate(row, axis=1))
    
    C_gpu_array = cp.concatenate(C_rows, axis=0)
    
    return C_gpu_array

def _sparseGradientMatrixAmplitudeWeighted(validLenslet, amplMask=None, overSampling=2):
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

def _build_reconstructor_model(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, use_float32=False):
    """
    GPU-optimized atmospheric tomography reconstructor builder
    
    Parameters:
        tomoParams: Tomography parameters
        lgsWfsParams: Laser guide star WFS parameters
        atmParams: Atmospheric parameters
        lgsAsterismParams: LGS asterism parameters
        use_float32: If True, use float32 precision instead of float64 for faster computation
    
    Returns:
        _reconstructor: Optimized tomographic reconstructor
    """
    cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
    # Set computation dtype
    dtype = cp.float32 if use_float32 else cp.float64
    
    # Create a CUDA stream for asynchronous operations
    stream = cp.cuda.Stream()
    with stream:
        # Get sparse gradient matrix (CPU operation)
        Gamma, _gridMask = _sparseGradientMatrixAmplitudeWeighted(
            lgsWfsParams.validLLMapSupport,
            amplMask=None, 
            overSampling=2
        )

        # Use the exact same block diagonal approach as the CPU version
        # This ensures identical matrix structure
        Gamma_list = []
        for kGs in range(lgsAsterismParams.nLGS):
            Gamma_list.append(Gamma)

        Gamma = cp.array(block_diag(Gamma_list).toarray(), dtype=dtype)
        cp.get_default_memory_pool().free_all_blocks()  # Free memory after Gamma computation

        Cxx = _auto_correlation(
            tomoParams,
            lgsWfsParams, 
            atmParams,
            lgsAsterismParams,
            _gridMask,
            use_float32
        ).astype(dtype)
        cp.get_default_memory_pool().free_all_blocks()  # Free memory after Cxx computation

        tomoParams.fitSrcWeight = cp.ones(tomoParams.nFitSrc**2)/tomoParams.nFitSrc**2
        # Update the tomography parameters with proper conversion
        if isinstance(tomoParams.fitSrcWeight, np.ndarray):
            tomoParams.fitSrcWeight = cp.array(tomoParams.fitSrcWeight, dtype=dtype)
        else:
            tomoParams.fitSrcWeight = cp.ones(tomoParams.nFitSrc**2, dtype=dtype)/tomoParams.nFitSrc**2

        Cox = _cross_correlation(
            tomoParams,
            lgsWfsParams, 
            atmParams,
            lgsAsterismParams,
            use_float32=use_float32
        ).astype(dtype)
        cp.get_default_memory_pool().free_all_blocks()  # Free memory after Cxx computation

        weighted_cox = Cox * tomoParams.fitSrcWeight[:, None, None]
        CoxOut = cp.sum(weighted_cox, axis=0)

        row_mask = _gridMask.ravel().astype(bool)
        col_mask = np.tile(_gridMask.ravel().astype(bool), lgsAsterismParams.nLGS)

        # Select submatrix using boolean masks with np.ix_ for correct indexing
        # DO NOT EDIT THIS WITH CUPY FUNCTIONS, IT WILL BREAK THE GPU VERSION
        idxs = np.ix_(row_mask, col_mask)
        Cox = CoxOut[idxs]

        # Calculate noise covariance
        CnZ = cp.eye(Gamma.shape[0], dtype=dtype) * 1/10 * cp.mean(cp.diag(Gamma @ Cxx @ Gamma.T))
        
        # Keep calculations separate to match CPU version exactly
        GammaCxxGammaT = Gamma @ Cxx @ Gamma.T
        GammaCxxGammaT_reg = GammaCxxGammaT + CnZ

        eye = cp.eye(GammaCxxGammaT_reg.shape[0], dtype=GammaCxxGammaT_reg.dtype)
        invCss = cp.linalg.solve(GammaCxxGammaT_reg, eye)

        # Final computation of reconstructor - match CPU exactly
        RecStatSA = Cox @ Gamma.T @ invCss
        
        # LGS WFS subapertures diameter
        d = lgsWfsParams.DSupport/lgsWfsParams.validLLMapSupport.shape[0]

        # Size of the pixel at Shannon sampling
        _wavefront2Meter = lgsAsterismParams.LGSwavelength/d/2

        # Compute final scaled reconstructor
        _reconstructor = cp.asnumpy(d * _wavefront2Meter * RecStatSA)
        
        # Clean up GPU memory
        cp.get_default_memory_pool().free_all_blocks()

        return _reconstructor, Gamma, _gridMask, Cxx, Cox, CnZ, RecStatSA

def _build_reconstructor_im(IM, tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, dmParams, use_float32=False):
    """
    GPU-optimized atmospheric tomography reconstructor builder
    
    Parameters:
        tomoParams: Tomography parameters
        lgsWfsParams: Laser guide star WFS parameters
        atmParams: Atmospheric parameters
        lgsAsterismParams: LGS asterism parameters
        use_float32: If True, use float32 precision instead of float64 for faster computation
    
    Returns:
        _reconstructor: Optimized tomographic reconstructor
    """
    cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
    # Set computation dtype
    dtype = cp.float32 if use_float32 else cp.float64
    
    # Create a CUDA stream for asynchronous operations
    stream = cp.cuda.Stream()
    with stream:
        # Define the grid mask based on the DM parameters
        gridMask = dmParams.validActuators
        
        # Update sampling parameter for Super Resolution
        tomoParams.sampling = gridMask.shape[0]
        cp.get_default_memory_pool().free_all_blocks()  # Free memory after _gridMask computation

        Cxx = _auto_correlation(
            tomoParams,
            lgsWfsParams, 
            atmParams,
            lgsAsterismParams,
            gridMask,
            use_float32
        ).astype(dtype)
        cp.get_default_memory_pool().free_all_blocks()  # Free memory after Cxx computation

        tomoParams.fitSrcWeight = cp.ones(tomoParams.nFitSrc**2)/tomoParams.nFitSrc**2
        # Update the tomography parameters with proper conversion
        if isinstance(tomoParams.fitSrcWeight, np.ndarray):
            tomoParams.fitSrcWeight = cp.array(tomoParams.fitSrcWeight, dtype=dtype)
        else:
            tomoParams.fitSrcWeight = cp.ones(tomoParams.nFitSrc**2, dtype=dtype)/tomoParams.nFitSrc**2

        Cox = _cross_correlation(
            tomoParams,
            lgsWfsParams, 
            atmParams,
            lgsAsterismParams,
            gridMask,
            use_float32=use_float32
        ).astype(dtype)
        cp.get_default_memory_pool().free_all_blocks()  # Free memory after Cxx computation

        Cox = cp.sqeeze(Cox)

        # Noise covariance matrix
        weight = cp.ones(IM.shape[0])
        alpha = 10
        CnZ = 1e-3 * alpha * cp.diag(1 / (weight.flatten(order='F')))
        
        # Noise covariance matrix
        weight = cp.ones(IM.shape[0])
        alpha = 10
        CnZ = 1e-3 * alpha * cp.diag(1 / (weight.flatten(order='F')))

        
        # Keep calculations separate to match CPU version exactly
        IMCxxIMT = IM @ Cxx @ IM.T
        IMCxxIMT_reg = IMCxxIMT + CnZ

        eye = cp.eye(IMCxxIMT_reg.shape[0], dtype=IMCxxIMT_reg.dtype)
        invCss = cp.linalg.solve(IMCxxIMT_reg, eye)

        # Final computation of reconstructor - match CPU exactly
        RecStatSA = Cox @ IM.T @ invCss

        # Compute final scaled reconstructor
        _reconstructor = cp.asnumpy(RecStatSA)
        
        # Clean up GPU memory
        cp.get_default_memory_pool().free_all_blocks()

        return _reconstructor, gridMask, Cxx, Cox, CnZ, RecStatSA