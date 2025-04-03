#%%
import numpy as np
import cupy as cp
from scipy.special import gamma
import math
from cupyx.scipy.special import gamma
from scipy.sparse import block_diag
from test_auto import sparseGradientMatrixAmplitudeWeighted

# Pre-computed gamma values
gamma_1_6 = 5.56631600178  # Gamma(1/6)
gamma_11_6 = 0.94065585824  # Gamma(11/6)

# Real-valued Bessel function kernel
kv56_real_kernel = cp.ElementwiseKernel(
    'float64 z',
    'float64 K',
    '''
    double v = 5.0 / 6.0;
    double z_abs = fabs(z);
    if (z_abs < 2.0) {
        double sum_a = 0.0;
        double sum_b = 0.0;
        double term_a = pow(0.5 * z, v) / gamma_11_6;
        double term_b = pow(0.5 * z, -v) / gamma_1_6;
        sum_a = term_a;
        sum_b = term_b;
        double z_sq_over_4 = pow(0.5 * z, 2);
        int k = 1;
        double tol = 1e-15;
        int max_iter = 1000;
        for (int i = 0; i < max_iter; ++i) {
            double factor_a = z_sq_over_4 / (k * (k + v));
            term_a *= factor_a;
            sum_a += term_a;
            double factor_b = z_sq_over_4 / (k * (k - v));
            term_b *= factor_b;
            sum_b += term_b;
            if (fabs(term_a) < tol * fabs(sum_a) && fabs(term_b) < tol * fabs(sum_b)) {
                break;
            }
            k += 1;
        }
        K = M_PI * (sum_b - sum_a);
    } else {
        double z_inv = 1.0 / z;
        double sum_terms = 1.0 + z_inv * (2.0/9.0 + z_inv * (
                    -7.0/81.0 + z_inv * (175.0/2187.0 + z_inv * (
                        -2275.0/19683.0 + z_inv * 5005.0/177147.0
                    )))); 
        double prefactor = sqrt(M_PI / (2.0 * z)) * exp(-z);
        K = prefactor * sum_terms;
    }
    ''',
    name='kv56_real_kernel',
    preamble=f'''
    const double gamma_1_6 = {gamma_1_6};
    const double gamma_11_6 = {gamma_11_6};
    '''
)


def _kv56_gpu(z_gpu):
    """GPU implementation of K_{5/6} Bessel function using ElementwiseKernel"""
    # Check if input is real or complex
    if cp.isrealobj(z_gpu):
        # For real inputs, use the real kernel
        out_gpu = cp.zeros_like(z_gpu, dtype=cp.float64)
        kv56_real_kernel(cp.real(z_gpu), out_gpu)
    else:
        print(z_gpu)
        raise ValueError("Input must be real-valued.")
    return out_gpu

def compute_block_gpu(rho_block_gpu, L0, cst, var_term):
    """GPU implementation of block computation"""
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
        bessel_output = _kv56_gpu(cp.real(bessel_input))
        
        # Compute final values
        covariance_values = cst * u_gpu**(5/6) * cp.real(bessel_output)
        
        # Update output array
        out_gpu[mask_gpu] = covariance_values
        
    return out_gpu

def rotateWFS_gpu(px_gpu, py_gpu, rotAngleInRadians):
    """GPU version of the WFS rotation function"""
    cos_angle = math.cos(rotAngleInRadians)
    sin_angle = math.sin(rotAngleInRadians)
    
    pxx_gpu = px_gpu * cos_angle - py_gpu * sin_angle
    pyy_gpu = py_gpu * cos_angle + px_gpu * sin_angle
    
    return pxx_gpu, pyy_gpu

def covariance_matrix_gpu(*args):
    """GPU implementation of covariance matrix calculation"""
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
                out_gpu[i:i_end, j:j_end] = compute_block_gpu(
                    block_gpu, L0, cst, var_term
                )
        
        out_gpu *= fractionalR0
        return out_gpu

    # Single block processing for smaller matrices
    out_gpu = compute_block_gpu(rho_gpu, L0, cst, var_term)
    return out_gpu * fractionalR0

def create_guide_star_grid_gpu(sampling, D, rotation_angle, offset_x, offset_y):
    """GPU version of guide star grid creation"""
    # Create coordinate grids
    x_range = cp.linspace(-1, 1, sampling) * D/2
    y_range = cp.linspace(-1, 1, sampling) * D/2
    x_gpu, y_gpu = cp.meshgrid(x_range, y_range)
    
    # Flatten, rotate, and apply offsets
    x_flat, y_flat = rotateWFS_gpu(x_gpu.flatten(), y_gpu.flatten(), rotation_angle * 180/cp.pi)
    x_flat = x_flat - offset_x * D
    y_flat = y_flat - offset_y * D
    
    # Reshape back to grid
    return x_flat.reshape(sampling, sampling), y_flat.reshape(sampling, sampling)

def calculate_scaled_shifted_coords_gpu(x_gpu, y_gpu, srcACdirectionVector_gpu, gs_index, 
                                        altitude, kLayer, srcACheight):
    """GPU version of coordinate calculation"""
    # Convert NumPy arrays to CuPy if needed
    if isinstance(srcACdirectionVector_gpu, np.ndarray):
        srcACdirectionVector_gpu = cp.array(srcACdirectionVector_gpu)
    
    # Calculate beta shift
    beta = srcACdirectionVector_gpu[:, gs_index] * altitude[kLayer]
    
    # Calculate scaling factor
    scale = 1 - altitude[kLayer] / srcACheight
    
    # Calculate scaled and shifted coordinates
    return x_gpu * scale + beta[0] + 1j * (y_gpu * scale + beta[1])

def auto_correlation_gpu(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, gridMask):
    """GPU-optimized auto-correlation function"""
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
    mask_flat = mask_gpu.flatten()
    mask_indices = cp.where(mask_flat)[0]
    mask_sum = len(mask_indices)
    S_gpu = [cp.zeros((mask_sum, mask_sum)) for _ in range(len(kGs))]
    
    for k in range(len(kGs)):
        # Get the indices
        jGs, iGs = np.unravel_index(kGs[k] - 1, (nGs, nGs))
        
        buf_gpu = cp.zeros((mask_sum, mask_sum))
        
        # Create grids for guide stars
        x1_gpu, y1_gpu = create_guide_star_grid_gpu(
            sampling, D, wfsLensletsRotation[iGs], 
            wfsLensletsOffset[0, iGs], wfsLensletsOffset[1, iGs]
        )
        
        x2_gpu, y2_gpu = create_guide_star_grid_gpu(
            sampling, D, wfsLensletsRotation[jGs], 
            wfsLensletsOffset[0, jGs], wfsLensletsOffset[1, jGs]
        )
        
        for kLayer in range(nLayer):
            # Calculate coordinates
            iZ_gpu = calculate_scaled_shifted_coords_gpu(
                x1_gpu, y1_gpu, srcACdirectionVector_gpu, iGs, 
                altitude, kLayer, srcACheight
            )
            
            jZ_gpu = calculate_scaled_shifted_coords_gpu(
                x2_gpu, y2_gpu, srcACdirectionVector_gpu, jGs, 
                altitude, kLayer, srcACheight
            )
            
            # Compute covariance matrix
            out_gpu = covariance_matrix_gpu(
                iZ_gpu.T, jZ_gpu.T, r0, L0, fractionnalR0[kLayer]
            )
            
            # Directly apply mask using precomputed indices
            # This avoids shape checking and conditional logic
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

def cross_correlation_gpu(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, gridMask=None):
    """GPU-optimized cross-correlation function"""
    # Parameter extraction
    sampling = tomoParams.sampling
    
    # Handle mask
    if gridMask is None:
        mask_gpu = cp.ones((sampling, sampling), dtype=bool)
    else:
        mask_gpu = cp.array(gridMask)
    
    # Precompute mask indices for faster indexing
    mask_flat = mask_gpu.flatten()
    mask_indices = cp.where(mask_flat)[0]
    mask_sum = len(mask_indices)
    
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
    
    # Initialize a 2D list of GPU arrays
    C_gpu = [[cp.zeros((mask_sum, mask_sum)) for _ in range(nGs)] for _ in range(nSs)]
    
    for k in range(nSs*nGs):
        # Get the indices kGs and jGs
        kGs, iGs = np.unravel_index(k, (nSs, nGs))
        
        buf_gpu = cp.zeros((mask_sum, mask_sum))
        
        # Create grids for the guide star
        x1_gpu, y1_gpu = create_guide_star_grid_gpu(
            sampling, D, wfsLensletsRotation[iGs], 
            wfsLensletsOffset[0, iGs], wfsLensletsOffset[1, iGs]
        )
        
        # Create grid for the science target 
        x_range = cp.linspace(-1, 1, sampling) * D/2
        y_range = cp.linspace(-1, 1, sampling) * D/2
        x2_gpu, y2_gpu = cp.meshgrid(x_range, y_range)
        
        for kLayer in range(nLayer):
            # Calculate coordinates
            iZ_gpu = calculate_scaled_shifted_coords_gpu(
                x1_gpu, y1_gpu, srcACdirectionVector_gpu, iGs, 
                altitude, kLayer, srcACheight
            )
            
            jZ_gpu = calculate_scaled_shifted_coords_gpu(
                x2_gpu, y2_gpu, srcCCdirectionVector_gpu, kGs, 
                altitude, kLayer, srcCCheight
            )
            
            # Compute covariance matrix
            out_gpu = covariance_matrix_gpu(
                iZ_gpu.T, jZ_gpu.T, r0, L0, fractionnalR0[kLayer]
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
    
    C_gpu_array = cp.array(C_rows)
    
    return C_gpu_array
        
def build_reconstructor_gpu(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams,use_float32=False):
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
    # Set computation dtype - always use float64 for matrix construction to match CPU
    # Only use float32 for internal calculations if specified
    dtype = cp.float32 if use_float32 else cp.float64
    
    # Create a CUDA stream for asynchronous operations
    stream = cp.cuda.Stream()
    with stream:
        # Get sparse gradient matrix (CPU operation)
        Gamma, _gridMask = sparseGradientMatrixAmplitudeWeighted(
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

        # Update sampling parameter for Super Resolution
        tomoParams.sampling = _gridMask.shape[0]

        Cxx = auto_correlation_gpu(
            tomoParams,
            lgsWfsParams, 
            atmParams,
            lgsAsterismParams,
            _gridMask
        ).astype(dtype)

        # Update the tomography parameters with proper conversion
        if isinstance(tomoParams.fitSrcWeight, np.ndarray):
            tomoParams.fitSrcWeight = cp.array(tomoParams.fitSrcWeight, dtype=dtype)
        else:
            tomoParams.fitSrcWeight = cp.ones(tomoParams.nFitSrc**2, dtype=dtype)/tomoParams.nFitSrc**2

        Cox = cross_correlation_gpu(
            tomoParams,
            lgsWfsParams, 
            atmParams,
            lgsAsterismParams
        ).astype(dtype)

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

        return _reconstructor