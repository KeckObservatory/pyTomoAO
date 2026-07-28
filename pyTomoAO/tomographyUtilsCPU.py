import math

import numba as nb
import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import block_diag
from scipy.special import gamma

# Constants shared by the complex and real K_{5/6} kernels below. They are module-level so
# that the two implementations cannot drift apart -- a coefficient that disagreed between
# copies of this expansion is exactly what issue #97 turned out to be. Numba folds
# module-level floats in as compile-time constants, so this costs nothing at runtime.
#
# Gamma values are given to full double precision: the series expansion subtracts two
# quantities that grow like exp(z), and that cancellation amplifies any truncation in the
# leading constants into the dominant error term.
_GAMMA_1_6 = 5.566316001780236  # Gamma(1/6)
_GAMMA_11_6 = 0.94065585825677167  # Gamma(11/6)
_NU = 5.0 / 6.0

# Crossover between the two expansions. The series stays accurate to ~2e-8 out to z ~ 9 and
# converges in 24 iterations there, whereas the asymptotic expansion is not converged below
# z ~ 8. Switching at 2.0 cost seven digits over the range carrying most pupil baselines.
_KV56_CROSSOVER = 9.0

# Asymptotic expansion coefficients, a_k = a_{k-1}*(4v^2 - (2k-1)^2)/(8k) for v = 5/6.
_A1 = 2.0 / 9.0
_A2 = -7.0 / 81.0
_A3 = 175.0 / 2187.0
_A4 = -2275.0 / 19683.0
_A5 = 40040.0 / 177147.0
_A6 = -2662660.0 / 4782969.0
_A7 = 71131060.0 / 43046721.0
_A8 = -2222845625.0 / 387420489.0


@nb.njit(nb.float64(nb.float64), cache=False)
def _kv56_real_scalar(z):
    """K_{5/6}(z) for a real, positive argument.

    The hot path. `_compute_block` feeds this real distances and uses the result directly,
    so routing through the complex kernel would double the arithmetic and force a
    complex128 copy of the input for no benefit.
    """
    gamma_1_6 = _GAMMA_1_6
    gamma_11_6 = _GAMMA_11_6
    v = _NU
    z_abs = abs(z)
    if z_abs < _KV56_CROSSOVER:
        term_a = (0.5 * z) ** v / gamma_11_6
        term_b = (0.5 * z) ** -v / gamma_1_6
        sum_a = term_a
        sum_b = term_b
        z_sq_over_4 = (0.5 * z) ** 2
        k = 1
        tol = 1e-15
        for _ in range(1000):
            term_a *= z_sq_over_4 / (k * (k + v))
            sum_a += term_a
            term_b *= z_sq_over_4 / (k * (k - v))
            sum_b += term_b
            if abs(term_a) < tol * abs(sum_a) and abs(term_b) < tol * abs(sum_b):
                break
            k += 1
        return np.pi * (sum_b - sum_a)
    z_inv = 1.0 / z
    # Horner's method: fewer multiplications than the explicit powers.
    sum_terms = 1.0 + z_inv * (
        _A1
        + z_inv
        * (
            _A2
            + z_inv
            * (_A3 + z_inv * (_A4 + z_inv * (_A5 + z_inv * (_A6 + z_inv * (_A7 + z_inv * _A8)))))
        )
    )
    return np.sqrt(np.pi / (2.0 * z)) * np.exp(-z) * sum_terms


@nb.vectorize([nb.float64(nb.float64)], nopython=True, target="parallel")
def _kv56_real(z):
    """Modified Bessel function K_{5/6}(z) for real numpy arrays."""
    return _kv56_real_scalar(z)


@nb.njit(nb.complex128(nb.complex128), cache=False)
def _kv56_scalar(z):
    """Scalar implementation used as kernel for array version"""
    gamma_1_6 = _GAMMA_1_6
    gamma_11_6 = _GAMMA_11_6
    v = _NU
    z_abs = np.abs(z)
    if z_abs < _KV56_CROSSOVER:
        # Series expansion for small |z|
        sum_a = 0.0j
        sum_b = 0.0j
        term_a = (0.5 * z) ** v / gamma_11_6
        term_b = (0.5 * z) ** -v / gamma_1_6
        sum_a += term_a
        sum_b += term_b
        z_sq_over_4 = (0.5 * z) ** 2
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
        # Asymptotic expansion for large |z|; coefficients shared with the real kernel.
        z_inv = 1.0 / z
        sum_terms = (
            1.0
            + _A1 * z_inv
            + _A2 * z_inv**2
            + _A3 * z_inv**3
            + _A4 * z_inv**4
            + _A5 * z_inv**5
            + _A6 * z_inv**6
            + _A7 * z_inv**7
            + _A8 * z_inv**8
        )
        prefactor = np.sqrt(np.pi / (2.0 * z)) * np.exp(-z)
        K = prefactor * sum_terms
    return K


# Vectorized version outside the class
@nb.vectorize(
    [
        nb.complex128(nb.complex128),  # Complex input
        nb.complex128(nb.float64),
    ],  # Real input
    nopython=True,
    target="parallel",
)
def _kv56(z):
    """
    Modified Bessel function K_{5/6}(z) for numpy arrays
    Handles both real and complex inputs efficiently
    """
    return _kv56_scalar(z)


def _rotateWFS(px, py, rotAngleInRadians):
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
    pyy = py * math.cos(rotAngleInRadians) + px * math.sin(rotAngleInRadians)
    return pxx, pyy


def _create_guide_star_grid(sampling, D, rotation_angle, offset_x, offset_y):
    """
    Create a grid of guide star positions based on the specified parameters.

    Parameters:
    -----------
        sampling (int): Number of samples in each dimension for the grid.
        D (float): Diameter of the telescope, used to scale the grid.
        rotation_angle (float): Angle to rotate the grid, in radians.
        offset_x (float): Offset in the x-direction to apply to the grid.
        offset_y (float): Offset in the y-direction to apply to the grid.

    Returns:
    --------
        tuple: Two 2D arrays representing the x and y coordinates of the guide stars.
    """

    # Create a grid of points in Cartesian coordinates
    x, y = np.meshgrid(np.linspace(-1, 1, sampling) * D / 2, np.linspace(-1, 1, sampling) * D / 2)

    # Flatten the grid, rotate the positions, and apply the specified offsets.
    # wfsLensletsRotation is specified in radians and _rotateWFS expects radians, so the
    # angle is passed straight through.
    x, y = _rotateWFS(x.flatten(), y.flatten(), rotation_angle)
    x = x - offset_x * D  # Apply x offset
    y = y - offset_y * D  # Apply y offset

    # Reshape the modified coordinates back to the original grid shape
    return x.reshape(sampling, sampling), y.reshape(sampling, sampling)


def _calculate_scaled_shifted_coords(
    x, y, srcACdirectionVector, gs_index, altitude, kLayer, srcACheight
):
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


def _compute_block(rho_block, L0, cst, var_term, zero_tol=0.0):
    """
    Vectorized computation of covariance values for a matrix block

    Separations at or below ``zero_tol`` are treated as coincident points and take the
    variance term. See :func:`_covariance_matrix` for why an exact test is not enough.
    """
    # Initialize output with variance term
    out = np.full(rho_block.shape, var_term, dtype=np.float64)
    # Find non-zero distances and compute covariance
    mask = rho_block > zero_tol
    u = (2 * np.pi * rho_block[mask]) / L0
    # Real-valued Bessel kernel: the distances are real and the result is used directly,
    # so the complex overload would only add a complex128 copy of u and twice the work.
    out[mask] = cst * u ** (5 / 6) * _kv56_real(u)
    return out


def _covariance_matrix(*args):
    """
    Optimized phase covariance matrix calculation using the Von Karman turbulence model.

    Parameters
    ----------
    *args : tuple
        Either ``(rho1, r0, L0, fractionalR0)`` for the auto-covariance, or
        ``(rho1, rho2, r0, L0, fractionalR0)`` for the cross-covariance, where:

        - ``rho1``, ``rho2`` : complex coordinate arrays (x + iy)
        - ``r0`` : Fried parameter [m]
        - ``L0`` : outer scale [m]
        - ``fractionalR0`` : turbulence layer weighting factor

    Returns
    -------
    numpy.ndarray
        Covariance matrix with the same dimensions as the input coordinates.

    Raises
    ------
    ValueError
        If the number of positional arguments is neither 4 nor 5.
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
    GAMMA_6_5 = gamma(6 / 5)
    GAMMA_11_6 = gamma(11 / 6)
    GAMMA_5_6 = gamma(5 / 6)

    # Base constant components
    BASE_CONST = (24 * GAMMA_6_5 / 5) ** (5 / 6)
    SCALE_FACTOR = GAMMA_11_6 / (2 ** (5 / 6) * np.pi ** (8 / 3))

    # L0/r0 ratio raised to 5/3 power
    L0_r0_ratio = (L0 / r0) ** (5 / 3)

    # Final constant for non-zero distances
    cst = BASE_CONST * SCALE_FACTOR * L0_r0_ratio

    # Variance term for zero distances (r=0 case)
    var_term = (BASE_CONST * GAMMA_11_6 * GAMMA_5_6 / (2 * np.pi ** (8 / 3))) * L0_r0_ratio

    # ==================================================================
    # Calculate pairwise distances
    # ==================================================================
    # Vectorized distance calculation using broadcasting
    rho = np.abs(rho1[:, np.newaxis] - rho2)
    n, m = rho.shape

    # Coincident points must take the variance term. The two coordinate grids are built by
    # different arithmetic, so a separation that is mathematically zero can evaluate to a few
    # ULPs instead; an exact `!= 0` test would send those into the Bessel branch and return a
    # badly wrong value for the largest entries of the matrix.
    zero_tol = 1e-12 * rho.max()

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
                out[i:i_end, j:j_end] = _compute_block(block, L0, cst, var_term, zero_tol)

        # Apply fractional weighting
        out *= fractionalR0
        return out

    # Single block processing for smaller matrices
    out = _compute_block(rho, L0, cst, var_term, zero_tol)
    return out * fractionalR0


def _auto_correlation(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, gridMask):
    """
    Computes the auto-correlation meta-matrix for tomographic atmospheric reconstruction.

    Parameters
    ----------
    tomoParams : object
        Tomography parameters:

        - ``sampling`` (int): number of grid samples per axis
        - ``mask`` (ndarray): 2D boolean grid mask

    lgsWfsParams : object
        LGS WFS parameters:

        - ``D`` (float): telescope diameter [m]
        - ``wfsLensletsRotation`` (ndarray): lenslet rotations [rad]
        - ``wfsLensletsOffset`` (ndarray): lenslet offsets [normalized]

    atmParams : object
        Atmospheric parameters:

        - ``nLayer`` (int): number of turbulence layers
        - ``altitude`` (ndarray): layer altitudes [m]
        - ``r0`` (float): Fried parameter [m]
        - ``L0`` (float): outer scale [m]
        - ``fractionnalR0`` (ndarray): turbulence strength per layer

    lgsAsterismParams : object
        LGS constellation parameters:

        - ``nLGS`` (int): number of LGS
        - ``directionVectorLGS`` (ndarray): direction vectors
        - ``LGSheight`` (float): LGS height [m]

    gridMask : ndarray
        2D boolean mask for valid grid points.

    Returns
    -------
    numpy.ndarray
        Auto-correlation meta-matrix of shape ``(nGs*valid_pts, nGs*valid_pts)``.
    """
    # print("-->> Computing auto-correlation meta-matrix <<--\n")
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

    # Generate indices for the upper triangular part of the matrix
    kGs = np.triu(np.arange(1, nGs**2 + 1).reshape(nGs, nGs).T, 1).T.reshape(nGs**2)
    kGs[0] = 1
    kGs = kGs[kGs != 0]

    # The covariance was previously evaluated over the full sampling x sampling grid and
    # then cut down to the valid points, discarding ~71% of the Bessel evaluations on the
    # function that dominates runtime. Masking the coordinates first is exactly equivalent:
    # the rows and columns of `out` are indexed by iZ.T.flatten() and jZ.T.flatten(), so
    # selecting the same entries from those vectors selects the same pairs. The mask does
    # not depend on the layer, so it is applied once, outside the loop.
    mask_flat = mask.flatten()
    S = [None] * len(kGs)

    for k in range(len(kGs)):
        # Get the indices iGs and jGs from the index kGs(k)
        jGs, iGs = np.unravel_index(kGs[k] - 1, (nGs, nGs))  # Adjust for 0-based index in Python

        buf = 0

        # Create grids for the first and second guide stars, keeping only valid points
        x1, y1 = _create_guide_star_grid(
            sampling,
            D,
            wfsLensletsRotation[iGs],
            wfsLensletsOffset[0, iGs],
            wfsLensletsOffset[1, iGs],
        )
        x2, y2 = _create_guide_star_grid(
            sampling,
            D,
            wfsLensletsRotation[jGs],
            wfsLensletsOffset[0, jGs],
            wfsLensletsOffset[1, jGs],
        )
        x1, y1 = x1.T.flatten()[mask_flat], y1.T.flatten()[mask_flat]
        x2, y2 = x2.T.flatten()[mask_flat], y2.T.flatten()[mask_flat]

        for kLayer in range(nLayer):
            # Calculate the scaled and shifted coordinates for the first and second guide stars
            iZ = _calculate_scaled_shifted_coords(
                x1, y1, srcACdirectionVector, iGs, altitude, kLayer, srcACheight
            )
            jZ = _calculate_scaled_shifted_coords(
                x2, y2, srcACdirectionVector, jGs, altitude, kLayer, srcACheight
            )

            # Compute the covariance matrix over the valid points only
            buf = buf + _covariance_matrix(iZ, jZ, r0, L0, fractionnalR0[kLayer])

        S[k] = buf.T

    # Rearrange the results into a full nGs x nGs matrix. The blocks not covered by kGs or
    # the diagonal stay zero and are discarded by the symmetrisation below, so they can all
    # share one read-only zero block rather than each allocating their own.
    buf = S
    zero_block = np.zeros((np.sum(mask), np.sum(mask)))
    S_tmp = [zero_block] * (nGs**2)
    for c, i in enumerate(kGs):
        S_tmp[i - 1] = buf[c]

    # If you want these as a 1D array of indices
    diagonal_indices_1d = np.diag_indices(nGs)[0] * nGs + np.diag_indices(nGs)[1]

    for i in diagonal_indices_1d:
        S_tmp[i] = S_tmp[0]

    S_tmp = np.stack(S_tmp, axis=0)
    S = (
        S_tmp.reshape(nGs, nGs, np.sum(mask), np.sum(mask))
        .transpose(0, 2, 1, 3)
        .reshape(nGs * np.sum(mask), nGs * np.sum(mask))
    )

    # Make the matrix symmetric
    return np.tril(S) + np.triu(S.T, 1)


def _cross_correlation(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, gridMask=None):
    """
    Computes the cross-correlation meta-matrix for tomographic atmospheric reconstruction.

    Parameters
    ----------
    tomoParams : object
        Tomography parameters:

        - ``sampling`` (int): number of grid samples per axis
        - ``mask`` (ndarray): 2D boolean grid mask

    lgsWfsParams : object
        LGS WFS parameters:

        - ``D`` (float): telescope diameter [m]
        - ``wfsLensletsRotation`` (ndarray): lenslet rotations [rad]
        - ``wfsLensletsOffset`` (ndarray): lenslet offsets [normalized]

    atmParams : object
        Atmospheric parameters:

        - ``nLayer`` (int): number of turbulence layers
        - ``altitude`` (ndarray): layer altitudes [m]
        - ``r0`` (float): Fried parameter [m]
        - ``L0`` (float): outer scale [m]
        - ``fractionnalR0`` (ndarray): turbulence strength per layer

    lgsAsterismParams : object
        LGS constellation parameters:

        - ``nLGS`` (int): number of LGS
        - ``directionVectorLGS`` (ndarray): direction vectors
        - ``LGSheight`` (float): LGS height [m]

    gridMask : ndarray, optional
        2D boolean mask for valid grid points.

    Returns
    -------
    numpy.ndarray
        Cross-correlation meta-matrix of shape ``(nGs*valid_pts, nGs*valid_pts)``.
    """
    # print("-->> Computing cross-correlation meta-matrix <<--\n")
    # ======================================================================
    # Parameter Extraction
    # ======================================================================
    # Tomography parameters
    try:
        tomoParams.sampling = gridMask.shape[0]
    except AttributeError:
        # No grid mask supplied; fall back to the default sampling.
        tomoParams.sampling = 49

    sampling = tomoParams.sampling

    mask = np.ones((sampling, sampling), dtype=bool) if gridMask is None else gridMask

    nSs = tomoParams.nFitSrc**2
    srcCCdirectionVector = tomoParams.directionVectorSrc
    srcCCheight = tomoParams.fitSrcHeight

    # LGS constellation parameters
    nGs = lgsAsterismParams.nLGS
    srcACdirectionVector = lgsAsterismParams.directionVectorLGS
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

    # Placeholders: every entry is overwritten below before it is read, so allocating
    # sampling**2 x sampling**2 zeros here was 46 MB per entry of pure page-faulting.
    C = [[None] * nGs for _ in range(nSs)]

    # As in _auto_correlation, the pupil mask is applied to the coordinates rather than to
    # the finished covariance matrix, and is hoisted out of the layer loop.
    mask_flat = mask.flatten()

    for k in range(nSs * nGs):
        # Get the indices kGs and jGs
        kGs, iGs = np.unravel_index(k, (nSs, nGs))

        buf = 0

        # Create grids for the first and second guide stars, keeping only valid points
        x1, y1 = _create_guide_star_grid(
            sampling,
            D,
            wfsLensletsRotation[iGs],
            wfsLensletsOffset[0, iGs],
            wfsLensletsOffset[1, iGs],
        )

        x2, y2 = np.meshgrid(
            np.linspace(-1, 1, sampling) * D / 2, np.linspace(-1, 1, sampling) * D / 2
        )

        x1, y1 = x1.T.flatten()[mask_flat], y1.T.flatten()[mask_flat]
        x2, y2 = x2.T.flatten()[mask_flat], y2.T.flatten()[mask_flat]

        for kLayer in range(nLayer):
            # Calculate the scaled and shifted coordinates for the first and second guide stars
            iZ = _calculate_scaled_shifted_coords(
                x1, y1, srcACdirectionVector, iGs, altitude, kLayer, srcACheight
            )
            jZ = _calculate_scaled_shifted_coords(
                x2, y2, srcCCdirectionVector, kGs, altitude, kLayer, srcCCheight
            )

            # Compute the covariance matrix over the valid points only
            buf = buf + _covariance_matrix(iZ, jZ, r0, L0, fractionnalR0[kLayer])

        C[kGs][iGs] = buf.T

    # Rearrange the results into a single array
    return np.array([np.concatenate(row, axis=1) for row in C])


def _sparseGradientMatrixAmplitudeWeighted(
    validLenslet, amplMask=None, overSampling=2, stencilSize=3
):
    """
    Computes the sparse gradient matrix (3x3 or 5x5 stencil) with amplitude mask.

    Parameters
    ----------
    validLenslet : numpy.ndarray
        2D valid lenslet map.
    amplMask : numpy.ndarray, optional
        2D amplitude weight mask. Defaults to uniform weighting.
    overSampling : int, optional
        Oversampling factor for the gridMask, either 2 or 4 (default is 2).

    Returns
    -------
    Gamma : scipy.sparse.csr_matrix
        Sparse gradient matrix.
    gridMask : numpy.ndarray
        2D mask used for the reconstructed phase.
    """
    # print("-->> Computing sparse gradient matrix <<--\n")

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

        s0x = np.array([-1 / 4, -1 / 2, -1 / 4, 0, 0, 0, 1 / 4, 1 / 2, 1 / 4])  # x-gradient weights
        s0y = -np.array(
            [1 / 4, 0, -1 / 4, 1 / 2, 0, -1 / 2, 1 / 4, 0, -1 / 4]
        )  # y-gradient weights
        num_points = 9
    elif stencilSize == 5:
        # 5x5 stencil for 4x oversampling

        s0x = np.array(
            [
                -1 / 16,
                -3 / 16,
                -1 / 2,
                -3 / 16,
                -1 / 16,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1 / 16,
                3 / 16,
                1 / 2,
                3 / 16,
                1 / 16,
            ]
        )  # x-gradient weights
        s0y = s0x.reshape(5, 5).T.flatten()  # y-gradient weights (transpose of x)
        num_points = 25
    else:
        raise ValueError("overSampling must be 2 or 4")

    # Initialize stencil position arrays
    i0x = np.tile(np.arange(1, stencilSize + 1), stencilSize)  # Row indices
    j0x = np.repeat(np.arange(1, stencilSize + 1), stencilSize)  # Column indices
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
    iMap0, jMap0 = np.meshgrid(np.arange(1, stencilSize + 1), np.arange(1, stencilSize + 1))
    gridMask = np.zeros((nMap, nMap), dtype=bool)
    u = np.arange(1, num_points + 1)  # Counter for filling arrays

    # Build sparse matrix by iterating over lenslets
    for jLenslet in range(1, nLenslet + 1):
        jOffset = overSampling * (jLenslet - 1)  # Column offset in oversampled grid
        for iLenslet in range(1, nLenslet + 1):
            if validLenslet[iLenslet - 1, jLenslet - 1]:  # Only process valid lenslets
                # Calculate indices in amplitude mask
                iAmpl = (iLenslet - 1) * overSampling + 1
                jAmpl = (jLenslet - 1) * overSampling + 1

                # Check if amplitude mask is valid for this lenslet
                if (
                    np.sum(
                        amplMask[iAmpl - 1 : iAmpl + overSampling, jAmpl - 1 : jAmpl + overSampling]
                    )
                    == (overSampling + 1) ** 2
                ):
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
    import numpy as np
    from scipy.sparse import csr_matrix

    indx = np.ravel_multi_index((i_x.astype(int) - 1, j_x.astype(int) - 1), (nMap, nMap), order="F")
    indy = np.ravel_multi_index((i_y.astype(int) - 1, j_y.astype(int) - 1), (nMap, nMap), order="F")
    v = np.tile(np.arange(1, 2 * nValidLenslet_ + 1), (u.size, 1)).T

    # Construct final sparse gradient matrix
    Gamma = csr_matrix(
        (np.concatenate((s_x, s_y)), (v.flatten() - 1, np.concatenate((indx, indy)))),
        shape=(2 * nValidLenslet_, nMap**2),
    )
    Gamma = Gamma[:, gridMask.ravel()]  # Apply mask to reduce matrix size

    return Gamma, gridMask


def _solve_spd_from_right(B, A):
    """Return ``B @ inv(A)`` without forming the inverse, for symmetric positive definite A.

    ``A`` here is the regularised slope covariance ``Gamma Cxx Gamma^T + Cn``, which is
    symmetric positive definite by construction. Solving is both cheaper than an explicit
    inverse and better conditioned: forming ``inv(A)`` and multiplying squares the condition
    number's effect on the result.

    ``X = B inv(A)`` means ``X A = B``; transposing and using ``A = A^T`` gives
    ``A X^T = B^T``, so a single Cholesky solve suffices.

    Parameters
    ----------
    B : numpy.ndarray
        Right-hand side, shape ``(m, n)``.
    A : numpy.ndarray
        Symmetric positive definite matrix, shape ``(n, n)``.

    Returns
    -------
    numpy.ndarray
        ``B @ inv(A)``, shape ``(m, n)``.
    """
    try:
        return cho_solve(cho_factor(A, lower=True, check_finite=False), B.T, check_finite=False).T
    except LinAlgError:
        # Regularisation should keep A positive definite; fall back rather than fail if a
        # pathological configuration makes it indefinite.
        return np.linalg.solve(A, B.T).T


def _build_reconstructor_model(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, alpha=1):
    """
    Build the model-based tomographic reconstructor on the CPU.

    Parameters
    ----------
    tomoParams, lgsWfsParams, atmParams, lgsAsterismParams : object
        Configuration objects held by the reconstructor.
    alpha : float, optional
        Regularization weight applied to the inversion (default is 1).

    Returns
    -------
    tuple
        ``(reconstructor, Gamma, gridMask, Cxx, Cox, Cnz, RecStatSA)``.
    """
    Gamma, gridMask = _sparseGradientMatrixAmplitudeWeighted(
        lgsWfsParams.validLLMapSupport, amplMask=None, overSampling=2
    )
    Gamma = block_diag([Gamma] * lgsAsterismParams.nLGS)

    # Update sampling parameter for Super Resolution
    tomoParams.sampling = gridMask.shape[0]

    Cxx = _auto_correlation(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, gridMask)

    # Update the tomography parameters to include the fitting weight for each source
    tomoParams.fitSrcWeight = np.ones(tomoParams.nFitSrc**2) / tomoParams.nFitSrc**2

    Cox = _cross_correlation(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, gridMask)

    # print(Cox.shape)

    CoxOut = 0
    for i in range(tomoParams.nFitSrc**2):
        CoxOut = CoxOut + Cox[i, :, :] * tomoParams.fitSrcWeight[i]

    # row_mask = gridMask.ravel().astype(bool)
    # col_mask = np.tile(gridMask.ravel().astype(bool), lgsAsterismParams.nLGS)

    # # Select submatrix using boolean masks with np.ix_ for correct indexing
    # Cox = CoxOut[np.ix_(row_mask, col_mask)]
    Cox = CoxOut

    Css = Gamma @ Cxx @ Gamma.T
    CnZ = np.eye(Gamma.shape[0]) * alpha * np.mean(np.diag(Css))

    RecStatSA = _solve_spd_from_right(Cox @ Gamma.T, Css + CnZ)

    # LGS WFS subapertures diameter
    d = lgsWfsParams.DSupport / lgsWfsParams.validLLMapSupport.shape[0]

    # Size of the pixel at Shannon sampling
    _wavefront2Meter = lgsAsterismParams.LGSwavelength / d / 2

    # Compute final scaled reconstructor
    _reconstructor = d * _wavefront2Meter * RecStatSA

    return _reconstructor, Gamma, gridMask, Cxx, Cox, CnZ, RecStatSA


def _build_reconstructor_im(
    IM, tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, dmParams, alpha=1
):
    """
    Build the interaction-matrix-based tomographic reconstructor on the CPU.

    Parameters
    ----------
    IM : numpy.ndarray
        Block-diagonal interaction matrix, one block per wavefront sensor.
    tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, dmParams : object
        Configuration objects held by the reconstructor.
    alpha : float, optional
        Regularization weight applied to the inversion (default is 1).

    Returns
    -------
    tuple
        ``(reconstructor, gridMask, Cxx, Cox, Cnz, RecStatSA)``.
    """
    # IM has to be a block diagonal matrix containing the IM for each LGS

    # Define gridMask based on the DM parameters
    gridMask = dmParams.validActuators

    # Update sampling parameter for Super Resolution
    tomoParams.sampling = gridMask.shape[0]

    Cxx = _auto_correlation(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, gridMask)

    # Update the tomography parameters to include the fitting weight for each source
    tomoParams.fitSrcWeight = np.ones(tomoParams.nFitSrc**2) / tomoParams.nFitSrc**2

    Cox = _cross_correlation(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, gridMask)

    Cox = np.squeeze(Cox)

    # Noise covariance matrix
    weight = np.ones(IM.shape[0])
    CnZ = alpha * np.diag(1 / (weight.flatten(order="F")))

    RecStatSA = _solve_spd_from_right(Cox @ IM.T, IM @ Cxx @ IM.T + CnZ)

    # Compute final scaled reconstructor
    _reconstructor = RecStatSA

    return _reconstructor, gridMask, Cxx, Cox, CnZ, RecStatSA
