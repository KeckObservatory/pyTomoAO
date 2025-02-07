import numpy as np
from scipy.sparse import spdiags, coo_matrix, csr_matrix

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

def cart2pol(x, y):
    """
    Convert Cartesian coordinates (x, y) to polar (theta, rho).
    theta is the angle in radians, rho is the radial distance.
    """
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return theta, rho


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
        s0x = np.array([-1/16, -3/16, -1/2, -3/16, -1/16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/16, 3/16, 1/2, 3/16, 1/16]) * (1/dsa) # x stencil weight
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


import numpy as np
import math

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
