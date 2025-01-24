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

    s1 = (1.0 - fxo)*(1.0 - fyo)
    s2 = fxo*(1.0 - fyo)
    s3 = (1.0 - fxo)*fyo
    s4 = fxo*fyo

    def sub2ind_nxo_nyo(irow, icol, nrow, ncol):
        return irow + icol*nrow

    o1 = sub2ind_nxo_nyo(oy, ox, nyo, nxo)
    o2 = sub2ind_nxo_nyo(oy, ox+1, nyo, nxo)
    o3 = sub2ind_nxo_nyo(oy+1, ox, nyo, nxo)
    o4 = sub2ind_nxo_nyo(oy+1, ox+1, nyo, nxo)

    # For each of these, some might be out of the domain if ox+1 >= nxo, etc.
    # We'll mask out the invalid ones:
    # (Though the initial range-check should handle most.)
    s1mask = s1 != 0
    s2mask = s2 != 0
    s3mask = s3 != 0
    s4mask = s4 != 0

    row1 = valid_idx[s1mask]
    col1 = o1[s1mask]
    data1 = s1[s1mask]

    row2 = valid_idx[s2mask]
    col2 = o2[s2mask]
    data2 = s2[s2mask]

    row3 = valid_idx[s3mask]
    col3 = o3[s3mask]
    data3 = s3[s3mask]

    row4 = valid_idx[s4mask]
    col4 = o4[s4mask]
    data4 = s4[s4mask]

    # Combine
    rows_all = np.concatenate([row1, row2, row3, row4])
    cols_all = np.concatenate([col1, col2, col3, col4])
    data_all = np.concatenate([data1, data2, data3, data4])

    P_coo = coo_matrix((data_all, (rows_all, cols_all)), shape=(ni, no))
    return P_coo.tocsr()

def cart2pol(x, y):
    """
    Convert Cartesian coordinates (x, y) to polar (theta, rho).
    theta is the angle in radians, rho is the radial distance.
    """
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return theta, rho