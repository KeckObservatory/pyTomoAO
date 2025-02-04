#%% 
import numpy as np

def compute_cross_covariance(s1, s2, max_shift):
    """
    Compute the cross-covariance between two 2D slope arrays for shifts along the columns (x-axis).
    
    Args:
        s1 (ndarray): Slope data for WFS 1 (2D array).
        s2 (ndarray): Slope data for WFS 2 (2D array).
        max_shift (int): Maximum number of subaperture shifts to compute.
    
    Returns:
        ndarray: Cross-covariance values for shifts 0 to max_shift.
    """
    n_rows, n_cols = s1.shape
    cross_cov = np.zeros(max_shift + 1)
    
    # Subtract mean from each subaperture's slopes
    s1 = s1 - np.mean(s1)
    s2 = s2 - np.mean(s2)
    
    for k in range(0, max_shift + 1):
        valid_cols = n_cols - k
        if valid_cols <= 0:
            cross_cov[k] = 0
            continue
        # Extract overlapping regions
        s1_part = s1[:, :valid_cols]
        s2_part = s2[:, k:k+valid_cols]
        # Compute covariance
        cross_cov[k] = np.mean(s1_part * s2_part)
    
    return cross_cov

def slodar_turbulence_profile(wfs_data, D, n_subap, theta_pairs, pair_indices, h_max=1e4, h_step=100):
    """
    Compute the turbulence profile (Cn²) using the SLODAR method.
    
    Args:
        wfs_data (list of tuples): List containing (x_slopes, y_slopes) for each WFS.
        D (float): Telescope diameter (meters).
        n_subap (int): Number of subapertures along the telescope diameter.
        theta_pairs (list of floats): Angular separations for each WFS pair (radians).
        pair_indices (list of tuples): Indices of WFS pairs corresponding to theta_pairs.
        h_max (float): Maximum altitude to consider (meters).
        h_step (float): Altitude bin size (meters).
    
    Returns:
        tuple: (h_bins, cn2_profile) where h_bins are altitude bins and cn2_profile is the normalized turbulence strength.
    """
    # Create altitude grid
    h_bins = np.arange(0, h_max + h_step, h_step)
    cn2_profile = np.zeros(len(h_bins))
    
    max_shift = n_subap - 1  # Maximum possible shift
    
    for (i, j), theta in zip(pair_indices, theta_pairs):
        # Calculate delta_h for this pair
        delta_h = D / ((n_subap - 1) * theta)
        
        # Get x and y slopes for both WFS
        s_i_x, s_i_y = wfs_data[i]
        s_j_x, s_j_y = wfs_data[j]
        
        # Compute cross-covariance for x and y slopes
        cross_cov_x = compute_cross_covariance(s_i_x, s_j_x, max_shift)
        cross_cov_y = compute_cross_covariance(s_i_y, s_j_y, max_shift)
        cross_cov = cross_cov_x + cross_cov_y  # Combine x and y contributions
        
        # Convert shift indices to altitudes
        shifts = np.arange(len(cross_cov))
        h_values = shifts * delta_h
        
        # Interpolate cross_cov onto the common altitude grid
        interp_cov = np.interp(h_bins, h_values, cross_cov, left=0, right=0)
        
        # Accumulate into the profile
        cn2_profile += interp_cov
    
    # Normalize the profile (for demonstration; calibration needed for real data)
    cn2_profile /= cn2_profile.sum()
    
    return h_bins, cn2_profile


if __name__ == "__main__":
    # Example parameters
    D = 8.0  # Telescope diameter (m)
    n_subap = 8  # Subapertures along the diameter
    theta_pairs = [np.radians(10e-3), np.radians(15e-3)]  # Angular separations in radians (e.g., 10 and 15 milliarcseconds)
    pair_indices = [(0, 1), (0, 2)]  # WFS pairs corresponding to theta_pairs

    # Simulated WFS data (x and y slopes for 3 WFSs, each with 8x8 subapertures)
    wfs_data = [
        (np.random.randn(8, 8), np.random.randn(8, 8)),  # WFS 0
        (np.random.randn(8, 8), np.random.randn(8, 8)),  # WFS 1
        (np.random.randn(8, 8), np.random.randn(8, 8)),  # WFS 2
    ]

    # Compute turbulence profile
    h_bins, cn2_profile = slodar_turbulence_profile(
        wfs_data, D, n_subap, theta_pairs, pair_indices, h_max=20000, h_step=100
    )

    # Plot the result
    import matplotlib.pyplot as plt
    plt.plot(h_bins, cn2_profile)
    plt.xlabel('Altitude (m)')
    plt.ylabel('Normalized Cn²')
    plt.title('SLODAR Turbulence Profile')
    plt.show()