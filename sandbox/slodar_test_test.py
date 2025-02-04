import numpy as np
import matplotlib.pyplot as plt

# Import your SLODAR functions (adjust the import if needed)
from slodar_test import compute_cross_covariance, slodar_turbulence_profile

# ------------------ TEST 1: Unit Test for Cross-Covariance ------------------

def test_compute_cross_covariance():
    rng = np.random.default_rng(42)
    s1 = rng.normal(size=(8, 8))
    shift_k = 3
    s2 = np.zeros_like(s1)
    
    valid_cols = s1.shape[1] - shift_k
    s2[:, shift_k:] = s1[:, :valid_cols]
    s2 += 0.05 * rng.normal(size=s2.shape)  # Add small noise

    max_shift = 7
    cross_cov = compute_cross_covariance(s1, s2, max_shift)
    recovered_shift = np.argmax(cross_cov)

    print(f"Expected shift: {shift_k}, Recovered shift: {recovered_shift}")
    assert recovered_shift == shift_k, f"Expected {shift_k}, got {recovered_shift}"
    print("✅ `compute_cross_covariance` test passed!\n")


# ------------------ TEST 2: Single-Layer Synthetic Data ------------------

def generate_single_layer_test_data(n_subap, h0, D, theta):
    rng = np.random.default_rng(1234)
    s0_x = rng.normal(size=(n_subap, n_subap))
    s0_y = rng.normal(size=(n_subap, n_subap))

    delta_h = D / ((n_subap - 1) * theta)
    k = int(round(h0 / delta_h))

    s1_x = np.zeros_like(s0_x)
    s1_y = np.zeros_like(s0_y)
    valid_cols = n_subap - k
    if valid_cols > 0:
        s1_x[:, k:] = s0_x[:, :valid_cols]
        s1_y[:, k:] = s0_y[:, :valid_cols]

    s1_x += 0.01 * rng.normal(size=s1_x.shape)
    s1_y += 0.01 * rng.normal(size=s1_y.shape)

    return (s0_x, s0_y), (s1_x, s1_y), k


def test_slodar_single_layer():
    D = 8.0
    n_subap = 20
    theta = np.radians(10e-3)
    h0 = 5000  # Known turbulence layer altitude

    wfs0, wfs1, true_shift = generate_single_layer_test_data(n_subap, h0, D, theta)
    wfs_data = [wfs0, wfs1]
    pair_indices = [(0, 1)]
    theta_pairs = [theta]

    h_bins, cn2_profile = slodar_turbulence_profile(
        wfs_data, D, n_subap, theta_pairs, pair_indices, h_max=20000, h_step=100
    )

    plt.figure()
    plt.plot(h_bins, cn2_profile, label="Recovered Cn² profile")
    plt.axvline(x=h0, color="r", linestyle="--", label="True layer altitude")
    plt.xlabel("Altitude (m)")
    plt.ylabel("Normalized Cn²")
    plt.legend()
    plt.title("Test: Single-Layer Turbulence")
    plt.show()

    recovered_h = h_bins[np.argmax(cn2_profile)]
    print(f"Expected altitude: {h0:.1f} m, Recovered peak altitude: {recovered_h:.1f} m")
    assert abs(recovered_h - h0) < 500, f"Peak altitude mismatch: Expected {h0}, got {recovered_h}"
    print("✅ Single-layer test passed!\n")


# ------------------ TEST 3: Multi-Layer Synthetic Data ------------------

def generate_multi_layer_test_data(n_subap, layers, D, theta):
    rng = np.random.default_rng(12345)
    base_s0_x = rng.normal(size=(n_subap, n_subap))
    base_s0_y = rng.normal(size=(n_subap, n_subap))

    s0_x = np.zeros_like(base_s0_x)
    s0_y = np.zeros_like(base_s0_y)
    s1_x = np.zeros_like(base_s0_x)
    s1_y = np.zeros_like(base_s0_y)

    for (h_i, str_i) in layers:
        delta_h = D / ((n_subap - 1) * theta)
        k = int(round(h_i / delta_h))

        layer_s0_x = str_i * rng.normal(size=(n_subap, n_subap))
        layer_s0_y = str_i * rng.normal(size=(n_subap, n_subap))

        s0_x += layer_s0_x
        s0_y += layer_s0_y

        layer_s1_x = np.zeros_like(layer_s0_x)
        layer_s1_y = np.zeros_like(layer_s0_y)
        valid_cols = n_subap - k
        if valid_cols > 0:
            layer_s1_x[:, k:] = layer_s0_x[:, :valid_cols]
            layer_s1_y[:, k:] = layer_s0_y[:, :valid_cols]

        s1_x += layer_s1_x
        s1_y += layer_s1_y

    return (s0_x, s0_y), (s1_x, s1_y)


def test_slodar_multi_layer():
    D = 8.0
    n_subap = 20
    theta = np.radians(10e-3)
    layers = [(3000, 1.0), (10000, 0.6)]  # Altitudes & relative strengths

    wfs0, wfs1 = generate_multi_layer_test_data(n_subap, layers, D, theta)
    wfs_data = [wfs0, wfs1]
    pair_indices = [(0, 1)]
    theta_pairs = [theta]

    h_bins, cn2_profile = slodar_turbulence_profile(
        wfs_data, D, n_subap, theta_pairs, pair_indices, h_max=20000, h_step=100
    )

    plt.figure()
    plt.plot(h_bins, cn2_profile, label="Recovered Cn² profile")
    for h_i, _ in layers:
        plt.axvline(x=h_i, color="r", linestyle="--", label=f"True layer at {h_i}m")
    plt.xlabel("Altitude (m)")
    plt.ylabel("Normalized Cn²")
    plt.legend()
    plt.title("Test: Multi-Layer Turbulence")
    plt.show()

    from scipy.signal import find_peaks

    # Find all local maxima in the cn2_profile
    peak_indices, _ = find_peaks(cn2_profile)

    # Check if each true layer altitude has a corresponding peak within 700 meters
    for h_i, _ in layers:
        # Convert true layer altitude to index
        true_index = np.searchsorted(h_bins, h_i)
        
        # Find the closest peak to the true layer index
        closest_peak_index = min(peak_indices, key=lambda x: abs(x - true_index))
        recovered_h = h_bins[closest_peak_index]
        
        # Print the recovered height
        print(f"Recovered height for true layer at {h_i}m: {recovered_h}m")
        
        assert abs(recovered_h - h_i) < 700, f"Peak mismatch: Expected {h_i}, got {recovered_h}"
    print("✅ Multi-layer test passed!\n")


# ------------------ TEST 4: Edge Cases ------------------

def test_normalization():
    D = 8.0
    n_subap = 20
    theta = np.radians(10e-3)
    h0 = 5000

    wfs0, wfs1, _ = generate_single_layer_test_data(n_subap, h0, D, theta)
    wfs_data = [wfs0, wfs1]
    pair_indices = [(0, 1)]
    theta_pairs = [theta]

    h_bins, cn2_profile = slodar_turbulence_profile(
        wfs_data, D, n_subap, theta_pairs, pair_indices, h_max=20000, h_step=100
    )

    assert np.isclose(cn2_profile.sum(), 1.0), "Cn² profile not normalized!"
    print("✅ Normalization test passed!\n")


# ------------------ RUN ALL TESTS ------------------

if __name__ == "__main__":
    test_compute_cross_covariance()
    test_slodar_single_layer()
    test_slodar_multi_layer()
    test_normalization()
