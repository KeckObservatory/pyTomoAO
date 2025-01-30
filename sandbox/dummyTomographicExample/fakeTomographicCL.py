import numpy as np
from Atmosphere import LayeredAtmosphere
from DM import GaussianDM
from WFS 



class LTAOSystem:
    """
    A (very) simplified LTAO system that takes multiple wavefront sensors
    and a layered atmosphere.
    """
    def __init__(self, wavefront_sensors, atmosphere, n_dm_actuators, regularization=1e-3):
        """
        Parameters
        ----------
        wavefront_sensors : list of DummyWFS (or real WFS objects)
            The wavefront sensors in the system.
        atmosphere : LayeredAtmosphere
            Defines altitudes and Cn^2 distribution.
        n_dm_actuators : int
            Number of actuators on the DM.
        regularization : float
            Tikhonov regularization parameter.
        """
        self.wavefront_sensors = wavefront_sensors
        self.atmosphere = atmosphere
        self.n_dm_actuators = n_dm_actuators
        self.regularization = regularization
        
        # Build the tomographic interaction matrix
        self.H_tomo = self._build_tomographic_interaction_matrix()
        
        # Compute the tomographic reconstructor
        self.R_tomo = self._compute_reconstructor(self.H_tomo, self.regularization)
        
        # Initialize DM commands
        self.dm_command = np.zeros(n_dm_actuators)
    
    def get_layer_interaction_matrix(self, layer_index):
        """
        Returns the DM->slopes matrix for the wavefront sensor,
        considering an atmosphere layer at altitude h_layer_index.
        
        For each actuator 'a', we apply a small poke, propagate that
        wavefront to altitude h_layer_index, and compute the subap
        slopes. The result is a (#slopes, #actuators) matrix.
        """
        h_layer = self.altitudes[layer_index]   # altitude in meters
        n_slopes = self.n_slopes   # e.g. 2 * (#subaps) if measuring x,y slopes
        n_actuators = self.n_actuators
        
        # Create an empty matrix
        H_layer = np.zeros((n_slopes, n_actuators))
        
        # Poke amplitude (e.g. 1 nm of DM surface displacement)
        poke_amplitude = 1e-9  
        
        # For each actuator
        for a in range(n_actuators):
            # 1) Construct a DM surface array (or vector) with zero except actuator a
            dm_shape = np.zeros(n_actuators)
            dm_shape[a] = poke_amplitude
            
            # 2) Convert DM vector to a continuous wavefront in the pupil plane
            #    (If your DM is physically modeled, you might do something like
            #     dm_map = self.dm_influence_functions @ dm_shape
            #     which sums the influence function of each actuator.)
            # For now, assume a placeholder function:
            wavefront_pupil = self._build_dm_wavefront_map(dm_shape)
            
            # 3) Propagate wavefront from the DM plane to altitude h_layer
            #    This could be geometry-based or wave-optics-based.
            wavefront_layer = self._propagate_to_altitude(wavefront_pupil, h_layer)
            
            # 4) For each subaperture, compute slope
            slopes_for_this_actuator = self._compute_wfs_slopes(wavefront_layer)
            
            # 5) Store in the matrix
            H_layer[:, a] = slopes_for_this_actuator / poke_amplitude
        
        return H_layer
    
    def get_slopes(self):
        """
        Returns the measured slopes (placeholder).
        """
        return 0.01 * np.random.randn(self.n_slopes)


    def _propagate_to_altitude(self, wavefront_pupil, h_layer):
        """
        Propagate the wavefront from the DM altitude (pupil) to the layer altitude.
        Could be geometric or wave-optics. This is a placeholder.
        """
        # For geometry: might scale/shift the wavefront depending on angle.
        # For wave-optics: might do Fresnel propagation.
        return wavefront_pupil  # Over-simplified

    def _build_tomographic_interaction_matrix(self):
        """
        Build a single global interaction matrix that incorporates 
        all layers weighted by the Cn^2 profile, then stacks across all WFS.
        
        Returns
        -------
        H_tomo : np.ndarray
            Tall matrix of shape (sum of slopes of all WFS, n_dm_actuators).
        """
        # We'll build for each WFS: 
        #    H_wfs = sum_l [ w_l * H_{l,wfs} ]
        # Then stack H_wfs for all WFS into a global matrix.
        
        big_matrix_list = []
        
        for wfs in self.wavefront_sensors:
            # Sum up layer contributions
            H_sum = 0.0
            for l_idx in range(self.atmosphere.n_layers):
                w_l = self.atmosphere.weights[l_idx]
                # get interaction matrix for layer l_idx from this wfs
                H_layer = wfs.get_layer_interaction_matrix(l_idx)
                H_sum += w_l * H_layer
            big_matrix_list.append(H_sum)
        
        # Now we stack each WFS block row-wise
        H_tomo = np.vstack(big_matrix_list)
        
        print(f"H_tomo shape: {H_tomo.shape}")
        return H_tomo
    
    def _compute_reconstructor(self, H, alpha):
        """
        Compute the tomographic reconstructor with Tikhonov regularization.
        
        R = (H^T H + alpha I)^-1 H^T
        """
        # Basic approach
        n_act = H.shape[1]
        I = np.eye(n_act)
        M = H.T @ H + alpha * I
        M_inv = np.linalg.inv(M)
        R = M_inv @ H.T
        print(f"Reconstructor shape: {R.shape}")
        return R
    
    def run_closed_loop(self, n_iterations=50, loop_gain=0.5):
        """
        Run a basic integrator-based closed-loop with the tomographic reconstructor.
        """
        for i in range(n_iterations):
            # Gather slopes from each WFS
            s_list = []
            for wfs in self.wavefront_sensors:
                s_wfs = wfs.get_slopes()
                s_list.append(s_wfs)
            s_global = np.concatenate(s_list)
            
            # DM update
            dm_update = -loop_gain * (self.R_tomo @ s_global)
            self.dm_command += dm_update
            
            # "Apply" DM command (placeholder)
            self._apply_dm_command(self.dm_command)
            
            # Optional performance metric: slope RMS
            residual_rms = np.sqrt(np.mean(s_global**2))
            print(f"Iteration {i+1}/{n_iterations}: Residual RMS = {residual_rms:.4e}")
    
    def _apply_dm_command(self, command):
        """
        Placeholder for applying the DM command in your simulator or hardware.
        """
        pass


if __name__ == "__main__":
    # --------------------------------------------------------
    # Example usage of the LTAOSystem with a layered atmosphere
    # --------------------------------------------------------
    
    # Define a simplistic atmospheric profile: 3 layers
    altitudes = [0.0, 5000.0, 10000.0]  # altitudes in meters (example)
    cn2_values = [0.5, 0.3, 0.2]       # relative strengths that sum to 1.0
    
    atmosphere = LayeredAtmosphere(altitudes, cn2_values)
    
    # Create wavefront sensors
    # Suppose each WFS sees 60 slopes, and the DM has 40 actuators
    wfs1 = DummyWFS(n_slopes=60, n_actuators=40, altitudes=altitudes)
    wfs2 = DummyWFS(n_slopes=60, n_actuators=40, altitudes=altitudes)
    
    # Build the LTAO system
    ltao_system = LTAOSystem(
        wavefront_sensors=[wfs1, wfs2],
        atmosphere=atmosphere,
        n_dm_actuators=40,
        regularization=1e-3
    )
    
    # Run the closed loop
    ltao_system.run_closed_loop(n_iterations=10, loop_gain=0.3)
