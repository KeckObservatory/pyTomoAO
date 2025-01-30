#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
class GaussianDM:
    """
    A simple class that defines:
      - A DM layout on a 17x17 grid.
      - Gaussian influence functions for each actuator.
      - Methods to build wavefront maps from actuator commands.
      - A plotting method to inspect the sum of influences.
    """
    def __init__(self, n_side=17, diameter=1.0, sigma=0.1):
        """
        Parameters
        ----------
        n_side : int
            Number of actuators along one dimension (17 means 17x17).
        diameter : float
            Physical diameter of the DM or pupil, in some units (e.g., meters).
        sigma : float
            Standard deviation of each actuator's Gaussian influence, in same units as 'diameter'.
        """
        self.n_side = n_side
        self.res = n_side*10
        self.diameter = diameter
        self.sigma = sigma
        
        # 2) Create the 2D coordinates for the pupil plane (same resolution as n_side x n_side),
        #    also flattened. We'll store them in self.grid_coords (shape: (n_points, 2)).
        self.grid_coords = self._compute_dm_layout(self.res, diameter)
        self.n_points = self.grid_coords.shape[0]
        
        # 1) Create the 2D coordinates for each actuator, flattened
        #    We'll store them in self.actuator_coords (shape: (n_act, 2)).
        self.actuator_coords = self._compute_dm_layout(self.n_side, diameter)
        self.actuator_coords = np.array([ac for ac in self.actuator_coords if np.linalg.norm(ac) < self.diameter/2])
        self.n_actuators = self.actuator_coords.shape[0]

        # 3) Precompute the influence function matrix, shape (n_points, n_actuators)
        self.dm_influence_functions = self._compute_influence_functions(
            self.actuator_coords,
            self.grid_coords,
            sigma
        )

    def _compute_dm_layout(self, n_side, diameter):
        """
        Returns a flattened list of (x,y) coordinates for an n_side x n_side grid,
        ranging from -diameter/2 to +diameter/2 in each axis.
        
        shape: (n_side^2, 2)
        """
        # Create a linear space from -d/2 to +d/2
        x_vals = np.linspace(-diameter/2, diameter/2, n_side)
        y_vals = -1*np.linspace(-diameter/2, diameter/2, n_side)
        
        # Create a 2D mesh
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Flatten to shape (n_side^2, 2)
        coords = np.vstack((X.flatten(),Y.flatten()))
        coords = np.moveaxis(coords,0,1)
        return coords

    def _compute_influence_functions(self, actuator_coords, grid_coords, sigma):
        """
        Build the matrix of Gaussian influence functions. For each grid point p,
        for each actuator a, we compute:
        
        IF[p,a] = exp( -((x_p - x_a)^2 + (y_p - y_a)^2) / (2 * sigma^2) ).
        
        The result is a matrix of shape (n_points, n_actuators).
        """
        n_points = grid_coords.shape[0]
        n_acts = actuator_coords.shape[0]
        
        infl_funcs = np.zeros((n_points, n_acts), dtype=float)
        
        for a in range(n_acts):
            x_a, y_a = actuator_coords[a]
            # Vectorized distance from actuator a to all grid points
            dx = grid_coords[:, 0] - x_a
            dy = grid_coords[:, 1] - y_a
            r2 = dx**2 + dy**2
            
            infl_funcs[:, a] = np.exp(-0.5 * r2 / sigma**2)
        
        return infl_funcs

    def build_dm_wavefront_map(self, dm_shape):
        """
        Given a vector of actuator commands (length = n_actuators),
        produce the wavefront map on the n_side x n_side grid.
        
        wavefront_map_1d = dm_influence_functions @ dm_shape
        then reshape to (n_side, n_side).
        """
        if len(dm_shape) != self.n_actuators:
            raise ValueError("dm_shape must be length of n_actuators")
        
        wavefront_1d = self.dm_influence_functions @ dm_shape
        wavefront_2d = wavefront_1d.reshape(self.res, self.res)
        return wavefront_2d

    def plot_influence_sum(self):
        """
        Plot the sum of all influence functions (i.e., if every actuator is set to 1).
        This can be useful for inspecting coverage or overlap of the actuator array.
        """
        # sum across actuators
        summed_influence_1d = np.sum(self.dm_influence_functions, axis=1)
        summed_influence_2d = summed_influence_1d.reshape(self.res, self.res)
        
        plt.figure(figsize=(10,10))
        im = plt.imshow(summed_influence_2d, cmap = 'plasma', alpha = 0.7, origin='lower', extent=[
            -self.diameter/2, self.diameter/2,
            -self.diameter/2, self.diameter/2
        ])
        plt.plot(dm.actuator_coords[:,0], dm.actuator_coords[:,1], "x", color = 'red')
        plt.colorbar(im, label="Sum of all actuator influences")
        plt.title("Summed Gaussian Influence Functions")
        plt.xlabel("x coordinate")
        plt.ylabel("y coordinate")
        plt.show()

    def plot_wavefront(self, dm_shape):

        # Build the wavefront map for these commands
        wavefront_map = dm.build_dm_wavefront_map(dm_shape)
        
        # Plot the resulting wavefront map
        plt.figure(figsize=(10,10))
        plt.imshow(wavefront_map, cmap = 'plasma', origin='lower', extent=[
            -dm.diameter/2, dm.diameter/2, -dm.diameter/2, dm.diameter/2
        ])
        plt.colorbar(label="Wavefront amplitude (arbitrary units)")
        plt.title("Example DM Wavefront Map (random commands)")
        plt.xlabel("x coordinate")
        plt.ylabel("y coordinate")
        plt.show()


#%%
# ------------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # Instantiate the DM with a 17x17 grid, diameter 1.0, and sigma=0.1
    dm = GaussianDM(n_side=17, diameter=1.0, sigma=0.025)
    
    # Plot the sum of all actuator influences
    dm.plot_influence_sum()
    
    # Example: create a random set of actuator commands
    dm_shape = np.random.randn(dm.n_actuators)
    
    dm.plot_wavefront(dm_shape)
# %%
