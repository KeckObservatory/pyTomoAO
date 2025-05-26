# reconstructor_revolt.py

# %%
import numpy as np
import matplotlib.pyplot as plt
from pyTomoAO.tomographicReconstructor import tomographicReconstructor

# %%
# Create the reconstructor
reconstructor = tomographicReconstructor("reconstructor_config_revolt.yaml")

# Build the model based reconstructor. To build the IM based reconstructor,
# pass the IM matrix as an argument.
# R = reconstructor.build_reconstructor(IM, use_float32=True) 
reconstructor.build_reconstructor(alpha=0.01)
        # Create model base reconstructor
reconstructor.assemble_reconstructor_and_fitting(nChannels=1, slopesOrder="keck", 
                                                            scalingFactor=2.3e5, stretch_factor=1.0, 
                                                            rotation=1)
# mask central actuator
#reconstructor.mask_DM_actuators(174)
FR = reconstructor.FR

print(f"Reconstructor and fitting matrix shape: {FR.shape}")

# Visualize the reconstructor
fig = plt.figure(figsize=(10, 8))
im = plt.imshow(FR)
cbar = plt.colorbar(im, fraction=0.028, pad=0.02)
plt.title('Reconstructor, MODEL based, REVOLT')
plt.xlabel('Slopes')
plt.ylabel('Actuators')
plt.tight_layout()
plt.show()
# %%
from scipy.linalg import block_diag

IM = np.load("IM_revolt.npy")
reconstructor.nLGS = 4
nLGS = reconstructor.nLGS
matrices = [IM] * nLGS
IM = block_diag(*matrices)
R = reconstructor.build_reconstructor(IM)
R = R[:,:reconstructor.lgsWfsParams.nValidSubap*2]*1/4*10
print(f"Reconstructor matrix shape: {R.shape}")

# Visualize the reconstructor
fig = plt.figure(figsize=(10, 8))
im = plt.imshow(R)
cbar = plt.colorbar(im, fraction=0.028, pad=0.02)
plt.title('Reconstructor, IM based, REVOLT')
plt.xlabel('Slopes')
plt.ylabel('Actuators')
plt.tight_layout()
plt.show()

# %%
