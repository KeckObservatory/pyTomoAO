#%%
import numpy as np
import matplotlib.pyplot as plt
from test_auto import build_reconstructor as build_reconstructor_cpu
from test_auto import auto_correlation as auto_correlation_cpu
from test_auto import cross_correlation as cross_correlation_cpu
import cupy as cp
from test_auto_gpu import build_reconstructor_gpu 
from test_auto_gpu import auto_correlation_gpu 
from test_auto_gpu import cross_correlation_gpu 
import time
from test_auto import sparseGradientMatrixAmplitudeWeighted

# Define the same parameter classes
class TomoParams:
    def __init__(self, nFitSrc, directionVectorSrc, fitSrcHeight):
        self.sampling = None
        self.nFitSrc = nFitSrc
        self.directionVectorSrc = directionVectorSrc
        self.fitSrcHeight = fitSrcHeight
        self.fitSrcWeight = np.ones(self.nFitSrc**2)/self.nFitSrc**2


class LgsWfsParams:
    def __init__(self, DSupport, wfsLensletsRotation, wfsLensletsOffset, validLLMap):
        self.DSupport = DSupport
        self.wfsLensletsRotation = wfsLensletsRotation
        self.wfsLensletsOffset = wfsLensletsOffset
        self.validLLMapSupport = np.pad(validLLMap, pad_width=2, mode='constant', constant_values=0)

class AtmParams:
    def __init__(self, nLayer, altitude, r0, L0, fractionnalR0):
        self.nLayer = nLayer
        self.altitude = altitude
        self.r0 = r0
        self.L0 = L0
        self.fractionnalR0 = fractionnalR0

class LgsAsterismParams:
    def __init__(self, nLGS, directionVectorLGS, LGSheight, LGSwavelength):
        self.nLGS = nLGS
        self.directionVectorLGS = directionVectorLGS
        self.LGSheight = LGSheight
        self.LGSwavelength = LGSwavelength

# %%
DSupport = 8.0
wfsLensletsRotation = np.zeros(4)
wfsLensletsOffset = np.zeros((2, 4))
nLayer = 2
altitude = np.array([    0.  ,         577.35026919,  1154.70053838,  2309.40107676,
4618.80215352,  9237.60430703, 18475.20861407])
r0 = 0.171
L0 = 30.0
nFitSrc = 1
directionVectorSrc = np.array([[0.0],
                                [0.0]])
fitSrcHeight = np.inf
fractionnalR0 = np.array([0.46, 0.13, 0.04, 0.05, 0.12, 0.09, 0.11])
nLGS = 4
directionVectorLGS = np.array([[ 3.68458398e-05,  2.25615699e-21, -3.68458398e-05, -6.76847096e-21],
                                [ 0.00000000e+00, 3.68458398e-05,  4.51231397e-21, -3.68458398e-05],
                                [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00]])
LGSheight = 103923.04845413263
validLLMap = np.array([
    [0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
    [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0]
])
LGSwavelength = 5.89e-7

# Create parameter objects
tomoParams = TomoParams(nFitSrc, directionVectorSrc, fitSrcHeight)
lgsWfsParams = LgsWfsParams(DSupport, wfsLensletsRotation, wfsLensletsOffset, validLLMap)
atmParams = AtmParams(nLayer, altitude, r0, L0, fractionnalR0)
lgsAsterismParams = LgsAsterismParams(nLGS, directionVectorLGS, LGSheight, LGSwavelength)
_, gridMask = sparseGradientMatrixAmplitudeWeighted(
    lgsWfsParams.validLLMapSupport,
    amplMask=None, 
    overSampling=2
)
tomoParams.sampling =  gridMask.shape[0]
#%%
# # Test the performance of the auto correlation
# print("\n=== Testing Auto Correlation Performance ===")
# cp.cuda.Stream.null.synchronize()  # Ensure GPU is clear
# start_time = time.time()
# R_gpu_auto = auto_correlation_gpu(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, gridMask)
# end_time = time.time()
# R_gpu_auto = cp.asnumpy(R_gpu_auto)

# # Create the reconstructor using CPU
# start_time_cpu = time.time()
# R_cpu_auto = auto_correlation_cpu(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, gridMask)
# end_time_cpu = time.time()

# print(f"GPU Auto Correlation Execution time: {end_time - start_time:.2f} seconds")
# print(f"CPU Auto Correlation Execution time: {end_time_cpu - start_time_cpu:.2f} seconds")
# print(f"Speedup: {(end_time_cpu - start_time_cpu) / (end_time - start_time):.2f}x")

# # Verify results
# if np.allclose(R_gpu_auto, R_cpu_auto, rtol=1e-5, atol=1e-8):
#     print("GPU and CPU auto correlation results match!")
# else:
#     print("Warning: GPU and CPU auto correlation results differ")
#     print(f"Max absolute difference: {np.max(np.abs(R_gpu_auto - R_cpu_auto))}")

# plt.imshow(R_gpu_auto, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.title("GPU Reconstructor")
# plt.show()
# plt.imshow(R_cpu_auto, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.title("CPU Reconstructor")
# plt.show()

# Test the performance of the cross correlation
# print("\n=== Testing Cross Correlation Performance ===")
# cp.cuda.Stream.null.synchronize()  # Ensure GPU is clear
# start_time = time.time()
# R_gpu_cross = cross_correlation_gpu(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, gridMask)
# end_time = time.time()
# R_gpu_cross = cp.asnumpy(R_gpu_cross).squeeze()
# # Create the reconstructor using CPU
# start_time_cpu = time.time()
# R_cpu_cross = cross_correlation_cpu(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, gridMask)
# end_time_cpu = time.time()
# R_cpu_cross = R_cpu_cross.squeeze()

# print(f"GPU Cross Correlation Execution time: {end_time - start_time:.2f} seconds")
# print(f"CPU Cross Correlation Execution time: {end_time_cpu - start_time_cpu:.2f} seconds")
# print(f"Speedup: {(end_time_cpu - start_time_cpu) / (end_time - start_time):.2f}x")

# # Verify results
# if np.allclose(R_gpu_cross, R_cpu_cross, rtol=1e-5, atol=1e-8):
#     print("GPU and CPU auto correlation results match!")
# else:
#     print("Warning: GPU and CPU auto correlation results differ")
#     print(f"Max absolute difference: {np.max(np.abs(R_gpu_cross - R_cpu_cross))}")

# plt.imshow(R_gpu_cross, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.title("GPU Reconstructor")
# plt.show()
# plt.imshow(R_cpu_cross, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.title("CPU Reconstructor")
# plt.show()

# Create the reconstructor using GPU (float64)
print("\n=== Testing Reconstructor Performance (float64) ===")
cp.cuda.Stream.null.synchronize()  # Ensure GPU is clear
start_time = time.time()
R_gpu_f64 = build_reconstructor_gpu(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, use_float32=False)
end_time = time.time()
print(f"GPU Reconstructor (float64) Execution time: {end_time - start_time:.2f} seconds")

# Create the reconstructor using CPU
start_time_cpu = time.time()
R_cpu = build_reconstructor_cpu(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams)
end_time_cpu = time.time()
print(f"CPU Reconstructor Execution time: {end_time_cpu - start_time_cpu:.2f} seconds")
print(f"Speedup (float64): {(end_time_cpu - start_time_cpu) / (end_time - start_time):.2f}x")

# Verify float64 results
print("\n=== Verification Analysis (float64) ===")
abs_diff_f64 = np.abs(R_gpu_f64 - R_cpu)
max_diff_f64 = np.max(abs_diff_f64)
mean_diff_f64 = np.mean(abs_diff_f64)
rel_diff_f64 = np.max(np.abs((R_gpu_f64 - R_cpu) / (R_cpu + np.finfo(float).eps)))

print(f"Max absolute difference (float64): {max_diff_f64:.2e}")
print(f"Mean absolute difference (float64): {mean_diff_f64:.2e}")
print(f"Max relative difference (float64): {rel_diff_f64:.2e}")

if np.allclose(R_gpu_f64, R_cpu, rtol=1e-5, atol=1e-8):
    print("GPU (float64) and CPU reconstructors match!")
else:
    print("Warning: GPU (float64) and CPU reconstructors differ")

# Try GPU with float32 for even more speed
print("\n=== Testing Reconstructor Performance (float32) ===")
cp.cuda.Stream.null.synchronize()  # Ensure GPU is clear
start_time = time.time()
R_gpu_f32 = build_reconstructor_gpu(tomoParams, lgsWfsParams, atmParams, lgsAsterismParams, use_float32=True)
end_time = time.time()
print(f"GPU Reconstructor (float32) Execution time: {end_time - start_time:.2f} seconds")

# Verify float32 results
print("\n=== Verification Analysis (float32) ===")
abs_diff_f32 = np.abs(R_gpu_f32 - R_cpu)
max_diff_f32 = np.max(abs_diff_f32)
mean_diff_f32 = np.mean(abs_diff_f32)
rel_diff_f32 = np.max(np.abs((R_gpu_f32 - R_cpu) / (R_cpu + np.finfo(float).eps)))

print(f"Max absolute difference (float32): {max_diff_f32:.2e}")
print(f"Mean absolute difference (float32): {mean_diff_f32:.2e}")
print(f"Max relative difference (float32): {rel_diff_f32:.2e}")

if np.allclose(R_gpu_f32, R_cpu, rtol=1e-4, atol=1e-6):
    print("GPU (float32) and CPU reconstructors match within relaxed tolerances!")
else:
    print("Warning: GPU (float32) and CPU reconstructors differ significantly")

# Visualize results
plt.figure(figsize=(15, 10))

# CPU Reconstructor
plt.subplot(2, 3, 1)
plt.imshow(R_cpu, cmap='viridis')
plt.colorbar()
plt.title("CPU Reconstructor")

# GPU Reconstructor (float64)
plt.subplot(2, 3, 2)
plt.imshow(R_gpu_f64, cmap='viridis')
plt.colorbar()
plt.title("GPU Reconstructor (float64)")

# Difference (float64)
plt.subplot(2, 3, 3)
plt.imshow(abs_diff_f64, cmap='hot')
plt.colorbar()
plt.title("Difference (float64)")

# GPU Reconstructor (float32)
plt.subplot(2, 3, 4)
plt.imshow(R_gpu_f32, cmap='viridis')
plt.colorbar()
plt.title("GPU Reconstructor (float32)")

# Difference (float32)
plt.subplot(2, 3, 5)
plt.imshow(abs_diff_f32, cmap='hot')
plt.colorbar()
plt.title("Difference (float32)")

plt.tight_layout()
plt.show()

# %%
