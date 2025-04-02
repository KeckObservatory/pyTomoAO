# %%
import matplotlib.pyplot as plt
import tomoAO
from tomoAO.Reconstruction.reconClassType import tomoReconstructor
from pyTomoAO import *
import time
import numpy as np

def benchmark_reconstructor(library, num_iterations=5):
    """
    Benchmark the time required to build reconstructors for different libraries.
    
    Parameters:
    -----------
    library : str
        The library to benchmark ('pyTomoAO' or 'tomoAO')
    num_iterations : int, optional
        Number of iterations to run the benchmark (default: 5)
        
    Returns:
    --------
    float
        Average time in seconds to build the reconstructor
    dict
        Additional statistics (min, max, std of times)
    """
    times = []
    
    # Setup configurations once outside the timing loop
    if library == "pyTomoAO":
        reconstructor = tomographicReconstructor("../pyTomoAO/examples/tomography_config_kapa.yaml")
    elif library == "tomoAO":
        ao_mode = "MLAO"
        config_dir = "./Demo/"
        config_file = "config.ini"
        config_vars = tomoAO.IO.load_from_ini(config_file, ao_mode=ao_mode, config_dir=config_dir)
        aoSys = tomoAO.Simulation.AOSystem(config_vars)
    
    for _ in range(num_iterations):
        start_time = time.time()
        try:
            if library == "pyTomoAO":
                rec = reconstructor.build_reconstructor()
            elif library == "tomoAO":
                rec = tomoReconstructor(aoSys=aoSys, alpha=10, os=2)
        except Exception as e:
            print(f"Error during reconstruction: {str(e)}")
            return None, None
            
        end_time = time.time()
        times.append(end_time - start_time)
    
    times = np.array(times)
    stats = {
        'min': np.min(times),
        'max': np.max(times),
        'std': np.std(times)
    }
    
    return np.mean(times), stats

# Example usage:
library1 = "pyTomoAO"
mean_time1, statistics1 = benchmark_reconstructor(library1)

library2 = "tomoAO"
mean_time2, statistics2 = benchmark_reconstructor(library2)

print(f"Average build time for {library1}: {mean_time1:.3f} seconds")
print(f"Statistics: {statistics1}")
print(f"Average build time for {library2}: {mean_time2:.3f} seconds")
print(f"Statistics: {statistics2}")
# %%
