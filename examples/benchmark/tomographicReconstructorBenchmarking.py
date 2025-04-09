# %%
import matplotlib.pyplot as plt
import tomoAO
from tomoAO.Reconstruction.reconClassType import tomoReconstructor
import sys
sys.path.insert(0, "/Users/urielconod/pyTomoAO")
from pyTomoAO.tomographicReconstructor import tomographicReconstructor
import time
import numpy as np
import os

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
        config_path = os.path.join(os.path.dirname(__file__), "tomography_config_kapa.yaml")
        reconstructor = tomographicReconstructor(config_path)
    elif library == "tomoAO":
        ao_mode = "MLAO"
        config_dir = "/Users/urielconod/PyTomo/Demo/"
        config_file = "config_kapa_single_channel.ini"
        
        # Read the ini file and replace occurrences of the specified path
        with open(config_dir+config_file, 'r') as file:
            config_data = file.read()
        
        # Replace the path
        old_path = "/home/joaomonteiro/Desktop/"
        new_path = "/home/aodev/uriel/"  # Replace with your desired folder name
        config_data = config_data.replace(old_path, new_path)
        
        # Write the updated content back to the ini file
        with open("./" + config_file, 'w') as file:
            file.write(config_data)

        config_vars = tomoAO.IO.load_from_ini(config_file, ao_mode=ao_mode, config_dir="./")
        
        aoSys = tomoAO.Simulation.AOSystem(config_vars)
    
    for _ in range(num_iterations):
        start_time = time.time()
        if library == "pyTomoAO":
            rec = reconstructor.build_reconstructor()
            plt.imshow(rec)
            plt.show()
        elif library == "tomoAO":
            rec = tomoReconstructor(aoSys=aoSys, alpha=10, os=2)
            plt.imshow(rec.Reconstructor[0])
            plt.show()
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    times = np.array(times)
    stats = {
        'min': np.min(times),
        'max': np.max(times),
        'std': np.std(times)
    }
    
    return np.mean(times), stats

library1 = "pyTomoAO"
mean_time1, statistics1 = benchmark_reconstructor(library1)

#Example usage:
library2 = "tomoAO"
mean_time2, statistics2 = benchmark_reconstructor(library2)

print(f"Average build time for {library1}: {mean_time1:.3f} seconds")
print(f"Statistics: {statistics1}")
print(f"Average build time for {library2}: {mean_time2:.3f} seconds")
print(f"Statistics: {statistics2}")
# %%

