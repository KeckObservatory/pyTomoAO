Installation
************

Firstly, you'll need Python. I recommend using Python 3.6 or later, as Python 2.7 is no longer supported. You can download the latest version of Python from the official website: https://www.python.org/downloads/.
Alternatively, you can use a package manager like Anaconda, which comes with many scientific libraries pre-installed. Anaconda is available for Windows, macOS, and Linux. You can download it from https://www.anaconda.com/products/distribution.
If you are using Anaconda, you can create a new environment with Python 3.6 or later by running the following command in your terminal or Anaconda Prompt:
```bash
conda create -n myenv python=3.6
```
Replace `myenv` with the name you want to give to your environment. After creating the environment, activate it with:
```bash
conda activate myenv
```
You can then install the required libraries using conda or pip as described below.
You can also use a package manager like `pip` to install the required libraries. If you have Python 3.6 or later installed, you can use pip to install the required libraries by running the following command in your terminal or command prompt:
```bash
pip install numpy scipy astropy pyfftw pyyaml numba aotools
```
If you are using Anaconda, you can also use conda to install the required libraries. For example, to install NumPy and SciPy, you can run:
```bash
conda install numpy scipy
```
If you are using a different operating system, the installation process may vary. Please refer to the documentation for your specific operating system for more information on how to install Python and the required libraries.

Installation
============
Once all the requirements outlined below are met, you are ready to install either using pip or by downloading the code from github.::

    git clone https://github.com/jacotay7/pyTomoAO.git
    cd pyTomoAO
    pip install .

or::

    pip install pyTomoAO


Required Libraries
==================

pyTomoAO requires a number of libraries to run. The following is a list of the required libraries, and how to install them. If you are using Anaconda, you can use conda to install most of these libraries. If you are using pip, you can install them using pip as shown below.
The following libraries are required for pyTomoAO to run::

    numpy
    scipy
    matplotlib
    pytest
    pyyaml
    numba
    logger

For GPU acceleration using CUDA::

    cupy
