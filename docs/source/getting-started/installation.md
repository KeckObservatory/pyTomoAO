# Installation

pyTomoAO requires **Python 3.8 or newer** and runs on Linux, macOS and Windows.

## From PyPI

```bash
pip install pyTomoAO
```

## From source

```bash
git clone https://github.com/KeckObservatory/pyTomoAO.git
cd pyTomoAO
pip install .
```

Use an editable install if you intend to modify the code:

```bash
pip install -e .
```

## In a fresh environment

Working in an isolated environment avoids clashes with other scientific stacks:

::::{tab-set}

:::{tab-item} venv
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install pyTomoAO
```
:::

:::{tab-item} conda
```bash
conda create -n pytomoao python=3.11
conda activate pytomoao
pip install pyTomoAO
```
:::

::::

## Dependencies

The following packages are installed automatically:

| Package      | Used for                                                    |
| ------------ | ----------------------------------------------------------- |
| `numpy`      | Array handling and linear algebra throughout                |
| `scipy`      | Sparse gradient matrices, `.mat` I/O for validation data     |
| `numba`      | JIT-compiled covariance kernels on the CPU path              |
| `matplotlib` | Reconstruction and DM command visualisations                 |
| `PyYAML`     | Reading the configuration file                               |
| `pytest`     | Test suite                                                   |

## Optional extras

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} GPU acceleration
Install [CuPy](https://cupy.dev) matching your CUDA toolkit, for example:

```bash
pip install cupy-cuda12x
```

pyTomoAO detects CuPy at import time and switches to the GPU covariance kernels
automatically. See {doc}`../user-guide/gpu`.
:::

:::{grid-item-card} Documentation toolchain
```bash
pip install ".[docs]"
```

Installs Sphinx, the Furo theme and the MyST/design extensions needed to build this site
locally. See {doc}`../development/documentation`.
:::

::::

## Verifying the installation

```bash
python -c "import pyTomoAO; print(pyTomoAO.__version__)"
```

On import you will see a log line reporting whether CUDA was found:

```text
INFO:pyTomoAO.tomographicReconstructor:
CUDA is not available. Using CPU for computations.
```

Both paths are fully supported — the message is informational, not an error.

Next: {doc}`quickstart`.
