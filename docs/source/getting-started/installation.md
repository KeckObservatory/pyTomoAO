# Installation

pyTomoAO requires **Python 3.9 or newer** and runs on Linux, macOS and Windows.
It is tested on 3.9 through 3.13.

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
```bash
pip install "pyTomoAO[gpu]"
```

which pulls in [CuPy](https://cupy.dev) for CUDA 12. On CUDA 11, install the matching wheel
yourself instead:

```bash
pip install cupy-cuda11x
```

pyTomoAO detects CuPy at import time and switches to the GPU covariance kernels
automatically. If CuPy is installed but fails to load — a driver or toolkit mismatch, or no
visible device — pyTomoAO logs a **warning** with the underlying error and falls back to the
CPU backend, rather than silently reporting that CUDA is unavailable. See
{doc}`../user-guide/gpu`.
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

## Logging

pyTomoAO logs through the standard {mod}`logging` module and does **not** configure
logging for you — importing it is silent. Turn its messages on from your own code:

```python
import logging

logging.basicConfig(level=logging.INFO)

from pyTomoAO.reconstructor import tomographicReconstructor
```

You will then see progress messages, including which backend was selected:

```text
INFO:pyTomoAO.reconstructor:
CUDA is not available. Using CPU for computations.
```

Both backends are fully supported — that message is informational, not an error.

To keep pyTomoAO quiet while your own application logs at `INFO`:

```python
logging.getLogger("pyTomoAO").setLevel(logging.WARNING)
```

Next: {doc}`quickstart`.
