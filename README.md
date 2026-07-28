# pyTomoAO

[![Tests](https://github.com/KeckObservatory/pyTomoAO/actions/workflows/test.yml/badge.svg)](https://github.com/KeckObservatory/pyTomoAO/actions/workflows/test.yml)
[![Docs](https://github.com/KeckObservatory/pyTomoAO/actions/workflows/docs.yml/badge.svg)](https://github.com/KeckObservatory/pyTomoAO/actions/workflows/docs.yml)
[![Code health](https://github.com/KeckObservatory/pyTomoAO/actions/workflows/code-health.yml/badge.svg)](https://github.com/KeckObservatory/pyTomoAO/actions/workflows/code-health.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI version](https://img.shields.io/pypi/v/pyTomoAO.svg)](https://pypi.org/project/pyTomoAO/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyTomoAO.svg)](https://pypi.org/project/pyTomoAO/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

📖 **[Documentation](https://keckobservatory.github.io/pyTomoAO/)** — installation, configuration reference, tutorials and API reference.

**pyTomoAO** is an open-source Python library for tomographic reconstruction in tomography-based Adaptive Optics (AO) systems. It provides tools to reconstruct atmospheric turbulence phase maps and project them onto deformable mirrors for different AO architectures, including:

- **LTAO (Laser Tomography Adaptive Optics)**
- **MOAO (Multi-Object Adaptive Optics)**

## Features

- Support for **LTAO, and MOAO** tomographic reconstructions.
- Efficient numerical solvers for tomographic phase reconstruction.
- Tools for **fitting reconstructed phase maps** onto deformable mirrors.
- Extensible and modular design to allow easy adaptation to different AO systems.
- Optimized for performance with **NumPy, SciPy, and Numba**.

## Installation

```sh
pip install pyTomoAO
```

or clone the repository:

```sh
git clone https://github.com/KeckObservatory/pyTomoAO.git
cd pyTomoAO
pip install -e .
```

See the [installation guide](https://keckobservatory.github.io/pyTomoAO/getting-started/installation.html) for GPU support and optional extras.

## Usage

Everything is driven by a single YAML configuration file:

```python
from pyTomoAO.tomographicReconstructor import tomographicReconstructor

# Build a tomographic reconstructor from a configuration file
rec = tomographicReconstructor("examples/benchmark/tomography_config_kapa.yaml")
rec.build_reconstructor()

# Fold in the DM fitting step to go from slopes straight to commands
FR = rec.assemble_reconstructor_and_fitting(nChannels=4, slopesOrder="simu")
commands = FR @ slopes
```

Full walkthrough: [quickstart](https://keckobservatory.github.io/pyTomoAO/getting-started/quickstart.html) and the [KAPA LTAO tutorial](https://keckobservatory.github.io/pyTomoAO/tutorials/ltao-kapa.html).

## Roadmap

- [ ] Implement fundamental reconstruction algorithms.
- [ ] Add GPU acceleration for real-time processing.
- [ ] Improve deformable mirror fitting routines.
- [x] Develop detailed documentation and examples.
- [ ] Implement MCAO reconstructor

## Development

### Setup

```sh
git clone https://github.com/KeckObservatory/pyTomoAO.git
cd pyTomoAO
pip install -e ".[docs,dev]"
```

### Code Style

Linting and formatting are handled by [ruff](https://docs.astral.sh/ruff/), configured in
`pyproject.toml` and enforced by the `Code health` workflow:

```sh
ruff check .            # lint
ruff format .           # format
ruff format --check .   # what CI checks
```

Naming rules are deliberately disabled: class names such as `tomographicReconstructor` are
public API, and matrix names such as `Gamma` and `Cxx` mirror the equations they implement.
See the [code style guide](https://keckobservatory.github.io/pyTomoAO/development/code-style.html).

### Documentation

The documentation is built with Sphinx and published to GitHub Pages at
<https://keckobservatory.github.io/pyTomoAO/> by the `Documentation` workflow on every push
to `main`. Pull requests build the docs with warnings treated as errors, so a broken link or
docstring fails CI.

To build it locally:

```sh
pip install -e ".[docs]"
make -C docs html          # output in docs/build/html
make -C docs strict        # exactly what CI runs
make -C docs livehtml      # auto-reloading preview (needs sphinx-autobuild)
```

See the [documentation guide](https://keckobservatory.github.io/pyTomoAO/development/documentation.html) for conventions and how publishing works.

## Testing

To run tests using `pytest`, ensure you have `pytest` installed. You can install it via pip:

```sh
pip install pytest
```

Once installed, you can run the tests by executing the following command in the root directory of the repository:

```sh
pytest
```

This will automatically discover and run all the test files in the repository.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history, and the `Unreleased` section for what
is coming next.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests. Branch from
`dev` and open the pull request against `dev`; see the
[contributing guide](https://keckobservatory.github.io/pyTomoAO/development/contributing.html)
for conventions. Pull requests must pass the tests, the ruff lint/format checks and the
documentation build, and should add an entry to the changelog.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions and discussions, open an issue on GitHub or contact one of:
- **Jacob Taylor** (Software): jacobataylor7@gmail.com
- **Uriel Conod** (Algorithm): urielconod@phas.ubc.ca