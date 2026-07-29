# pyTomoAO

[![Tests](https://github.com/KeckObservatory/pyTomoAO/actions/workflows/test.yml/badge.svg)](https://github.com/KeckObservatory/pyTomoAO/actions/workflows/test.yml)
[![Docs](https://github.com/KeckObservatory/pyTomoAO/actions/workflows/docs.yml/badge.svg)](https://github.com/KeckObservatory/pyTomoAO/actions/workflows/docs.yml)
[![Code health](https://github.com/KeckObservatory/pyTomoAO/actions/workflows/code-health.yml/badge.svg)](https://github.com/KeckObservatory/pyTomoAO/actions/workflows/code-health.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI version](https://img.shields.io/pypi/v/pyTomoAO.svg)](https://pypi.org/project/pyTomoAO/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyTomoAO.svg)](https://pypi.org/project/pyTomoAO/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

📖 **[Documentation](https://keckobservatory.github.io/pyTomoAO/)** — installation, configuration reference, tutorials and API reference.

**pyTomoAO** is an open-source Python library for tomographic reconstruction in Adaptive Optics (AO) systems. It reconstructs atmospheric turbulence phase maps from several laser guide star Shack–Hartmann wavefront sensors and projects them onto a deformable mirror.

## Features

- **Minimum-mean-square-error tomographic reconstruction** from multiple LGS wavefront
  sensors, driven by a layered Von Kármán turbulence model. This is the single-DM,
  single-optimisation-direction case used for **LTAO**; MOAO and MCAO are on the roadmap
  below.
- Both a **model-based** reconstructor and one built from a measured **interaction matrix**.
- **Super-resolution** support: per-WFS lenslet rotation and lateral offset.
- Tools for **fitting reconstructed phase maps** onto deformable mirrors.
- **GPU acceleration** through CuPy, selected automatically when it is available, with a
  NumPy/Numba CPU backend otherwise.

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
from pyTomoAO import example_config
from pyTomoAO.reconstructor import tomographicReconstructor

# Build a tomographic reconstructor from a configuration file
rec = tomographicReconstructor(example_config("kapa"))
rec.build_reconstructor()

# Fold in the DM fitting step to go from slopes straight to commands
FR = rec.assemble_reconstructor_and_fitting(nChannels=4, slopesOrder="simu")
commands = FR @ slopes
```

Full walkthrough: [quickstart](https://keckobservatory.github.io/pyTomoAO/getting-started/quickstart.html) and the [KAPA LTAO tutorial](https://keckobservatory.github.io/pyTomoAO/tutorials/ltao-kapa.html).

## Roadmap

- [x] Fundamental tomographic reconstruction algorithms.
- [x] GPU acceleration.
- [x] Deformable mirror fitting routines.
- [x] Detailed documentation and examples.
- [ ] MOAO reconstructor (per-direction outputs).
- [ ] MCAO reconstructor (multiple DM altitudes).

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

### Testing

The `dev` extra installs pytest and coverage:

```sh
pip install -e ".[dev]"
pytest
```

CI runs the suite against Python 3.9–3.13 and applies a coverage threshold. See the
[testing guide](https://keckobservatory.github.io/pyTomoAO/development/testing.html).

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