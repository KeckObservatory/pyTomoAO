# Changelog

All notable changes to pyTomoAO are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> Entries for releases before this file existed were reconstructed from the git history and
> the tagged releases, so they summarise the significant changes rather than every commit.
> Releases are listed newest first; note that `0.1.0-test1` was tagged before `0.0.1`, so
> the version numbers are not monotonic with the dates.

## [Unreleased]

### Added

- Documentation site built with Sphinx, MyST and the Furo theme, published to GitHub Pages
  at <https://keckobservatory.github.io/pyTomoAO/>. The site covers installation, a
  quickstart, a configuration reference with units and validation rules for every key, a
  user guide (concepts, reconstruction modes, DM fitting, GPU acceleration), a KAPA LTAO
  tutorial, a generated API reference and a development section.
- `Documentation` workflow: builds the docs on pull requests and pushes to `main`/`dev` with
  warnings treated as errors, and deploys to GitHub Pages on pushes to `main`.
- `Code health` workflow: `ruff check` and `ruff format --check` on pull requests and
  pushes, with the rule set configured in `pyproject.toml`.
- `python_requires=">=3.8"`, matching the classifiers, so pip refuses to install on
  interpreters the package does not support.
- Package metadata now carries a long description (the README), so the PyPI project page is
  no longer blank.
- Dependabot configuration for GitHub Actions and the pinned docs toolchain, targeting
  `dev`.
- This changelog.
- MIT license (#78).
- Developer guide and usage examples (#76).

### Changed

- The whole codebase is now formatted with `ruff format` (100-column lines). This is a
  formatting-only change; no behaviour was altered.
- Normalised the NumPy-style docstring sections in `tomographyUtilsCPU` and
  `tomographyUtilsGPU`, and added docstrings to the CPU reconstructor builders, so the API
  reference renders them correctly.
- `[docs]` extra now installs the Sphinx/Furo/MyST toolchain, pinned in
  `docs/requirements.txt`.
- `Run Pytest` now runs on pushes to `main`/`dev` as well as pull requests, tests a matrix
  of Python 3.8–3.12, and installs the package itself instead of `requirements.txt` so that
  dependency metadata is exercised. The coverage gate runs once, on 3.12. The workflow's
  `actions/checkout@v2` and `actions/setup-python@v2` pins, which use a retired Node
  runtime, were updated to v4/v5.
- `Publish Python Package to PyPI` now verifies distributions with `twine check --strict`
  and installs the wheel into a clean virtualenv before publishing, and publishes through a
  `pypi` GitHub environment that can carry a review rule.
- Contact email for Jacob Taylor updated to jacobataylor7@gmail.com.
- Project URLs point at <https://github.com/KeckObservatory/pyTomoAO>.

### Fixed

- **Importing pyTomoAO no longer reconfigures logging for the whole application.**
  `tomographicReconstructor` called `logging.basicConfig(level=logging.DEBUG)` and
  `fitting` called it with `CRITICAL`, so importing the package switched on debug logging
  for the host application (or not, depending on import order) and muted matplotlib's
  logger. The package now attaches a `NullHandler` and leaves configuration to the caller.
  To see pyTomoAO's messages, call `logging.basicConfig(level=logging.INFO)` yourself.
- **`pytest` is no longer a runtime dependency.** It was listed in `install_requires`, so
  every installation of pyTomoAO pulled in pytest; it now lives in the `dev` extra.
- `tomographyUtilsGPU` imported `gamma` from both `cupyx.scipy.special` and
  `scipy.special`, so the first import was dead. Removed it; the module only ever calls
  `gamma` on Python floats, so behaviour is unchanged.
- Unused imports and variables, bare `except` clauses, identity/equality comparison slips
  and other issues surfaced by the new lint gate.

### Removed

- The previous Read the Docs oriented Sphinx configuration, including the `sphinx_rtd_theme`
  dependency. Documentation is now published to GitHub Pages.
- Top-level `requirements.txt`, which duplicated `install_requires` and had already drifted
  from it. Install the package instead: `pip install -e ".[dev]"`.

## [1.0.1] - 2025-05-13

### Added

- User documentation: introduction, installation, basic usage and tutorial pages, with
  figures for the reconstructed wavefront and DM commands (#71, #72).
- `github_pytest_workflow.py`, a wrapper that runs the test suite with coverage and fails
  the build below a coverage threshold or when large source files have no tests (#70).

### Changed

- Substantial work on `tomographicReconstructor` and `fitting`, including the influence
  function model and the reconstructor/fitting assembly.
- Docstrings corrected across the package (#73).
- Test and example configurations trimmed.

### Removed

- `pyTomoAO/Fitting_template.py`, superseded by `fitting.py`.

## [1.0.0] - 2025-04-28

First release published to the production PyPI index.

### Added

- Single source of truth for the version: `setup.py` reads `__version__` from
  `pyTomoAO/__init__.py` (#57, #59).

### Changed

- The publish workflow now uploads to PyPI rather than Test PyPI (#60).
- Documentation build fixes (#56).

## [0.0.1] - 2025-04-25

### Changed

- Packaging and the publish workflow reworked for Test PyPI, with the version derived from
  the release tag (#53, #54).

### Removed

- The legacy `.ini` configuration format; YAML is the only supported configuration
  format (#55).

## [0.1.0-test1] - 2025-04-22

First tagged release, published to Test PyPI.

### Added

- `tomographicReconstructor`, the configuration-driven entry point, with the model-based
  MMSE reconstructor built from the atmosphere, asterism, WFS and DM parameter classes.
- Interaction-matrix-based reconstructor (#50).
- `fitting` with the double-Gaussian DM influence function model (#37).
- CPU and GPU (CuPy) implementations of the covariance and reconstructor kernels, selected
  automatically at import.
- Single-channel operation and reconstructor tuning for Keck K1 (#46).
- Benchmarking scripts comparing the CPU and GPU paths (#39, #41).
- Test suite covering the parameter classes, fitting and the reconstructor (#36), run by a
  GitHub Actions workflow (#23, #24).
- Initial Sphinx documentation scaffolding (#29).

[Unreleased]: https://github.com/KeckObservatory/pyTomoAO/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/KeckObservatory/pyTomoAO/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/KeckObservatory/pyTomoAO/compare/v0.0.1...v1.0.0
[0.0.1]: https://github.com/KeckObservatory/pyTomoAO/compare/v0.1.0-test1...v0.0.1
[0.1.0-test1]: https://github.com/KeckObservatory/pyTomoAO/releases/tag/v0.1.0-test1
