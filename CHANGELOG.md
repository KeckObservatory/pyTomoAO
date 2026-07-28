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
- Contact email for Jacob Taylor updated to jacobataylor7@gmail.com.
- Project URLs point at <https://github.com/KeckObservatory/pyTomoAO>.

### Fixed

- `tomographyUtilsGPU` imported `gamma` from both `cupyx.scipy.special` and
  `scipy.special`, so the first import was dead. Removed it; the module only ever calls
  `gamma` on Python floats, so behaviour is unchanged.
- Unused imports and variables, bare `except` clauses, identity/equality comparison slips
  and other issues surfaced by the new lint gate.

### Removed

- The previous Read the Docs oriented Sphinx configuration, including the `sphinx_rtd_theme`
  dependency. Documentation is now published to GitHub Pages.

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
