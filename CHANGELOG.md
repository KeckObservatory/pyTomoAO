# Changelog

All notable changes to pyTomoAO are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> Entries for releases before this file existed were reconstructed from the git history and
> the tagged releases, so they summarise the significant changes rather than every commit.
> Releases are listed newest first; note that `0.1.0-test1` was tagged before `0.0.1`, so
> the version numbers are not monotonic with the dates.

## [Unreleased]

### Breaking

- **The reconstructor no longer forwards arbitrary attributes to the parameter objects, and
  assigning an unknown name now raises `AttributeError`** (#117).

  `__getattr__` and `__setattr__` used to resolve any unknown attribute by searching five
  parameter objects in turn — about 200 lines, plus a hand-maintained 15-entry
  `special_attrs` list and a bespoke fan-out for `nLGS`. The cost was that a misspelled
  parameter fell through to `object.__setattr__` and became a new attribute:

  ```python
  rec.r0_zenit = 0.1          # accepted silently before 2.0
  rec.build_reconstructor()   # built with the old r0
  ```

  Nothing forwarded was visible to `dir()`, IDE completion or type checkers, and the search
  ran on every internal assignment, doing up to five `hasattr` calls each — any of which
  could trigger a property getter that computes an array.

  `nLGS`, `r0`, `r0_zenith` and `L0` remain available directly on the reconstructor as
  explicit properties; setting `nLGS` still updates every parameter object that tracks it.
  Those were the only forwarded names used anywhere in the tests, docs, examples or README.
  Everything else is reached through the object that owns it — `rec.atmParams.altitude`,
  `rec.lgsWfsParams.nValidSubap`, `rec.dmParams.validActuators`.

- **Two modules were renamed so that no module shadows a class of the same name** (#115):

  | before | after |
  | --- | --- |
  | `pyTomoAO.tomographicReconstructor` | `pyTomoAO.reconstructor` |
  | `pyTomoAO.fitting` | `pyTomoAO.dm_fitting` |

  `__init__.py` re-exports `tomographicReconstructor` and `fitting`, so those names bound to
  the **classes** and the modules underneath became unreachable by attribute lookup:
  `import pyTomoAO.fitting as m; m.fitting` raised `AttributeError`, and `unittest.mock`
  string targets such as `"pyTomoAO.fitting.fitting"` silently resolved to the class —
  working on Python 3.11+, which resolves modules first, and failing on 3.9 and 3.10, which
  walk attributes. The tests carried `importlib.import_module` workarounds and a written
  explanation for exactly this; both are now gone, along with the corresponding caveats in
  the contributing and testing guides.

  Class names are unchanged: `tomographicReconstructor`, `fitting` and the `*Parameters`
  classes keep the naming the code-style guide deliberately preserves. Update imports of the
  form `from pyTomoAO.fitting import fitting` to `from pyTomoAO.dm_fitting import fitting`;
  `from pyTomoAO import fitting` is unaffected.

- **Reconstruction grid points are now indexed consistently in C order, and
  `reconstruct_wavefront` no longer returns a transposed wavefront** (#104).

  `_sparseGradientMatrixAmplitudeWeighted` indexed the grid in Fortran order — a MATLAB
  port artefact, since MATLAB is column-major throughout — while masking it with a C-order
  boolean, and the covariance kernels matched that. `reconstruct_wavefront` then scattered
  the result in C order, so the map it returned was the transpose of the real wavefront.
  Only `visualize_reconstruction` compensated, by displaying `reconstructed_wavefront.T`;
  the plots looked right while the returned array did not.

  Two consequences, both fixed:

  - A pure x-gradient slope vector reconstructed to a ramp along **y**. Verified directly:
    before, variation along x was 1.3e-22 against 1.6e-06 along y; after, the reverse.
  - On a pupil that is **not symmetric under transpose**, the gradient operator was simply
    wrong: a flat wavefront produced slopes of magnitude 1.0 (12 of 84 non-zero). On a
    symmetric pupil the two conventions coincide and the error vanishes, which is why every
    configuration shipped with the package — all of them symmetric — hid it.

  The compensating transpose in `visualize_reconstruction` is removed with the fix, so
  plots are unchanged. Anything consuming `reconstruct_wavefront` directly, or comparing
  against a stored reference wavefront, will see the corrected orientation.

### Added

- **The reference configurations now ship inside the package**, so `pip install pyTomoAO` is
  enough to run the documented examples. Previously the published wheel contained ten `.py`
  files and no data, and the configuration path in the README raised `FileNotFoundError` for
  anyone who had not cloned the repository (#106):

  ```python
  from pyTomoAO import example_config, list_example_configs
  rec = tomographicReconstructor(example_config("kapa"))
  ```

  `list_example_configs()` returns `['kapa', 'kapa-single-channel', 'keck', 'revolt']`. The
  YAML files moved from `examples/benchmark/` to `pyTomoAO/data/`; `example_config` returns a
  path inside the installed package, so copy one before editing.
- `pip install "pyTomoAO[gpu]"` extra, which pulls in `cupy-cuda12x` (#109).
- `CITATION.cff`, so GitHub renders a "Cite this repository" button (#108).

- Documentation site built with Sphinx, MyST and the Furo theme, published to GitHub Pages
  at <https://keckobservatory.github.io/pyTomoAO/>. The site covers installation, a
  quickstart, a configuration reference with units and validation rules for every key, a
  user guide (concepts, reconstruction modes, DM fitting, GPU acceleration), a KAPA LTAO
  tutorial, a generated API reference and a development section.
- `Documentation` workflow: builds the docs on pull requests and pushes to `main`/`dev` with
  warnings treated as errors, and deploys to GitHub Pages on pushes to `main`.
- `Code health` workflow: `ruff check` and `ruff format --check` on pull requests and
  pushes, with the rule set configured in `pyproject.toml`.
- `python_requires=">=3.9"`, so pip refuses to install on interpreters the package does
  not support.
- Package metadata now carries a long description (the README), so the PyPI project page is
  no longer blank.
- Dependabot configuration for GitHub Actions and the pinned docs toolchain, targeting
  `dev`.
- This changelog.
- MIT license (#78).
- Developer guide and usage examples (#76).

### Changed

- **`force_cpu=True` now actually selects the CPU kernels.** It used to flip a module-level
  `CUDA` flag, but the GPU functions were already bound to module names at import time, so
  the reconstructor logged "Forcing CPU usage" and then ran the GPU kernels in float64.
  `Cxx`, `Cox`, `CnZ` and `RecStatSA` came back as `cupy.ndarray`, and the option could not
  serve its main purpose of side-stepping a misbehaving GPU. The backend is now resolved per
  instance by the new `pyTomoAO.backend` module and exposed as `rec.backend` (`"cpu"` or
  `"gpu"`); constructing one reconstructor with `force_cpu=True` no longer changes the
  backend of any other. The module-level `pyTomoAO.reconstructor.CUDA` remains as
  a read-only "is CuPy importable" flag (#112).
- The reconstructor now **solves** the regularised system instead of forming an explicit
  inverse and multiplying. `Γ·Cxx·Γᵀ + Cₙ` is symmetric positive definite, so a Cholesky
  solve is both cheaper and better conditioned; `build_reconstructor` drops from 2.9 s to
  2.4 s on CPU for the KAPA configuration. Results move only at round-off (8.7e-15 relative
  in float64).
- **A CuPy that is installed but fails to load is now reported as a warning**, with the
  underlying exception, instead of an `INFO` message claiming CUDA is unavailable. A driver
  or toolkit mismatch was indistinguishable from CuPy simply not being installed, so users
  silently took the CPU path and a ~35× slowdown. CuPy genuinely not being installed stays
  at `INFO` and now points at the `[gpu]` extra (#110).
- The README no longer advertises MOAO support. There is one reconstructor and no
  MOAO-specific code path; the feature list now describes what the library actually does,
  and the roadmap ticks the items that are already shipped rather than listing GPU support
  and DM fitting as outstanding (#107).
- **The covariance kernels are ~5× faster and produce bit-identical results.** They used to
  evaluate the covariance over the full `sampling × sampling` grid and only then cut it down
  to the valid pupil points, discarding 71% of the Bessel evaluations on the function that
  is 89% of runtime. The pupil mask is now applied to the coordinates, once per guide-star
  pair rather than once per turbulence layer. `build_reconstructor` on the KAPA
  configuration drops from **14.4 s to 2.9 s** on CPU and 0.11 s to 0.08 s on GPU, and the
  CPU test suite from 63 s to 16 s (#101, #103).
- A real-valued `float64 → float64` Bessel kernel (`_kv56_real`) is now used on the hot
  path. The distances are real and the result was immediately passed through `np.real`, so
  the complex overload only added a `complex128` copy of the input and twice the arithmetic;
  the copy alone was ~7% of `build_reconstructor`. The complex `_kv56` remains for
  compatibility, and both now share one set of module-level expansion constants so they
  cannot drift apart (#102).
- The whole codebase is now formatted with `ruff format` (100-column lines). This is a
  formatting-only change; no behaviour was altered.
- Normalised the NumPy-style docstring sections in `tomographyUtilsCPU` and
  `tomographyUtilsGPU`, and added docstrings to the CPU reconstructor builders, so the API
  reference renders them correctly.
- `[docs]` extra now installs the Sphinx/Furo/MyST toolchain, pinned in
  `docs/requirements.txt`.
- `Run Pytest` now runs on pushes to `main`/`dev` as well as pull requests, tests a matrix
  of Python 3.9–3.13, and installs the package itself instead of `requirements.txt` so that
  dependency metadata is exercised. The coverage gate runs once, on 3.12. The workflow's
  `actions/checkout@v2` and `actions/setup-python@v2` pins, which use a retired Node
  runtime, were updated to v4/v5. The plain `pytest` step is skipped on 3.12, where the
  coverage gate already runs the same suite and fails the job on any test failure —
  previously the slowest job in the matrix ran every test twice (#95).
- The coverage wrapper now runs pytest as `sys.executable -m pytest` instead of whichever
  `pytest` is first on `PATH`, so the coverage run always matches the environment under
  test.
- `Publish Python Package to PyPI` now verifies distributions with `twine check --strict`
  and installs the wheel into a clean virtualenv before publishing, and publishes through a
  `pypi` GitHub environment that can carry a review rule.
- The reconstruction integration test no longer pins 7-significant-figure mean-OPD values
  with `rtol=0`. Those failed on any machine with a CUDA device, because the float32 GPU
  path lands 1.3e-3 away from the float64 reference — so the suite was red on developer
  machines and green in CI only because the runners have no GPU. Reconstruction accuracy is
  now checked by a physical round trip (known phase → gradient operator → reconstruction),
  which holds on both backends and survives numerical improvements (#98).
- Added `tests/conftest.py` with path fixtures resolved from `__file__`, so the tests no
  longer depend on being run from the repository root, and moved the temporary config
  written by `simple_config` into `tmp_path` (#99).
- Contact email for Jacob Taylor updated to jacobataylor7@gmail.com.
- Project URLs point at <https://github.com/KeckObservatory/pyTomoAO>.

### Fixed

- **The `K_{5/6}` Bessel kernel lost seven digits of accuracy above `z = 2`.** Three
  compounding defects: the `1/z^5` coefficient of the asymptotic expansion read
  `5005/177147` where the recurrence `a_k = a_{k-1}(4v²-(2k-1)²)/(8k)` gives
  `40040/177147` (exactly 8× too small); the series/asymptotic crossover sat at `z = 2`,
  where the asymptotic expansion is nowhere near converged; and `Γ(11/6)` was stored to
  only 12 digits, which the series' `exp(z)` cancellation amplified into the dominant error
  term above `z ≈ 4`. Worst-case relative error against `scipy.special.kv` drops from
  **2.1e-3 to 3.6e-8** in double precision (2.1e-4 in single, where cancellation is the
  limit). With the KAPA configuration 17% of point pairs fell in the affected range, so
  reconstructor values shift slightly — about 0.1–0.4% on mean reconstructed OPD (#97).
- **`wfsLensletsRotation` was applied in the wrong units.** `_create_guide_star_grid`
  converted the angle from radians to degrees and then passed it to `_rotateWFS`, which
  treats its argument as radians, so a requested rotation of θ was applied as `57.3·θ`. Both
  backends were affected. This was invisible in every shipped example configuration, all of
  which set the rotation to zero (#92).
- **`assemble_reconstructor_and_fitting` was not idempotent.** The `simu` and `keck` branches
  wrote their X/Y block swap back into `_reconstructor`, so calling the method a second
  time — natural when tuning `scalingFactor`, `rotation` or `stretch_factor` — swapped the
  blocks again and silently returned a different, wrong `FR`. The reordering is now derived
  into a local and `reconstructor`/`R` keeps the matrix `build_reconstructor` produced (#93).
- **Valid grid points that reconstructed to exactly zero were turned into NaN.**
  `reconstruct_wavefront` and `visualize_commands` used zero as the "outside the pupil"
  sentinel; they now build their output from the boolean mask (#94).
- **Zero-separation covariance entries were wrong by a factor of ~1.887 on GPU.** The CUDA
  tiny-argument shortcut for `K_{5/6}` used a coefficient of `1.89719` where the
  small-argument limit gives `2^(5/6)·Γ(5/6)/2 = 1.005635`, and both backends selected the
  zero-separation case with an exact `rho != 0` test. Because the two coordinate grids are
  built by different arithmetic, a mathematically-zero separation could evaluate to a few
  ULPs instead and take the Bessel branch — corrupting the largest entries of `Cxx`/`Cox`.
  The constant is corrected and the selection is now tolerance-based (#90).
- **The GPU interaction-matrix reconstructor could not run at all.** `_build_reconstructor_im`
  called `cp.sqeeze`, and the `IM` argument was never copied to the device (#89).
- **Importing pyTomoAO no longer reconfigures logging for the whole application.**
  `tomographicReconstructor` called `logging.basicConfig(level=logging.DEBUG)` and
  `fitting` called it with `CRITICAL`, so importing the package switched on debug logging
  for the host application (or not, depending on import order) and muted matplotlib's
  logger. The package now attaches a `NullHandler` and leaves configuration to the caller.
  To see pyTomoAO's messages, call `logging.basicConfig(level=logging.INFO)` yourself.
- **`pytest` is no longer a runtime dependency.** It was listed in `install_requires`, so
  every installation of pyTomoAO pulled in pytest; it now lives in the `dev` extra.
- **Tests failed on Python 3.9 and 3.10.** `test_tomographicReconstructor` and
  `test_fitting` patched dotted string targets such as
  `"pyTomoAO.tomographicReconstructor.atmosphereParameters"`. Because `__init__.py`
  re-exports those classes under their modules' names, the dotted path resolves to the
  class; `unittest.mock` resolves modules first on 3.11+ but walks attributes on older
  versions, so the patches raised `AttributeError` and `ModuleNotFoundError` there. The
  tests now patch the module and class objects directly.
- `tomographyUtilsGPU` imported `gamma` from both `cupyx.scipy.special` and
  `scipy.special`, so the first import was dead. Removed it; the module only ever calls
  `gamma` on Python floats, so behaviour is unchanged.
- Unused imports and variables, bare `except` clauses, identity/equality comparison slips
  and other issues surfaced by the new lint gate.

### Removed

- **Support for Python 3.8**, which reached end of life in October 2024. The supported and
  tested range is now 3.9 through 3.13.
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
