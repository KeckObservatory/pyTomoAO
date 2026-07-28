# Testing

## Running the suite

```bash
pip install -e ".[dev]"
pytest
```

`pytest.ini` points at `tests/` and enables live INFO logging, so you see the same
reconstructor log lines the library emits at runtime.

Useful invocations:

```bash
pytest tests/test_tomographicReconstructor.py    # one module
pytest -k "atmosphere"                           # by name
pytest -x -vv                                    # stop at the first failure, verbose
pytest --cov=pyTomoAO --cov-report=term-missing  # coverage with uncovered lines
```

Version-specific behaviour is easiest to reproduce in a throwaway environment:

```bash
conda create -y -n pytomoao310 python=3.10
conda run -n pytomoao310 pip install ".[dev]"
conda run -n pytomoao310 pytest
```

## What is covered

| Test module                         | Covers                                            |
| ----------------------------------- | ------------------------------------------------- |
| `test_atmosphereParametersClass.py` | Layer validation, airmass and `r0` scaling         |
| `test_lgsAsterismParametersClass.py`| Asterism geometry and direction vectors            |
| `test_lgsWfsParametersClass.py`     | Lenslet maps, `nValidSubap`, support padding       |
| `test_dmParametersClass.py`         | Actuator maps and geometry validation              |
| `test_fitting.py`                   | Influence functions and the fitting matrix         |
| `test_tomographicReconstructor.py`  | End-to-end reconstructor construction              |

The parameter classes are the cheapest place to add tests, and the place where a bad
configuration does the most damage — new validation logic should always come with a test
for both the accepted and rejected cases.

## The CI gate

`.github/workflows/test.yml` runs on pull requests and on pushes to `main` and `dev`. It
installs the package itself — `pip install ".[dev]"` — and runs `pytest` against a matrix of
Python 3.9 through 3.13, so both the code and the dependency metadata are checked on every
supported version.

On the 3.12 leg it additionally runs a wrapper script rather than pytest directly:

```bash
python .github/github_pytest_workflow.py --fail-on-problems --verbose --coverage-threshold 50
```

The wrapper runs pytest with coverage and then applies two additional checks:

- **Coverage threshold** — the run fails if total coverage falls below the given percentage
  (50% in CI; the script's own default is 75%).
- **Untested-file heuristic** — source files above `--line-count-threshold` lines (default
  150) with no corresponding test file are reported as problems.

With `--fail-on-problems` set, either condition fails the job. Run the same command locally
before opening a pull request if you want to see exactly what CI will say.

## Writing tests

- Use the configurations under `examples/benchmark/` rather than inventing new YAML where
  possible — they exercise realistic geometries.
- Reconstructor construction for the full KAPA configuration is not fast; prefer the
  single-channel or REVOLT configurations for tests that only need *a* reconstructor.
- Test validation failures explicitly, since raising the right exception type is part of
  the parameter classes' contract:

  ```python
  import pytest

  def test_negative_diameter_rejected(config):
      config["lgs_wfs_parameters"]["D"] = -1
      with pytest.raises(ValueError):
          lgsWfsParameters(config, asterism_params)
  ```

- GPU code paths cannot run on the CI runner. Guard any CuPy-dependent test with
  `pytest.importorskip("cupy")`.
- **Do not use string patch targets for pyTomoAO modules.** `pyTomoAO/__init__.py`
  re-exports `tomographicReconstructor` and `fitting` under the same names as their
  modules, so `"pyTomoAO.fitting.fitting"` resolves to the class, not the module. On
  Python 3.11+ `unittest.mock` resolves modules first and it happens to work; on 3.9 and
  3.10 mock walks attributes and the patch fails. Patch the object instead:

  ```python
  import importlib

  reconstructor_module = importlib.import_module("pyTomoAO.tomographicReconstructor")
  with patch.object(reconstructor_module, "atmosphereParameters"):
      ...
  ```

  Note that `import pyTomoAO.tomographicReconstructor as m` binds the *class* for the same
  reason — use `importlib.import_module` when you need the module object.

:::{note}
Coverage depends on which backend the machine selects. `tomographyUtilsCPU` and
`tomographyUtilsGPU` are alternative implementations chosen at import, so whichever one is
not loaded reports 0%: the CI runner has no GPU and covers the CPU kernels, while a
CUDA-equipped workstation covers the GPU kernels instead. Construct a reconstructor with
`force_cpu=True` if you want to exercise the CPU path on a GPU machine.
:::
