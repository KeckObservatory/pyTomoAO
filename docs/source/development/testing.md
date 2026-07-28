# Testing

## Running the suite

```bash
pip install pytest pytest-cov
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

Pull requests run `.github/workflows/test.yml`, which calls a wrapper script rather than
pytest directly:

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
