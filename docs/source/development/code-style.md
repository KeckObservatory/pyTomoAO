# Code style

pyTomoAO uses [ruff](https://docs.astral.sh/ruff/) for both linting and formatting. The
configuration lives in `pyproject.toml`, and the `Code health` workflow enforces it on every
pull request.

## Running the checks

```bash
pip install -e ".[dev]"

ruff check .            # lint
ruff check --fix .      # lint and apply the safe fixes
ruff format .           # format
ruff format --check .   # verify formatting without writing
```

CI runs exactly these two commands, with the ruff version pinned in
`.github/workflows/code-health.yml`:

```bash
ruff check --output-format=github .
ruff format --check --diff .
```

If CI disagrees with your local run, check that your ruff version matches the pin — the
`[dev]` extra installs the same one.

## Formatting

The whole codebase is formatted by `ruff format` at a 100-column line length. Do not
hand-format around it; run the formatter and commit the result. Code blocks inside
docstrings are formatted too (`docstring-code-format`), so examples in the API reference
stay consistent with the code around them.

## Lint rules

The enabled rule families are pyflakes, pycodestyle, isort, pyupgrade, bugbear,
comprehensions, pie, return, simplify, unused-arguments, pylint and ruff's own rules. A few
are switched off deliberately, and the reasons are recorded in `pyproject.toml`:

**Naming rules (`N`) are not enabled.**
: This package is a translation of MATLAB adaptive-optics code. Class names such as
  `tomographicReconstructor` are public API that user scripts import, and matrix names such
  as `Gamma`, `Cxx`, `Cox` and `R` match the equations they implement. Renaming them would
  break users and make the mathematics harder to follow.

**`PLR2004` (magic values), `PLR0912`/`PLR0913`/`PLR0915` (branches, arguments, statements).**
: Numerical kernels are long, linear routines full of meaningful constants. Splitting them
  up to satisfy a counter would hurt readability.

**`PLC0415` (imports inside functions) and `PLW0603` (global statement).**
: The CPU/GPU backend is chosen at import time through a `try`/`except` around the CuPy
  import and a module-level flag.

`sandbox/` is excluded entirely: it holds exploratory scripts that are not packaged or
tested, and gating real work on them would be noise.

## Suppressing a rule

Prefer fixing the code. When a violation is genuinely intentional, silence it on the
specific line with a reason next to it:

```python
# use_float32 is unused here but kept so the GPU and CPU kernels share a signature.
def _rotateWFS(px_gpu, py_gpu, rotAngleInRadians, use_float32=False):  # noqa: ARG001
```

A blanket `# noqa` with no rule code will itself be flagged. If a rule is wrong for the
whole project rather than one line, change `pyproject.toml` and explain why in a comment —
future readers need the reasoning more than the rule.

## Conventions the linter cannot check

See {doc}`contributing` for the patterns that matter but are not mechanically enforced:
validation in property setters, derived values as properties, NumPy-style docstrings, and
keeping the CPU and GPU kernels in lockstep.
