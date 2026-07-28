# Contributing

## Branching model

`main`
: Released, stable code. Documentation published to GitHub Pages is built from this branch.

`dev`
: Integration branch. Feature branches merge here first, and `dev` is merged into `main`
  when a release is cut.

Feature branches
: Named after the issue they address, e.g. `30-read-the-docs-hook-up`. Branch from `dev`
  and open the pull request against `dev`.

## Making a change

1. Open (or claim) an issue describing the change.
2. Branch from `dev`.
3. Make the change, with tests for anything that could regress.
4. Run the checks locally:

   ```bash
   pytest
   ruff check .
   ruff format .
   ```

   See {doc}`testing` and {doc}`code-style` for what these enforce.
5. Update the documentation if behaviour or configuration keys changed, and add an entry to
   the `Unreleased` section of `CHANGELOG.md`.
6. Open a pull request against `dev` and reference the issue.

Every pull request runs the test suite, the lint and format checks, and a documentation
build. All three must pass.

## Coding conventions

Formatting and lint rules are enforced by ruff and described in {doc}`code-style`. Beyond
what a linter can check, the codebase follows a few consistent patterns. Match them rather
than the style you would choose in a new project:

- **Class names are lowerCamelCase** (`tomographicReconstructor`, `dmParameters`). This is
  unusual for Python but consistent throughout, and changing it would break every existing
  configuration script. The naming lint rules are switched off for this reason.
- **Parameters are validated in property setters.** Each configuration class exposes
  properties that raise `TypeError` for the wrong type and `ValueError` for out-of-range
  values, with a message naming the parameter. New configuration keys should follow suit.
- **Derived quantities are properties, not stored state**, so that changing an input
  updates everything downstream (`atmParams.r0` recomputes from `r0_zenith` and the zenith
  angle each time it is read).
- **Docstrings are NumPy style** — `Parameters`, `Returns`, `Raises`, `Notes` sections with
  underlines. These are rendered directly into {doc}`../api/index`, so a good docstring is a
  documentation contribution.
- **Logging, not printing.** Use the module-level `logger`; the reconstructor threads a
  logger through to the fitting object.
- **CPU and GPU kernels stay in lockstep.** `tomographyUtilsCPU` and `tomographyUtilsGPU`
  expose the same private function names and signatures; a change to one needs the matching
  change in the other.

## Adding a configuration parameter

1. Add the key to the relevant `*ParametersClass`, with a validating property setter.
2. Read it in that class's `_initialize_properties`.
3. Add it to the `__str__` output so it shows up when a user prints the object.
4. Document it in {doc}`../user-guide/configuration`, including units and the valid range.
5. Add it to the example configurations under `examples/benchmark/` if it is not optional.
6. Note it in the `Unreleased` section of `CHANGELOG.md`.

## Reporting a bug

Useful bug reports include:

- The configuration file, or the part of it that matters.
- The full traceback.
- Whether CUDA was detected — the log line emitted at import when logging is enabled at
  `INFO` (`logging.basicConfig(level=logging.INFO)`).
- `pyTomoAO.__version__`, NumPy version, and Python version.

## Sandbox scripts

`sandbox/` holds exploratory work — comparison scripts, prototypes for SLODAR, WFS map
utilities. It is deliberately outside the package and is not tested or packaged. Anything
there that becomes load-bearing should move into `pyTomoAO/` with tests.
