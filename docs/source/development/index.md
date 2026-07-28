# Development

pyTomoAO is developed in the open at
[KeckObservatory/pyTomoAO](https://github.com/KeckObservatory/pyTomoAO). Bug reports,
configuration questions and pull requests are all welcome.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`git-pull-request;1.2em` Contributing
:link: contributing
:link-type: doc

Branching model, coding conventions and the pull request checklist.
:::

:::{grid-item-card} {octicon}`sparkle-fill;1.2em` Code style
:link: code-style
:link-type: doc

The ruff lint and format gate, and why certain rules are switched off.
:::

:::{grid-item-card} {octicon}`checklist;1.2em` Testing
:link: testing
:link-type: doc

Running pytest, the coverage gate and what CI enforces.
:::

:::{grid-item-card} {octicon}`book;1.2em` Documentation
:link: documentation
:link-type: doc

Building this site locally and how it is published to GitHub Pages.
:::

:::{grid-item-card} {octicon}`tag;1.2em` Releasing
:link: releasing
:link-type: doc

Cutting a version and publishing to PyPI.
:::

::::

## Repository layout

```text
pyTomoAO/
├── pyTomoAO/           # the package
│   ├── tomographicReconstructor.py   # main entry point
│   ├── fitting.py                    # DM influence functions and fitting
│   ├── *ParametersClass.py           # configuration objects
│   └── tomographyUtils{CPU,GPU}.py   # covariance and reconstructor kernels
├── tests/              # pytest suite
├── examples/benchmark/ # example configurations and benchmark scripts
├── sandbox/            # exploratory scripts, not part of the package
└── docs/               # this documentation
```

## Development install

```bash
git clone https://github.com/KeckObservatory/pyTomoAO.git
cd pyTomoAO
pip install -e ".[docs,dev]"
```

The `dev` extra installs pytest, coverage and the pinned ruff used by CI; `docs` installs
the Sphinx toolchain.

## What CI checks

| Workflow        | Runs on                      | Gate                                                  |
| --------------- | ---------------------------- | ----------------------------------------------------- |
| `Run Pytest`    | pull requests, `main`, `dev` | Test suite on Python 3.9–3.13, plus a coverage gate    |
| `Code health`   | pull requests, `main`, `dev` | `ruff check` and `ruff format --check`                 |
| `Documentation` | pull requests, `main`, `dev` | Sphinx build with warnings as errors                   |

`Run Pytest` installs the built package (`pip install ".[dev]"`) rather than a requirements
file, so the dependency metadata users resolve is exercised on every run. The coverage gate
runs once, on 3.12.

`Documentation` additionally deploys to GitHub Pages on pushes to `main`.
`Publish Python Package to PyPI` runs when a GitHub release is created: it builds the sdist
and wheel, runs `twine check --strict`, installs the wheel into a clean virtualenv and
imports it, and only then publishes through the `pypi` environment.
