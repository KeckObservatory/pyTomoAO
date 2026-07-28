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
pip install -e ".[docs]"
pip install pytest pytest-cov
```
