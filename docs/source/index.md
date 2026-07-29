---
sd_hide_title: true
---

# pyTomoAO

:::{div} sd-text-center sd-fs-1 sd-font-weight-bold
pyTomoAO
:::

:::{div} sd-text-center sd-fs-5 sd-text-muted
Tomographic wavefront reconstruction for adaptive optics, in Python.
:::

```{div} sd-text-center
[![Tests](https://github.com/KeckObservatory/pyTomoAO/actions/workflows/test.yml/badge.svg)](https://github.com/KeckObservatory/pyTomoAO/actions/workflows/test.yml)
[![Docs](https://github.com/KeckObservatory/pyTomoAO/actions/workflows/docs.yml/badge.svg)](https://github.com/KeckObservatory/pyTomoAO/actions/workflows/docs.yml)
[![PyPI](https://img.shields.io/pypi/v/pyTomoAO.svg)](https://pypi.org/project/pyTomoAO/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/KeckObservatory/pyTomoAO/blob/main/LICENSE)
```

**pyTomoAO** computes reconstructors for tomographic adaptive optics systems — LTAO and
MOAO — from the measurements of several Shack–Hartmann wavefront sensors coupled to laser
guide stars (LGS) or natural guide stars (NGS). It uses a Minimum Mean Square Error (MMSE)
estimator to reconstruct the turbulent volume above a telescope and to derive the
deformable mirror (DM) commands that correct it.

Everything is driven by a single YAML configuration file:

```python
from pyTomoAO import example_config
from pyTomoAO.reconstructor import tomographicReconstructor

reconstructor = tomographicReconstructor(example_config("kapa"))
reconstructor.build_reconstructor()
FR = reconstructor.assemble_reconstructor_and_fitting(nChannels=4, slopesOrder="simu")
```

---

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.2em` Getting started
:link: getting-started/installation
:link-type: doc

Install pyTomoAO and build your first reconstructor in a few minutes.
:::

:::{grid-item-card} {octicon}`book;1.2em` User guide
:link: user-guide/index
:link-type: doc

Concepts, the configuration reference, reconstruction modes, DM fitting and GPU support.
:::

:::{grid-item-card} {octicon}`beaker;1.2em` Tutorials
:link: tutorials/index
:link-type: doc

Worked end-to-end examples, including an LTAO reconstructor for KAPA.
:::

:::{grid-item-card} {octicon}`code;1.2em` API reference
:link: api/index
:link-type: doc

Generated documentation for every public class, method and property.
:::

::::

## Highlights

- **Two reconstruction modes** — model-based (reconstruct the phase, then fit the DM) and
  interaction-matrix-based (go straight to DM commands).
- **Configuration driven** — atmosphere, asterism, WFS, DM and tomography parameters all
  live in one YAML file, validated on load.
- **CPU and GPU** — covariance matrices are computed with NumPy/Numba, or with CuPy when a
  CUDA device is available. The GPU path is selected automatically.
- **Not just tomography** — the same machinery builds reconstructors for single-WFS
  systems.

## Project status

pyTomoAO is under active development at the W. M. Keck Observatory. SLODAR-based turbulence
profiling and MCAO reconstructors are on the roadmap; see the
[issue tracker](https://github.com/KeckObservatory/pyTomoAO/issues) for what is in flight.

```{toctree}
:hidden:
:caption: Getting started

getting-started/installation
getting-started/quickstart
getting-started/migrating-to-2.0
```

```{toctree}
:hidden:
:caption: User guide

user-guide/index
user-guide/concepts
user-guide/configuration
user-guide/reconstruction
user-guide/fitting
user-guide/gpu
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/index
tutorials/ltao-kapa
```

```{toctree}
:hidden:
:caption: Reference

api/index
changelog
```

```{toctree}
:hidden:
:caption: Development

development/index
development/contributing
development/code-style
development/testing
development/documentation
development/releasing
```

```{toctree}
:hidden:
:caption: Links

GitHub repository <https://github.com/KeckObservatory/pyTomoAO>
PyPI package <https://pypi.org/project/pyTomoAO/>
```
