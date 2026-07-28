# API reference

Generated from the source docstrings. If you are looking for how the pieces fit together,
start with {doc}`../user-guide/concepts` instead.

## At a glance

| Class                                                                    | What it does                                                    |
| ------------------------------------------------------------------------ | --------------------------------------------------------------- |
| {py:class}`~pyTomoAO.tomographicReconstructor.tomographicReconstructor`   | Loads the configuration and builds the reconstruction matrices   |
| {py:class}`~pyTomoAO.fitting.fitting`                                    | Influence functions and the phase-to-command projection          |
| {py:class}`~pyTomoAO.atmosphereParametersClass.atmosphereParameters`     | Layered turbulence model with airmass corrections                |
| {py:class}`~pyTomoAO.lgsAsterismParametersClass.lgsAsterismParameters`   | Guide star asterism geometry and direction vectors               |
| {py:class}`~pyTomoAO.lgsWfsParametersClass.lgsWfsParameters`             | Shack–Hartmann geometry and valid subaperture map                |
| {py:class}`~pyTomoAO.dmParametersClass.dmParameters`                     | Deformable mirror actuator geometry                              |
| {py:class}`~pyTomoAO.tomographyParametersClass.tomographyParameters`     | Optimisation field of view and source sampling                   |

All seven classes are re-exported at package level, so either import style works:

```python
from pyTomoAO import tomographicReconstructor              # convenient
from pyTomoAO.tomographicReconstructor import tomographicReconstructor   # explicit
```

## Classes

Each class gets its own page with every public method and property.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   pyTomoAO.tomographicReconstructor.tomographicReconstructor
   pyTomoAO.fitting.fitting
   pyTomoAO.atmosphereParametersClass.atmosphereParameters
   pyTomoAO.lgsAsterismParametersClass.lgsAsterismParameters
   pyTomoAO.lgsWfsParametersClass.lgsWfsParameters
   pyTomoAO.dmParametersClass.dmParameters
   pyTomoAO.tomographyParametersClass.tomographyParameters
```

## Internal computation kernels

`pyTomoAO.tomographyUtilsCPU` and `pyTomoAO.tomographyUtilsGPU` implement the covariance
and reconstructor kernels behind matching private interfaces. They are selected
automatically at import time and are not part of the public API — their signatures may
change without notice.

```{toctree}
:maxdepth: 1

internals
```
