# API reference

Generated from the source docstrings. If you are looking for how the pieces fit together,
start with {doc}`../user-guide/concepts` instead.

## At a glance

| Class                                                                    | What it does                                                    |
| ------------------------------------------------------------------------ | --------------------------------------------------------------- |
| {py:class}`~pyTomoAO.reconstructor.tomographicReconstructor`   | Loads the configuration and builds the reconstruction matrices   |
| {py:class}`~pyTomoAO.dm_fitting.fitting`                                    | Influence functions and the phase-to-command projection          |
| {py:class}`~pyTomoAO.atmosphereParametersClass.atmosphereParameters`     | Layered turbulence model with airmass corrections                |
| {py:class}`~pyTomoAO.lgsAsterismParametersClass.lgsAsterismParameters`   | Guide star asterism geometry and direction vectors               |
| {py:class}`~pyTomoAO.lgsWfsParametersClass.lgsWfsParameters`             | Shack–Hartmann geometry and valid subaperture map                |
| {py:class}`~pyTomoAO.dmParametersClass.dmParameters`                     | Deformable mirror actuator geometry                              |
| {py:class}`~pyTomoAO.tomographyParametersClass.tomographyParameters`     | Optimisation field of view and source sampling                   |
| {py:class}`~pyTomoAO.slodar.slodarProfiler`                              | SLODAR turbulence profiling from multi-WFS slopes                |
| {py:func}`~pyTomoAO.slope_conversion.slope_conversion_matrix`            | Converts external slopes (e.g. SPECULA) to the reconstructor's convention |

All these classes are re-exported at package level, so either import style works:

```python
from pyTomoAO import tomographicReconstructor                # convenient
from pyTomoAO.reconstructor import tomographicReconstructor  # explicit
```

Module names are distinct from the class names, so `import pyTomoAO.reconstructor` gives
the module and `from pyTomoAO import tomographicReconstructor` gives the class.

## Classes

Each class gets its own page with every public method and property.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   pyTomoAO.reconstructor.tomographicReconstructor
   pyTomoAO.dm_fitting.fitting
   pyTomoAO.atmosphereParametersClass.atmosphereParameters
   pyTomoAO.lgsAsterismParametersClass.lgsAsterismParameters
   pyTomoAO.lgsWfsParametersClass.lgsWfsParameters
   pyTomoAO.dmParametersClass.dmParameters
   pyTomoAO.tomographyParametersClass.tomographyParameters
   pyTomoAO.slodar.slodarProfiler
   pyTomoAO.slodar.slodarResult
```

## Functions

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   pyTomoAO.slope_conversion.slope_conversion_matrix
   pyTomoAO.slope_conversion.specula_slope_scale
   pyTomoAO.slodar.pseudo_open_loop_slopes
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
