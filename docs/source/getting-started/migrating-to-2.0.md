# Migrating to 2.0

2.0 corrects a wavefront-orientation bug, tightens the public API and makes `matplotlib`
optional. This page lists everything that can require a change in your own code. The
{doc}`changelog <../changelog>` has the full detail and the reasoning behind each one.

## Reconstructed wavefronts are no longer transposed

**This is the change most likely to affect results rather than imports.**

The gradient operator indexed the reconstruction grid in Fortran order — a MATLAB port
artefact — while masking it in C order, and `reconstruct_wavefront` scattered its output in
C order. The array it returned was therefore the transpose of the real wavefront. Only
`visualize_reconstruction` compensated, by plotting `reconstructed_wavefront.T`, so plots
looked right while the returned array did not.

```python
wavefront = rec.reconstruct_wavefront(slopes)   # now correctly oriented
```

What to check:

- **Plots are unaffected** — the compensating transpose was removed together with the bug.
- **A reference wavefront saved with 1.x needs transposing** before you compare against it.
- **Non-symmetric pupils were affected more than orientation.** Where the valid-lenslet map
  is not symmetric under transpose, the old gradient operator was simply wrong: a flat
  wavefront produced non-zero slopes. If you use a vignetted or segmented aperture, previous
  results are not trustworthy.

## Two modules were renamed

| 1.x | 2.0 |
| --- | --- |
| `pyTomoAO.tomographicReconstructor` | `pyTomoAO.reconstructor` |
| `pyTomoAO.fitting` | `pyTomoAO.dm_fitting` |

```python
# before
from pyTomoAO.tomographicReconstructor import tomographicReconstructor
from pyTomoAO.fitting import fitting

# after
from pyTomoAO.reconstructor import tomographicReconstructor
from pyTomoAO.dm_fitting import fitting
```

The package-level imports are unchanged, and are the form used throughout the
documentation:

```python
from pyTomoAO import tomographicReconstructor, fitting   # works in both
```

Class names are untouched.

## Parameters are read from the object that owns them

The reconstructor no longer forwards arbitrary attributes to the parameter objects, and
assigning a name it does not recognise now raises `AttributeError` instead of silently
creating one.

```python
# before
rec.altitude
rec.nValidSubap
rec.validActuators

# after
rec.atmParams.altitude
rec.lgsWfsParams.nValidSubap
rec.dmParams.validActuators
```

Four names remain directly on the reconstructor, because they are the ones routinely
adjusted between builds:

```python
rec.nLGS        # setting this updates every parameter object that tracks it
rec.r0          # read-only; derived from r0_zenith and the zenith angle
rec.r0_zenith
rec.L0
```

The upside of the stricter behaviour:

```python
>>> rec.r0_zenit = 0.1      # a typo
AttributeError: 'tomographicReconstructor' object has no attribute 'r0_zenit'
```

In 1.x that was accepted, and the reconstructor went on to build with the previous `r0`.

## Configurations ship inside the package

The reference YAML files moved from `examples/benchmark/` into the package, so a `pip
install` is enough to run the documented examples:

```python
# before
rec = tomographicReconstructor("examples/benchmark/tomography_config_kapa.yaml")

# after
from pyTomoAO import example_config
rec = tomographicReconstructor(example_config("kapa"))
```

`list_example_configs()` returns `['kapa', 'kapa-single-channel', 'keck', 'revolt']`.
`example_config` returns a path inside the installed package, so copy one before editing it.

## matplotlib is an optional dependency

It is needed only by `visualize_reconstruction`, `visualize_commands` and the
`display=True` branch of `set_influence_function`:

```bash
pip install "pyTomoAO[plot]"
```

Calling one of those without it raises an `ImportError` naming the extra. Everything else —
building reconstructors, fitting, reconstructing wavefronts — has no matplotlib dependency.

## Smaller changes

- `force_cpu=True` now actually selects the CPU kernels. In 1.x it flipped a module-level
  flag while the GPU functions stayed bound, so it logged that it was forcing the CPU and
  then ran on the GPU, returning `cupy.ndarray` intermediates. It also changed the backend
  for every other reconstructor in the process; it no longer does. The backend in use is
  readable as `rec.backend`.
- `_test_against_matlab` is removed. It read an attribute that was never assigned, so every
  comparison it made failed into its own exception handler.
- The benchmark scripts under `examples/benchmark/` were replaced by `benchmark.py`, which
  drives the package rather than a forked copy of its kernels.

## Results will shift slightly

Two corrections move reconstructor values by small amounts. Neither is a regression — both
replace a wrong value with a right one — but pinned numerical references will need updating:

- The `K_{5/6}` Bessel kernel had a coefficient that was 8× too small, a series/asymptotic
  crossover set far too low, and a truncated `Γ(11/6)`. Worst-case error against
  `scipy.special.kv` drops from 2.1e-3 to 3.6e-8, moving mean reconstructed OPD by
  0.1–0.4% on the bundled configurations.
- Zero-separation covariance entries on the GPU were 1.887× too large.
