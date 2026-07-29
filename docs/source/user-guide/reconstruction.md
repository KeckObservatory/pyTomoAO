# Building reconstructors

{py:meth}`~pyTomoAO.reconstructor.tomographicReconstructor.build_reconstructor`
is the single entry point for both reconstruction modes. Which one you get depends on
whether you pass an interaction matrix.

```python
R = reconstructor.build_reconstructor()             # model-based: slopes -> phase
R = reconstructor.build_reconstructor(IM)           # IM-based:    slopes -> commands
```

The mode is recorded in `reconstructor.method` (`"Model"` or `"IM"`), and several downstream
methods check it.

## Model-based reconstruction

The model-based path builds the MMSE estimator purely from the configuration: the layered
atmosphere, the asterism geometry and the lenslet array. Its output lives on the pupil grid
described by `reconstructor.gridMask`.

```python
R = reconstructor.build_reconstructor(alpha=10)
print(R.shape)                    # (n_valid_phase_points, 2 * nValidSubap * nLGS)
```

Along the way the object caches the intermediate matrices, which are useful when
diagnosing a configuration:

| Attribute      | Meaning                                                        |
| -------------- | -------------------------------------------------------------- |
| `Gamma`        | Sparse gradient (phase-to-slope) operator                       |
| `gridMask`     | Boolean pupil mask defining the reconstruction grid             |
| `Cxx`          | Slope-to-slope covariance across all WFS pairs                  |
| `Cox`          | Optimisation-direction-to-slope covariance                      |
| `CnZ`          | Noise covariance contribution                                   |
| `RecStatSA`    | Static reconstructor for the sub-aperture geometry              |

Because the output is a phase map, you need the fitting step to obtain DM commands — see
{doc}`fitting`.

### Reconstructing a wavefront

{py:meth}`~pyTomoAO.reconstructor.tomographicReconstructor.reconstruct_wavefront`
applies the reconstructor and reshapes the result onto the pupil grid, with `NaN` outside
the mask so that it plots cleanly:

```python
wavefront = reconstructor.reconstruct_wavefront(slopes)   # 2D array, NaN outside the pupil
fig = reconstructor.visualize_reconstruction(slopes)
```

Pass a `reference_wavefront` to `visualize_reconstruction` to get a side-by-side comparison
with the residual — the quickest way to sanity-check a new configuration against a
simulation.

:::{important}
`reconstruct_wavefront` and `assemble_reconstructor_and_fitting` require `method == "Model"`
and raise `ValueError` otherwise. An IM-based reconstructor already outputs commands, so
there is no phase to reconstruct.
:::

## Interaction-matrix-based reconstruction

If you have measured an interaction matrix for your system, pass it in:

```python
R = reconstructor.build_reconstructor(IM)
commands = R @ slopes
fig = reconstructor.visualize_commands(slopes)
```

The IM is expected to be **block diagonal**, with one block per wavefront sensor, matching
the order of the guide stars in the asterism. The DM enters the estimator through the IM,
so the influence function model is not used at all and no assembly step is needed.

Choose this path when the measured system deviates from the idealised model — pupil
misregistration, actuator coupling that differs from the double-Gaussian model, or a DM
with dead actuators whose real behaviour is captured in the measurement.

## Regularisation

`alpha` sets the weight of the noise/regularisation term in the inversion:

```python
R = reconstructor.build_reconstructor(alpha=10)     # default
```

- **Larger `alpha`** — stronger regularisation. Less noise propagation, more smoothing, and
  a reconstruction that leans harder on the atmospheric prior.
- **Smaller `alpha`** — closer fit to the measurements. Sharper, but noisier, and more
  sensitive to errors in the turbulence profile.

The useful range depends on the noise level and slope units of your system; scan it against
a simulated or on-sky data set rather than assuming the default is optimal.

## Precision

`use_float32=True` halves memory use and is typically faster on GPUs whose double-precision
throughput is limited:

```python
R = reconstructor.build_reconstructor(use_float32=True)
```

Note that when CUDA is available the GPU kernels are currently invoked with `use_float32=True`
regardless of this argument; on the CPU path the flag selects float32 versus float64. See
{doc}`gpu`.

## Caching behaviour

The `reconstructor` property builds the matrix on first access, so this is enough for a
one-liner:

```python
rec = tomographicReconstructor("config.yaml")
R = rec.reconstructor          # builds a model-based reconstructor on demand
```

`R` and `FR` are also exposed as properties — `R` is the reconstructor matrix and `FR` the
combined fitting-and-reconstruction matrix. Rebuilding after a parameter change is explicit:
call `build_reconstructor()` again.
