# DM fitting

The model-based reconstructor produces a phase map on the pupil grid. Turning it into
actuator commands is the job of {py:class}`~pyTomoAO.fitting.fitting`, which pyTomoAO drives
for you through
{py:meth}`~pyTomoAO.tomographicReconstructor.tomographicReconstructor.assemble_reconstructor_and_fitting`.

```python
FR = reconstructor.assemble_reconstructor_and_fitting(
    nChannels=4,
    slopesOrder="simu",
    scalingFactor=1.65e7,
    stretch_factor=1.03,
)
```

The result, also available as `reconstructor.FR`, maps a raw slope vector straight to DM
commands:

```python
commands = FR @ slopes
```

## What the assembly step does

1. Builds the influence functions for every valid actuator at the resolution of the
   reconstruction grid (`gridMask.shape[0]`).
2. Optionally rotates and flips them to match the DM's orientation relative to the WFS.
3. Masks them to the pupil and pseudo-inverts the resulting modal matrix to get the fitting
   matrix `fit.F`.
4. Reorders the reconstructor's columns to match the slope ordering of your system.
5. Combines the two: `FR = -F @ R * scalingFactor`.

## Influence functions

Actuators are modelled with a **difference of two normalised Gaussians**,

$$
f(r) = \frac{w_1}{2\pi\sigma_1^2} e^{-r^2 / 2\sigma_1^2}
     + \frac{w_2}{2\pi\sigma_2^2} e^{-r^2 / 2\sigma_2^2},
$$

which reproduces the slightly negative outer skirt typical of a stack-array or voice-coil
DM. With the defaults $w_1 = 2$ and $w_2 = -1$ the second term subtracts a broader pedestal
from a narrow core.

`sigma1`, `sigma2`
: Widths of the two components, in **grid pixels** of the reconstruction grid.
  `assemble_reconstructor_and_fitting` uses `sigma1=1.0`, `sigma2=1.7`.

`stretch_factor`
: Scales the actuator grid relative to the pupil (default `1.03`). Use it to compensate for
  a DM whose projected pitch does not exactly match the nominal pupil sampling. Small
  changes here have a noticeable effect on edge actuators.

Actuator positions come from `dmParams.validActuatorsSupport` — the valid-actuator map
padded by two elements on each side — remapped onto the reconstruction grid.

:::{note}
`dmCrossCoupling` is validated and stored by
{py:class}`~pyTomoAO.dmParametersClass.dmParameters`, but the current influence function
model does not consume it: coupling is controlled by `sigma1`/`sigma2` instead. Tune those
if you need to match a measured coupling value.
:::

To inspect the influence functions directly:

```python
from pyTomoAO.fitting import fitting

fit = fitting(reconstructor.dmParams)
modes = fit.set_influence_function(resolution=49, display=True)
```

## Slope ordering

`slopesOrder` must match how your wavefront sensor concatenates its measurements. This is
the setting most likely to bite you, because every option produces a matrix of the same
shape:

`"simu"` (default)
: `[all X slopes, all Y slopes]` per sensor — the convention used by most simulations.

`"keck"`
: `[Xi, Yi, Xi+1, Yi+1, ...]` — X and Y interleaved per subaperture, as delivered by the
  Keck real-time controller.

`"inverted"`
: `[all Y slopes, all X slopes]`.

Anything else raises `ValueError`.

`nChannels` is the number of WFS channels in the slope vector. Setting `nChannels=1`
truncates the reconstructor to the first sensor's subapertures, which is how you use a
tomographic configuration on a single-WFS system.

## Scaling

`scalingFactor` converts the reconstructor's internal units into whatever your DM expects
(volts, microns, normalised stroke). The default `1.65e7` is a Keck-specific value — treat
it as a placeholder until you have calibrated your own.

A practical way to set it: apply a known aberration to the system, run the resulting slopes
through `FR`, and compare the commands to the ones your existing controller produces. The
ratio is your scaling factor.

## Orientation

If the DM is clocked or mirrored with respect to the WFS, correct it during assembly rather
than by permuting matrices afterwards:

```python
FR = reconstructor.assemble_reconstructor_and_fitting(
    rotation=1,      # number of 90-degree rotations: 0, 1, 2 or 3
    flip=True,       # additionally flip the modes vertically
)
```

## Masking actuators

Dead or slaved actuators can be zeroed out after assembly:

```python
reconstructor.mask_DM_actuators(174)             # single actuator
reconstructor.mask_DM_actuators([12, 174, 200])  # several
FR = reconstructor.FR
```

The method zeroes the corresponding rows of `FR` for a model-based reconstructor, or of the
reconstructor itself in the IM-based case, and raises `ValueError` if the relevant matrix
has not been built yet.

## Fitting a phase map directly

For a one-off projection — for example fitting a simulated OPD map onto the DM — use the
fitting object on its own:

```python
commands = reconstructor.fit.fit(opd_map)
```

`fit.F` must have been computed first, which `assemble_reconstructor_and_fitting` does.
