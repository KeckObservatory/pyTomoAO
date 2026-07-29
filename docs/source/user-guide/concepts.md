# Concepts

## The problem

A single wavefront sensor measures the turbulence integrated along one line of sight. That
is enough to correct the star you are looking at, but the correction degrades quickly off
axis — the anisoplanatic error. Tomographic adaptive optics attacks this by observing
several guide stars at once: because each guide star samples a different cone through the
atmosphere, the combined measurements constrain the three-dimensional distribution of
turbulence.

pyTomoAO takes the slope measurements from $N$ Shack–Hartmann wavefront sensors, each
looking at a laser or natural guide star, and estimates the phase in the science direction.

## The estimator

Reconstruction uses a **Minimum Mean Square Error** (MMSE) estimator. Writing $s$ for the
concatenated slope vector and $\phi$ for the phase to estimate, the reconstructor is

$$
R = C_{\phi s} \left( C_{ss} + C_n \right)^{-1}
$$

where $C_{\phi s}$ is the cross-covariance between the target phase and the measurements,
$C_{ss}$ the measurement auto-covariance and $C_n$ the noise covariance. Both covariance
terms follow from the atmospheric model: a stack of frozen-flow layers, each with its own
altitude, fractional $r_0$ and outer scale $L_0$, propagated through the guide star and
science directions defined by the asterism geometry.

In practice pyTomoAO computes:

`auto_correlation`
: The slope-to-slope covariance $C_{ss}$ across all WFS pairs, formed from the layer model
  and the LGS cone geometry.

`cross_correlation`
: The phase-to-slope covariance $C_{\phi s}$ between the optimisation directions and each
  WFS.

The inversion is regularised by the `alpha` argument of
{py:meth}`~pyTomoAO.reconstructor.tomographicReconstructor.build_reconstructor`,
which trades noise propagation against fitting the measurements closely.

## Laser guide stars and the cone effect

A sodium laser guide star is at a finite altitude (typically 90 km), so the beam samples a
cone rather than a cylinder through the atmosphere. pyTomoAO accounts for this by scaling
each layer's sampling with altitude — `lgsAsterismParameters` derives the effective LGS
height from `baseLGSHeight` and the zenith angle, and the covariance kernels use the scaled
coordinates. Setting a very large `baseLGSHeight` recovers the NGS (cylindrical) case.

## From phase to commands

The MMSE estimate lives on a pupil-plane grid, not on the deformable mirror. Converting one
to the other is the **fitting** step: pyTomoAO models each actuator with a double-Gaussian
influence function, assembles them into a modal matrix, and takes its pseudo-inverse. The
result is folded into the reconstructor so that a single matrix multiplication turns raw
slopes into DM commands — which is what a real-time controller needs.

If you already have a measured **interaction matrix** for your system, you can skip the
influence function model entirely and build an IM-based reconstructor instead. That path
inherits any misregistration and non-linearity captured by the measurement, at the cost of
needing a good IM.

## The objects you will use

| Object                                                                        | Role                                                       |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------- |
| {py:class}`~pyTomoAO.reconstructor.tomographicReconstructor`        | Entry point; owns the configuration and builds matrices     |
| {py:class}`~pyTomoAO.atmosphereParametersClass.atmosphereParameters`           | Layered turbulence model, airmass-corrected                 |
| {py:class}`~pyTomoAO.lgsAsterismParametersClass.lgsAsterismParameters`         | Guide star positions, wavelength and altitude               |
| {py:class}`~pyTomoAO.lgsWfsParametersClass.lgsWfsParameters`                   | Lenslet array geometry and valid subaperture map            |
| {py:class}`~pyTomoAO.dmParametersClass.dmParameters`                           | Actuator grid, pitch, cross-coupling and validity map       |
| {py:class}`~pyTomoAO.tomographyParametersClass.tomographyParameters`           | Optimisation field of view and source sampling              |
| {py:class}`~pyTomoAO.dm_fitting.fitting`                                          | Influence functions and the phase-to-command projection     |

Each parameter class validates its inputs on construction and raises `ValueError` or
`TypeError` with a message naming the offending key, so a malformed configuration fails
immediately rather than producing a silently wrong reconstructor.

## Reaching the parameters

Each parameter object hangs off the reconstructor under its own name, and that is where its
values live:

```python
rec.atmParams.altitude          # layer altitudes, metres
rec.lgsWfsParams.nValidSubap    # subapertures per sensor
rec.dmParams.validActuators     # actuator map
```

Four names are also available directly on the reconstructor, because they are the ones
routinely adjusted between builds: `nLGS`, `r0`, `r0_zenith` and `L0`. Setting `nLGS`
updates every parameter object that tracks it.

:::{note}
Assigning a name the reconstructor does not recognise raises `AttributeError`. Before 2.0
an unknown name was silently accepted, so a typo such as `rec.r0_zenit = 0.1` created a new
attribute and the reconstructor went on to build with the previous `r0`.
:::

## Coordinate and unit conventions

- Altitudes are given in **kilometres** in the configuration and converted to metres
  internally; `atmParams.altitude` returns metres, `atmParams.altitude_km` kilometres.
- `r0` in the configuration is the value **at zenith**; `atmParams.r0` returns the value
  along the line of sight, corrected by the airmass derived from `zenithAngleInDeg`.
- Guide star radius (`radiusAst`) and the optimisation field (`fovOptimization`) are in
  **arcseconds**; wind directions in **degrees**.
- Wavelengths are in **metres** (`5.0e-7` for 500 nm).
