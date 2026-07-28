# Configuration reference

Everything pyTomoAO needs comes from a single YAML file, passed to
{py:class}`~pyTomoAO.tomographicReconstructor.tomographicReconstructor`. The file has five
sections, each of which maps onto one parameter class:

| YAML section            | Parameter class                                                       |
| ----------------------- | --------------------------------------------------------------------- |
| `atmosphere_parameters` | {py:class}`~pyTomoAO.atmosphereParametersClass.atmosphereParameters`   |
| `lgs_asterism`          | {py:class}`~pyTomoAO.lgsAsterismParametersClass.lgsAsterismParameters` |
| `lgs_wfs_parameters`    | {py:class}`~pyTomoAO.lgsWfsParametersClass.lgsWfsParameters`           |
| `dm_parameters`         | {py:class}`~pyTomoAO.dmParametersClass.dmParameters`                   |
| `tomography_parameters` | {py:class}`~pyTomoAO.tomographyParametersClass.tomographyParameters`   |

All five sections are required. Values are validated as they are read, and an invalid entry
raises `TypeError` (wrong type) or `ValueError` (out of range) naming the parameter.

:::{tip}
Start from one of the configurations bundled with the package rather than writing a file
from scratch — `validLLMap` and `validActuators` are hand-authored 2D maps of several
hundred entries each.

```python
import shutil
from pyTomoAO import example_config, list_example_configs

print(list_example_configs())        # ['kapa', 'kapa-single-channel', 'keck', 'revolt']
shutil.copy(example_config("kapa"), "my_config.yaml")
```

`example_config` returns a path inside the installed package, so treat it as read-only and
edit a copy.
:::

## `atmosphere_parameters`

The layered turbulence model. Array-valued keys must all have `nLayer` entries.

`nLayer` (int)
: Number of turbulence layers. Must be positive.

`zenithAngleInDeg` (float)
: Zenith angle of the observation in degrees. Drives the airmass, which scales layer
  altitudes and the LGS height, and de-rates `r0`.

`altitude` (list of float, **km**)
: Layer altitudes at zenith. `atmParams.altitude` returns these in metres scaled by the
  airmass; `atmParams.altitude_km` returns the unscaled kilometres.

`r0` (float, m)
: Fried parameter **at zenith** and at `wavelength`. The line-of-sight value is derived as
  $r_0 \cos(z)^{3/5}$ and exposed as `atmParams.r0`; the input value stays available as
  `atmParams.r0_zenith`.

`L0` (float, m)
: Outer scale of turbulence, shared by all layers.

`fractionnalR0` (list of float)
: Fraction of the total turbulence strength in each layer. Should sum to 1.

`wavelength` (float, m)
: Wavelength at which `r0` is specified, e.g. `5.0e-7` for 500 nm.

`windSpeed` (list of float, m/s)
: Wind speed per layer. Not used by the static reconstructor, but carried for profiling and
  future predictive control.

`windDirection` (list of float, degrees)
: Wind direction per layer. Exposed in radians as `atmParams.windDirection`, with the
  Cartesian components available as `windVx` / `windVy`.

```yaml
atmosphere_parameters:
  nLayer: 7
  zenithAngleInDeg: 30.0
  altitude: [0, 0.5, 1, 2, 4, 8, 16]        # km
  L0: 30
  r0: 0.186                                  # m, at zenith
  fractionnalR0: [0.4557, 0.1295, 0.0442, 0.0506, 0.1167, 0.0926, 0.1107]
  wavelength: 5.0e-7
  windDirection: [190, 255, 270, 350, 17, 29, 66]
  windSpeed: [5.6, 5.77, 6.25, 7.57, 13.31, 19.06, 12.14]
```

## `lgs_asterism`

Guide star geometry. The asterism is a regular polygon: `nLGS` stars are placed at equal
azimuth spacing on a circle of radius `radiusAst`.

`nLGS` (int)
: Number of guide stars. This is the authoritative count — the WFS section inherits it.

`radiusAst` (float, arcsec)
: Angular radius of the asterism.

`LGSwavelength` (float, m)
: Guide star wavelength, e.g. `5.89e-7` for sodium.

`baseLGSHeight` (float, m)
: Guide star altitude at zenith, e.g. `90000.0` for the sodium layer. The effective height
  used in the cone-effect calculation is `baseLGSHeight × airmass`. For natural guide stars,
  set this to a very large value so the cone approaches a cylinder.

```yaml
lgs_asterism:
  radiusAst: 7.6            # arcsec
  LGSwavelength: 5.89e-7
  baseLGSHeight: 90000.0    # m
  nLGS: 4
```

## `lgs_wfs_parameters`

The Shack–Hartmann geometry, shared by all guide stars.

`D` (float, m)
: Telescope pupil diameter.

`nLenslet` (int)
: Lenslets across the pupil. `validLLMap` should be `nLenslet × nLenslet`; only
  two-dimensionality is enforced, so a mismatch will surface later as a shape error rather
  than at load time.

`nPx` (int)
: Detector pixels per lenslet.

`fieldStopSize` (float, arcsec)
: Field stop size per subaperture.

`validLLMap` (2D list of 0/1)
: Which lenslets are illuminated. Rows are outer lists; `1` marks a valid subaperture. The
  number of ones sets `nValidSubap`, and therefore the length of each WFS's slope vector
  (`2 × nValidSubap`).

`wfsLensletsRotation` (list of float, radians, optional)
: Per-WFS clocking of the lenslet array relative to the pupil. Defaults to zeros. Must have
  `nLGS` entries.

`wfsLensletsOffset` (2 × `nLGS` list of float, normalised, optional)
: Per-WFS lateral misregistration as a fraction of the pupil, given as `[[x...], [y...]]`.
  Defaults to zeros.

:::{note}
Some example configurations also carry an `nLGS` key inside `lgs_wfs_parameters`. It is
ignored — the guide star count always comes from `lgs_asterism.nLGS` so that the two
sections cannot disagree.
:::

```yaml
lgs_wfs_parameters:
  D: 10
  nLenslet: 20
  nPx: 8
  fieldStopSize: 4
  validLLMap:
    - [0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0]
    # ... 20 rows of 20 entries
  wfsLensletsRotation: [0, 0, 0, 0]
  wfsLensletsOffset:
    - [0, 0, 0, 0]     # x offsets
    - [0, 0, 0, 0]     # y offsets
```

## `dm_parameters`

The deformable mirror. Each key is a list so that multi-DM (MCAO) configurations can be
expressed once that support lands; today a single entry is expected.

`nActuators` (list of int)
: Actuators across the DM. `validActuators` should be `nActuators × nActuators`; as with
  the lenslet map, only two-dimensionality is checked on load.

`dmPitch` (list of float, m)
: Inter-actuator spacing projected onto the pupil.

`dmHeights` (list of float, m)
: Conjugation altitude of the mirror; `0.0` for a pupil-conjugated DM.

`dmCrossCoupling` (float)
: Mechanical coupling between neighbouring actuators, typically 0.1–0.3. Used when building
  the influence functions.

`validActuators` (2D list of 0/1)
: Which actuators are controlled. The count of ones determines the length of the command
  vector produced by the fitting step.

```yaml
dm_parameters:
  dmHeights: [0.0]
  dmPitch: [0.5]
  dmCrossCoupling: 0.15
  nActuators: [21]
  validActuators:
    - [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0]
    # ... 21 rows of 21 entries
```

## `tomography_parameters`

Where in the field the reconstruction is optimised.

`nFitSrc` (int)
: Number of optimisation sources per axis. The sources form an `nFitSrc × nFitSrc` grid
  across the optimisation field. `1` optimises on axis.

`fovOptimization` (float, arcsec)
: Side length of the square optimisation field. Must be `0` when `nFitSrc` is 1, and
  positive when `nFitSrc` is greater than 1 — otherwise construction raises `ValueError`.

```yaml
tomography_parameters:
  fovOptimization: 0    # arcsec; 0 = on-axis optimisation
  nFitSrc: 1
```

Widening `fovOptimization` trades peak on-axis performance for a more uniform correction
across the field — the LTAO-versus-GLAO knob.

## Unused sections

The bundled example files also contain a `noise_parameters` section with an `iNoiseVar`
key. It is a leftover from an earlier interface and is **not read** by the current code:
measurement-noise regularisation is controlled by the `alpha` argument of
{py:meth}`~pyTomoAO.tomographicReconstructor.tomographicReconstructor.build_reconstructor`.
Leaving the section in place is harmless — any key pyTomoAO does not recognise is ignored.

## Inspecting a loaded configuration

Each parameter object has a readable `__str__`, which is the fastest way to confirm that a
file was interpreted the way you intended:

```python
rec = tomographicReconstructor("my_config.yaml")

print(rec.atmParams)           # layer table, airmass, derived r0
print(rec.lgsAsterismParams)   # asterism geometry, effective LGS height
print(rec.lgsWfsParams)        # lenslet counts, nValidSubap
print(rec.dmParams)            # actuator counts, pitch
print(rec.tomoParams)          # optimisation grid
```

Derived quantities are computed on demand, so changing a parameter in place is picked up by
everything downstream:

```python
rec.atmParams.zenithAngleInDeg = 45.0
print(rec.atmParams.r0)        # line-of-sight r0 updates with the airmass
```

Matrices already built are *not* invalidated automatically — rebuild with
`build_reconstructor()` after changing parameters.
