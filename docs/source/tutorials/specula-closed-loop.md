# pyTomoAO in a SPECULA closed loop

This tutorial closes an LTAO loop in the [SPECULA](https://github.com/ArcetriAdaptiveOptics/SPECULA)
end-to-end simulator, with pyTomoAO's model-based MMSE reconstructor as the real-time
reconstructor. It covers the parts that are easy to get wrong when the reconstructor and
the simulator come from different codes: matching the geometry, slope units and ordering,
the reconstruction grid, the DM fitting step and the pseudo-open-loop controller.

The system is a Keck-like LTAO configuration:

- **Telescope:** 10 m circular pupil, sampled with 160 pixels.
- **Guide stars and sensors:** six sodium LGS at 90 km on a 60″ ring, each with a 20×20
  Shack–Hartmann on the KAPA valid-subaperture map (304 subapertures, d = 0.5 m).
- **DM:** ground conjugated, 349 actuators on the KAPA 21×21 map.
- **Atmosphere:** the seven-layer Maunakea Ridge median profile (KAON 503), r0 = 0.16 m at
  500 nm, L0 = 25 m.
- **Control:** 500 Hz pseudo-open-loop control with an integrator.

Everything is scripted in `examples/specula_ltao/`. The tutorial walks through what each
step does and why.

## Setup

```bash
git clone https://github.com/KeckObservatory/pyTomoAO.git
cd pyTomoAO
pip install -e ".[plot]"
pip install specula            # tested with SPECULA 1.1.0
cd examples/specula_ltao
```

SPECULA runs on the CPU by default. With CuPy installed, add `--gpu` to the `calibrate` and
`run` steps.

The whole tutorial in five commands:

```bash
python specula_ltao.py setup
python specula_ltao.py calibrate
python specula_ltao.py build --alpha 1e-4
python specula_ltao.py run --time 2
python specula_ltao.py analyse
```

Each step writes into `specula_ltao_run/`. The SPECULA system is described in
`params_ltao.yml`, and the interaction-matrix calibration in `calib_im.yml`.

## Step 1 — Describe the same system twice

SPECULA and pyTomoAO each need their own description of the system, and the two must agree.
`setup` writes three things:

**SPECULA subaperture data.** The KAPA valid-subaperture map, with subapertures enumerated
**row-major** (`np.nonzero` order). This choice matters later: SPECULA's slope vectors
follow this order.

**DM influence functions.** Gaussian influence functions with 15% inter-actuator coupling,
evaluated on SPECULA's pupil and saved as a SPECULA `IFunc`. The *same* Python function is
evaluated again on pyTomoAO's grid in Step 3, so the simulator and the reconstructor share
one DM model. For a real system, both would come from the DM's measured influence functions.

**A pyTomoAO configuration.** It starts from the bundled KAPA configuration and is changed
to match the simulation:

```python
cfg["atmosphere_parameters"].update(
    nLayer=7, zenithAngleInDeg=0.0,
    altitude=[0, 0.5, 1, 2, 4, 8, 16],               # km
    fractionnalR0=[0.517, 0.119, 0.063, 0.061, 0.105, 0.081, 0.054],
    r0=0.16, L0=25.0, wavelength=5e-7, ...)
cfg["lgs_asterism"].update(radiusAst=60.0, nLGS=6, baseLGSHeight=90e3)
cfg["tomography_parameters"] = {"fovOptimization": 0, "nFitSrc": 1}   # on axis
```

pyTomoAO places `nLGS` guide stars on a ring at azimuths `0, 360/nLGS, …`, measured from +x
towards +y. SPECULA's `polar_coordinates: [60, az]` uses the same convention, so the LGS in
`params_ltao.yml` are listed in the same order (`lgs1` at 0°, `lgs2` at 60°, …). If the two
orders differed, the slope blocks would have to be permuted to match.

## Step 2 — Measure the interaction matrix in SPECULA

```bash
python specula_ltao.py calibrate
```

`calib_im.yml` pushes and pulls each actuator by 50 nm and records the slopes of one LGS
wavefront sensor. A ground-conjugated DM looks the same to every LGS (no shift between
footprints and no cone compression at h = 0), so the single-sensor IM is tiled over the six
sensors.

This IM is used only by the **controller** (pseudo-open-loop reconstruction, Step 4). The
reconstructor itself is model based and never sees it. That makes it a useful independent
check: pyTomoAO's gradient operator applied to the same influence functions reproduces this
measured IM with correlation 0.998.

## Step 3 — Build the reconstructor and convert it to SPECULA units

```bash
python specula_ltao.py build --alpha 1e-4
```

The command matrix that SPECULA applies is a product of three pieces:

```python
from pyTomoAO import specula_slope_scale
from pyTomoAO.reconstructor import tomographicReconstructor

rec = tomographicReconstructor("pytomoao_config.yaml")
rec.build_reconstructor(alpha=1e-4)

# 1. SPECULA slopes -> pyTomoAO's slope convention
P = rec.slope_conversion_matrix(subapOrder="C", slopeScale=specula_slope_scale(4.0, 8))
# 2. slopes -> on-axis OPD on the reconstruction grid
R = rec.RecStatSA
R = R.get() if hasattr(R, "get") else R     # CuPy array when pyTomoAO runs on the GPU
# 3. OPD -> DM commands: least-squares fit of the influence functions on that grid
gx, gy = rec.grid_coordinates()
F = np.linalg.pinv(influence_functions(gx, gy), rcond=1e-6)

recmat = 1e9 * F @ R @ P      # SPECULA slopes -> commands [nm]
```

**The slope convention (`P`).** pyTomoAO's reconstructor expects, per sensor, all X slopes
then all Y slopes, valid subapertures enumerated column-major, in units of `d × angle of
arrival` (metres). Fed with that, `RecStatSA` returns OPD in metres. SPECULA's slopes differ
in two ways:

- **Order:** subapertures are row-major, hence `subapOrder="C"`.
- **Units:** slopes are normalised to the sensor's field of view. For the ideal sensor and
  the Shack–Hartmann slope computer,
  `specula_slope_scale(fov, pixelsPerSubaperture)` converts them to radians.

Both use X along the map columns and the same sign, so no axis flip is needed. `P` is a
sparse matrix with one non-zero per row (see {doc}`../user-guide/reconstruction`).

**The reconstruction grid.** `RecStatSA` reconstructs the phase on a grid twice as fine as
the lenslets, spanning the *padded* lenslet support `lgsWfsParams.DSupport`. For KAPA that
is 12 m, not the 10 m pupil. `rec.grid_coordinates()` returns the true positions of the
grid points. Fitting the influence functions on positions computed from `D` instead would
mis-scale the whole correction.

**Regularisation.** `alpha` scales a diagonal noise term relative to the mean slope variance.
The default, `alpha = 10`, assumes noisy data. These slopes are noiseless, and for them the
closed-loop residual is flat for `alpha ≤ 1e-3` and degrades quickly above 0.1. With real
noise, set the noise term from the measured slope noise instead, using
`build_reconstructor(noiseCovariance=...)`.

**Why not `assemble_reconstructor_and_fitting`?** That method builds its own influence
functions and assumes slopes in Shannon-sampled pixels, which suits the Keck real-time
controller. Here the influence functions must be exactly SPECULA's, and the slope units are
SPECULA's, so the three factors are assembled explicitly.

## Step 4 — Close the loop

```bash
python specula_ltao.py run --time 2        # add --gpu to run SPECULA on the GPU
```

The relevant part of `params_ltao.yml`:

```yaml
rec:
  class:             'ModalrecExplicitPolc'
  recmat_object:     'pytomoao_rec'
  intmat_object:     'kapa_dm349_6lgs_im'
  inputs:
    in_slopes_list:  ['wfs1.out_slopes', ..., 'wfs6.out_slopes']
    in_commands:     'control.out_comm:-1'

control:
  class:             'Integrator'
  int_gain:          [0.5]
  inputs:
    delta_comm:      'rec.out_modes'

dm:
  class:             'DM'
  ifunc_object:      'kapa_dm349'
  inputs:
    in_command:      'control.out_comm'
```

**Pseudo-open-loop control.** A tomographic MMSE reconstructor is built for *open-loop*
measurements, meaning the full turbulence. In closed loop, the sensors only see the residual,
so `ModalrecExplicitPolc` rebuilds the open-loop slopes as `s + IM · c(t−1)`, reconstructs
from those, and subtracts the commands already applied. The integrator then updates the
commands. Feeding closed-loop residual slopes directly into an MMSE tomographic reconstructor
would weight them with the wrong prior.

**Sign conventions.** The IM is measured with `sign: 1` (a positive command adds positive
OPD), while the DM in the loop uses SPECULA's default `sign: -1`, which subtracts the
commanded shape. Together, a positive command is a correction of the estimated turbulence,
which is what `recmat` produces.

**Timing.** The DM output enters the propagation with a one-frame lag
(`'dm.out_layer:-1'`), and the integrator adds no further delay. Science phase is stored
every fifth step.

## Step 5 — Look at the result

```bash
python specula_ltao.py analyse
```

It prints the residual and Strehl, and writes `closed_loop.png` to the work directory: the
residual against time, a residual map, the residual rms map and the long-exposure PSF. The
numbers also go to `closed_loop.json`.

The residual is evaluated with piston and tip-tilt removed, assuming an NGS controls
tip-tilt, and only over the part of the pupil the WFSs sense. KAPA's lenslet map has a
central gap (12 missing subapertures) that no reconstructor can correct. With a full-disc
SPECULA pupil, that unsensed region would otherwise dominate the residual.

Over 2 s at 500 Hz (input 798 nm rms, tip-tilt removed), the closed-loop residual and
long-exposure Strehl are:

| Reconstructor profile | Residual | Strehl (1.65 µm) |
|---|---|---|
| Simulated profile (`build --profile true`) | 144 nm rms | 0.74 |
| Generic KAPA config profile (`build --profile kapa`) | 145 nm rms | 0.74 |

For context:

- **Fitting:** the DM fitting error alone is about 98 nm for this 0.5 m actuator pitch.
- **Tomography and lag:** tomography adds about 100 nm and the one-frame lag about 27 nm,
  in quadrature.
- **Profile sensitivity:** with noiseless slopes, the result is insensitive to the exact
  profile shape. A wrong *outer scale* is a different matter: building with L0 = 100 m costs
  about 0.08 in Strehl. {doc}`../user-guide/slodar` measures the profile and L0 from the
  same wavefront sensors.

## Going further

- **Noise.** Swap `IdealDerivativeSensor` for SPECULA's Shack–Hartmann and CCD, then set the
  reconstructor's noise term from the measured slope noise with
  `build_reconstructor(noiseCovariance=...)` rather than `alpha`. At low flux, also
  calibrate the centroid gain: thresholded centroids lose gain, and the IM and the
  reconstructor then disagree with the sensor.
- **Profile from telemetry.** Record `rec.out_pseudo_ol_modes`, or rebuild pseudo-open-loop
  slopes with `pyTomoAO.slodar.pseudo_open_loop_slopes`. Run SLODAR on them and rebuild the
  reconstructor from the measured profile.
- **Gain.** At these noise levels, 0.5–0.7 is about optimal at 500 Hz. Lower it as noise
  increases.
