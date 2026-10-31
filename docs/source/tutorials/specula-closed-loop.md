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

Steps 1–7 use SPECULA's ideal derivative sensor, which returns **noiseless** geometric slopes:
the cleanest setting for checking that the two codes agree. Step 8 switches to physical
Shack–Hartmann sensors with photon and read noise, and sets the reconstructor's noise term
from the noise measured on the telemetry.

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
`build_reconstructor(noiseCovariance=...)` (Step 8).

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

## Step 6 — Measure the profile from the closed loop

On sky the true profile is unknown. The reconstructor has to be built on a profile measured
from the same wavefront sensors, while the loop runs. The example reproduces that cycle:

```bash
python specula_ltao.py run --time 20 --record     # closed loop + slopes and commands
python specula_ltao.py slodar                     # SLODAR on pseudo-open-loop slopes
python specula_ltao.py build --profile slodar     # reconstructor on the measured profile
python specula_ltao.py run --time 2
python specula_ltao.py analyse
```

`run --record` saves the six WFSs' slopes and the DM commands at every loop step. `slodar`
then does three things:

1. **Rebuilds the open-loop slopes:**

   ```python
   pol = pseudo_open_loop_slopes(slopes, commands, IM, delay=1, sign=1)
   ```

   SPECULA's DM applies the *opposite* of the command, while the IM was poked with `+c`, so
   `s = s_ol − M c(t−1)` and `sign=+1`. The step prints a check: the frame-to-frame increments
   of `pol` are about 2.3× smaller with the right sign than with the wrong one.
2. **Fits the profile** with `slodarProfiler` (L0 fitted, or fixed with `--L0`) and estimates
   the slope noise with `temporal_noise_variance`.
3. **Writes** `slodar_profile.png` and a pyTomoAO configuration on the measured profile.

With 20 s of telemetry (10 000 frames at 500 Hz):

| Reconstructor profile | Residual | Strehl (1.65 µm) |
|---|---|---|
| Simulated (true) profile | 144 nm rms | 0.74 |
| SLODAR from closed-loop telemetry, L0 fitted (19 m) | 147 nm rms | 0.73 |
| SLODAR from closed-loop telemetry, L0 fixed at 25 m | 146 nm rms | 0.74 |

**How much telemetry.** With only 6 s, the same pipeline gave r0 17% high and L0 = 52 m,
and it placed the 16 km layer at 13.8 km. The reconstructor built on it reached only 165 nm.
Fitting the two halves of the 6 s separately showed this is sample variance: the halves
disagreed with each other by more than with the truth. Two parts of the profile need the
most data:

- **The outer scale.** From a few seconds, its fit scatters by a factor of two.
- **The highest layer.** It is seen only on the longest baselines, with the fewest
  overlapping subapertures, and its altitude matters most for on-axis tomography.

Record at least 20–30 s. Fitting L0 is generally better than fixing it, even at its true
value: on a finite record, the fitted L0 absorbs part of that realisation's large-scale
power, while a fixed one pushes the mismatch into r0 and the profile (see Step 7). If the
fitted L0 lands on an edge of the search grid, `slodar` warns (`slodarResult.L0AtGridEdge`).
L0 is then not constrained by the data, so record longer before trusting the profile.

The evaluation run reuses the recording's atmosphere seed. For a stricter test, add
`--seed 312` to the second `run`. All the numbers above come from a single atmosphere seed.
SLODAR's realisation scatter (see {doc}`../user-guide/slodar`, *Record length and
uncertainty*) is large enough that conclusions about the measured profile should be checked
over several seeds (`run --seed N`).

## Step 7 — Another atmosphere

`setup --atmosphere FILE` simulates a different turbulence profile. Later steps read it from
the work directory, so the rest of the workflow is unchanged. `atmosphere_5layer.yml` is a
second test case:

- **Layers:** 0, 1.5, 5, 10 and 15 km, with 70% of the turbulence at the ground.
- **Conditions:** seeing 0.75″ (r0 = 0.134 m) and L0 = 30 m.
- **Winds:** different speeds and directions from the default case.

```bash
python specula_ltao.py setup --workdir atm5 --atmosphere atmosphere_5layer.yml
python specula_ltao.py calibrate --workdir atm5
python specula_ltao.py build --workdir atm5
python specula_ltao.py run --workdir atm5 --time 2
python specula_ltao.py analyse --workdir atm5
```

The file format is self-explanatory: altitudes, Cn² fractions, seeing, L0 and per-layer
winds.

We repeated the cycle from Step 6 over five atmosphere seeds (312–316):

1. Record 20 s of closed-loop telemetry with the true-profile reconstructor.
2. Run SLODAR on it and rebuild the reconstructor on the measured profile.
3. Score each reconstructor on a 2 s closed loop with an *unseen* atmosphere seed.

| Reconstructor profile | Residual, mean (range) | Strehl (1.65 µm) |
|---|---|---|
| Simulated (true) profile | 174 nm (170–179) | 0.65 |
| SLODAR from closed-loop telemetry, L0 fitted | 184 nm (170–209) | 0.62 |
| Generic KAPA config profile (`--profile kapa`) | 246 nm (240–250) | 0.42 |

- **The generic profile:** unlike the default atmosphere, where it was nearly as good as the
  true one, it fits this atmosphere badly. This is the case where measuring the profile pays
  off.
- **SLODAR:** it recovers about 85% of that loss. On four seeds it is within 0–9 nm of the
  true profile. On the fifth, the L0 fit ran to the 200 m edge of its grid (input 30 m) and
  cost 33 nm; that is the case the edge warning flags.
- **Fixing L0 at its true value:** this made things worse on the two seeds tried (+10 and
  +15 nm), for the reason given in Step 6.

These are 20 s records. Expect the high-layer altitudes and L0 to scatter from one record to
the next by the amounts given in {doc}`../user-guide/slodar` (*Record length and
uncertainty*).

## Step 8 — Add wavefront-sensor noise

So far the slopes were noiseless, so the reconstructor only needed a token regularisation
(`alpha = 1e-4`). Real LGS wavefront sensors are noisy, and the noise has to enter the
reconstructor's model. `setup --wfs sh` replaces the six ideal sensors with a physical
Shack–Hartmann chain per LGS (`wfs_sh.yml`):

- **`SH`:** 20×20 lenslets, 8×8 pixels per subaperture at 0.486″/pixel (3.9″ field), and a
  1″ Gaussian LGS spot (no elongation).
- **`CCD`:** photon noise and 3 e⁻ rms read noise. `--photons` sets the LGS flux in detected
  photons per subaperture and frame (the LGS magnitude is computed from it); `--photons 0`
  makes the detector noiseless.
- **`ShSlopec`:** centre of gravity above a threshold at 20% of the peak.

`run` adds `wfs_sh.yml` on top of `params_ltao.yml`. The slope computers keep the names
`wfs1`–`wfs6`, so the controller and the telemetry are wired exactly as before. The detector
noise seeds follow `--seed`.

### The noiseless case: the Shack–Hartmann itself

```bash
python specula_ltao.py setup --workdir sh0 --wfs sh --photons 0
python specula_ltao.py calibrate --workdir sh0
python specula_ltao.py build --workdir sh0 --alpha 1e-4
python specula_ltao.py run --workdir sh0 --time 2
python specula_ltao.py analyse --workdir sh0
```

**The slope unit.** The centroids of a physical Shack–Hartmann are not in the ideal sensor's
unit. `calibrate` therefore measures two interaction matrices on the same DM pokes, one with
the ideal sensor and one with the Shack–Hartmann (noiseless detector, as a calibration would
be on sky), and prints their ratio:

```text
calibrate: Shack-Hartmann IM = 1.397 x ideal-sensor IM (correlation 0.997)
```

The ratio is stored in `wfs.json`, and `build` and `slodar` divide
`specula_slope_scale(4.0, 8)` by it. The 0.997 correlation also confirms that the
subaperture order and the slope signs are the same as the ideal sensor's, so the conversion
matrix `P` of Step 3 is otherwise unchanged. The loop uses the Shack–Hartmann IM.

Without noise, the physical sensor costs about 10 nm over the ideal one (155 nm against
144 nm on seed 311): the Shack–Hartmann measures the average tilt of the diffraction-limited
spot, not the exact phase gradient the reconstructor's model assumes.

### The noisy case

```bash
python specula_ltao.py setup --workdir sh200 --wfs sh --photons 200
python specula_ltao.py calibrate --workdir sh200
python specula_ltao.py build --workdir sh200 --alpha 1e-4
python specula_ltao.py run --workdir sh200 --time 20 --record   # a noisy loop, recorded
python specula_ltao.py slodar --workdir sh200                   # profile + slope noise
python specula_ltao.py build --workdir sh200 --noise measured   # noise-aware reconstructor
python specula_ltao.py run --workdir sh200 --time 2
python specula_ltao.py analyse --workdir sh200
```

At 200 photons per subaperture and frame, the slope noise is about 0.057″ rms per axis,
against 0.16″ rms of turbulent slope in open loop.

**Measuring the noise.** `slodar` reports it from the same pseudo-open-loop telemetry as the
profile, with `slodarProfiler.temporal_noise_variance`. White noise decorrelates from one
frame to the next, while the turbulence barely changes in 2 ms, so the noise variance is the
slope variance minus the one-frame covariance, `C(0) − C(1)`:

```text
slodar: r0 0.1704 m (input 0.1601), L0 19.1 m (input 25), profile TVD 0.162,
        temporal noise 0.0574 arcsec rms
```

In the SPECULA noise study, this estimator was within 15% of the true noise (measured
directly with a noiseless twin of each detector). SLODAR's own noise term, the zero-offset
excess of the auto-covariance, is not suitable here: it also absorbs the Shack–Hartmann's
departure from the geometric slope model, and overestimated the noise by up to 5×.

**Using it.** `build --noise measured` converts the per-WFS variance to the units of the
reconstructor's slope covariance and passes it in place of `alpha`:

```python
from pyTomoAO.slodar import slope_noise_covariance

var = prof.temporal_noise_variance(pol, removeTT=True)    # rad^2 of angle, per WFS
Cn = slope_noise_covariance(var, reconstructor=rec)       # (2 pi d / lambda)^2 * var
rec.build_reconstructor(noiseCovariance=Cn)
```

`--profile slodar` can be combined with it, for the fully measured case.

### Results

Mean residual over three atmosphere seeds (311–313, 2 s each, gain 0.5), with the range in
brackets:

| WFS | Reconstructor | Residual | Strehl (1.65 µm) |
|---|---|---|---|
| Shack–Hartmann, noiseless | true profile, `alpha = 1e-4` | 160 nm (155–163) | 0.70 |
| 200 photons | true profile, `alpha = 1e-4` (tuned for no noise) | 207 nm (203–210) | 0.54 |
| 200 photons | true profile, measured noise (`--noise measured`) | 176 nm (171–180) | 0.64 |
| 200 photons | SLODAR profile, measured noise (`--profile slodar --noise measured`) | 181 nm (176–184) | 0.63 |
| 200 photons | true profile, `alpha = 0.05` (tuned by hand) | 174 nm (169–178) | 0.65 |

The noise and the SLODAR profile were measured once, from the 20 s recording on seed 311, and
reused on the other two seeds.

- **Noise matters once it is in the model.** With the noiseless-tuned `alpha = 1e-4`, the
  reconstructor treats the noise as signal and the residual grows by about 45 nm.
  With the measured noise in the model, the reconstructor recovers about two thirds of
  that loss, with no tuning: the remaining 16 nm is the noise itself.
- **The profile matters little at this flux.** The SLODAR profile measured from the noisy telemetry costs
  about 5 nm over the true one, as in the noiseless case. The noise does not hurt SLODAR,
  because it is white and only adds to the zero-lag auto-covariance.
- **The measured noise is not quite the optimal regularisation.** In closed loop the
  integrator already averages the noise, so the best noise term is somewhat lighter than the
  measured variance, typically 0.2–0.6× in the SPECULA noise study. The hand-tuned
  `alpha = 0.05` in the table is about 0.4× the measured noise, and gains 2 nm. The loop gain can be
  retuned too: at 200 photons its optimum is about 0.4, against 0.5–0.7 without noise.

**Caveats.**

- **Centroid gain.** A thresholded centre of gravity loses gain at low flux (0.92 at
  200 photons, 0.43 at 50). The interaction matrix is calibrated with a noiseless detector, so
  the loop then sees a slightly different sensor from the one it was calibrated on. At
  200 photons this costs under 1 nm. At 50 photons it costs 40–50 nm and biases SLODAR's r0,
  and the centroid gain must be calibrated (on sky, by dithering).
- **LGS elongation.** The spot here is round. With elongation, the noise and the centroid
  gain differ between subapertures and between axes. Fit the profile on the
  cross-covariances only in that case.
- **Run time.** A physical Shack–Hartmann is much slower than the ideal sensor: about 0.4 s
  per loop step for the six sensors on two CPU cores, so the 20 s recording takes over an
  hour. Use `--gpu` if you can.

## Going further

- **Other fluxes.** `setup --wfs sh --photons N` sets the flux. Below about 100 photons the
  centroid gain and the loop gain need attention (see the caveats above).
- **Gain.** Without noise, 0.5–0.7 is about optimal at 500 Hz. Lower it as the noise
  increases.
