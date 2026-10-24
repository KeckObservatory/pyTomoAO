# Turbulence profiling (SLODAR)

{py:class}`~pyTomoAO.slodar.slodarProfiler` estimates the vertical turbulence profile
$C_n^2(h)$ from open-loop (or pseudo-open-loop) slopes recorded simultaneously by the
LGS wavefront sensors. The result is expressed in the same layered von Kármán model the
reconstructor uses and can be written straight back into an `atmosphere_parameters`
section.

```python
from pyTomoAO import example_config
from pyTomoAO.slodar import slodarProfiler

prof = slodarProfiler.from_config(example_config("kapa"))   # geometry from the YAML file
prof.compute_covariance_maps(slopes, removeTT=True)          # slopes: (nFrames, nLGS*2*nValid) [rad]
result = prof.fit_profile()
print(result)
atm = result.to_atmosphere_config(minFraction=0.01)          # drop-in atmosphere_parameters
```

## Method

For every pair of sensors $(i, j)$ and slope components $a, b \in \{x, y\}$ the module
computes the covariance map

$$
C_{ij}^{ab}(\boldsymbol\delta) = \frac{1}{N_{\boldsymbol\delta}}
\sum_{\mathbf p} \left\langle s_i^a(\mathbf p)\, s_j^b(\mathbf p+\boldsymbol\delta)\right\rangle_t ,
$$

normalised by the number $N_{\boldsymbol\delta}$ of valid subaperture pairs at offset
$\boldsymbol\delta$ (FFT-based, all offsets at once). A layer at range $h$ produces a
cross-covariance peak at offset $\boldsymbol\delta = -(\boldsymbol\theta_j-\boldsymbol\theta_i)h / (d(1-h/H))$,
so the native altitude bins of a baseline $\theta$ are

$$
h_k = \frac{k\,d'}{\theta + k\,d'/H},
$$

with $d'$ the subaperture lattice step along the baseline ($d$ along X or Y, $\sqrt2 d$ on
a diagonal) and $H$ the LGS range ($H=\infty$ recovers the classical $h_k = k d/\theta$).

The profile is the non-negative least-squares fit of thin-layer **response maps** to the
measured maps. A response map is the slope covariance a unit-strength layer at range $h$
would produce, computed from pyTomoAO's own von Kármán phase covariance and the 3×3
Fried-geometry stencil of the tomographic gradient operator, including:

- the LGS cone effect (sub-aperture footprints shrink and offsets compress by $1-h/H$);
- the finite outer scale $L_0$ (fixed, or estimated by a grid search with `fitL0=True`);
- per-frame tip-tilt removal (`removeTT=True`, needed for LGS), applied identically to data
  and model;
- an optional per-WFS white-noise term on the auto-covariance (`fitNoise=True`);
- generalised SLODAR (`conjugationAltitude < 0`), which moves the ground layer off zero
  offset.

All WFS pairs are fitted jointly by default, so the widest baselines set the low-altitude
resolution and the narrower ones extend the profile upwards
({py:meth}`~pyTomoAO.slodar.slodarProfiler.default_altitudes`).

## Conventions

Slopes are angles of arrival in radians (`slopeScale` converts other units). Per sensor
they are ordered as elsewhere in pyTomoAO (`slopesOrder="simu"`, `"inverted"` or
`"keck"`), with valid subapertures enumerated column-major (`subapOrder="F"`, the order of
the gradient operator) or row-major (`subapOrder="C"`). X runs along the columns of
`validLLMap`. Guide-star directions are Cartesian `[x, y]` offsets in radians.

## Closed-loop telemetry and noise

On sky the loop is closed, so SLODAR runs on pseudo-open-loop slopes
$s_{\rm pol}(t) = s(t) + M_{\rm int}\,c(t-1)$
({py:func}`~pyTomoAO.slodar.pseudo_open_loop_slopes`).

Two settings need checking on the telemetry:

- **Delay.** The number of frames between computing a command and the slopes it affects.
- **Sign.** `sign=+1` when the loop applies the opposite of the recorded command
  ($s = s_{\rm ol} - M c$, e.g. SPECULA); `sign=-1` when it applies the recorded command and
  $M$ was poked with it ($s = s_{\rm ol} + M c$). In a well-corrected loop the wrong sign
  barely changes the slope variance, but its frame-to-frame increments are 2–4× larger.

WFS noise is white, so it only biases the zero-offset auto-covariance, and `fitNoise=True`
absorbs it into per-WFS noise variances. That keeps the *profile* unbiased. For the
*reconstructor's* noise covariance, though, use the temporal estimate:

```python
from pyTomoAO.slodar import pseudo_open_loop_slopes, slope_noise_covariance

pol = pseudo_open_loop_slopes(slopes, commands, IM)        # same units as the slopes
res = prof.profile(pol, removeTT=True, fitL0=True)         # profile, for the turbulence model
var = prof.temporal_noise_variance(pol, removeTT=True)     # per-WFS noise [rad^2]: C(0) - C(1)
rec.build_reconstructor(noiseCovariance=slope_noise_covariance(var, reconstructor=rec))
```

{py:meth}`~pyTomoAO.slodar.slodarProfiler.temporal_noise_variance` subtracts the slope
covariance at a one-frame lag from the variance. White noise is uncorrelated from frame to
frame, while the turbulence barely changes.
{py:func}`~pyTomoAO.slodar.slope_noise_covariance` converts the angle-of-arrival variances
to the reconstructor's units: rad² of phase difference across a subaperture at the
atmosphere wavelength. It takes `d`, `nValidSubap` and the wavelength from the reconstructor.

:::{caution}
Two effects found in SPECULA closed-loop tests with a physical Shack–Hartmann (photon and
read noise):

- **The zero-offset noise fit is unreliable as a noise estimate.** It also absorbs the
  mismatch between the real sensor and the geometric slope model at zero offset, and
  overestimated the noise by up to 5× (`res.noise_covariance()` inherits this). The temporal
  estimate was within 15%.
- **Centroid gain at low flux.** A thresholded centre of gravity loses gain at low SNR (0.92
  at 200 e⁻/subaperture/frame, 0.43 at 50 e⁻). Uncorrected, this biases the SLODAR r0 by up
  to +56%, even from cross-covariances alone, and mismatches the interaction matrix.
  Calibrate the centroid gain (e.g. dithering or modulation on sky) and rescale the slopes
  before profiling.
:::

## Record length and uncertainty

With frozen flow, a record of length $T$ samples only a strip of each layer about
$v\,T$ long. The covariance maps of a short record therefore carry the shape of one
turbulence realisation, not the ensemble average the response maps describe. The fit
absorbs that mismatch partly in the wrong altitude bins. This is sampling physics, not an
estimator bias: on SPECULA simulations, averaged over independent atmosphere seeds, SLODAR
is unbiased.

Typical realisation scatter (1σ), from SPECULA and an independent frozen-flow simulation,
with 6 LGS on a 60″ ring, 20×20 subapertures and a 5 m/s ground layer:

| Case | Record | Scatter |
|---|---|---|
| Strong high layer (40% at 15 km) | 4 s | ±1.6% in altitude (±0.24 km) |
| Weak high layer (5% at 15 km) | 8 s | ±2 km |
| Total r0 | 4–8 s | biased high by 5–7% |

The r0 bias comes from subtracting the record's temporal mean, which removes static
large-scale power.

Consequences:

- **Use long records.** Precision improves with the number of independent realisations,
  roughly $T\,v_{\rm ground}/D$. Weak high layers and the outer scale need the most data;
  20–30 s or more, or several records averaged.
- **Quote the scatter between independent records**, or blocks at least $D/v_{\rm ground}$
  long, not the scatter between 1 s chunks of one record. In the tests above, chunk scatter
  understated the true uncertainty by about 2×.
- **Validate on several realisations.** In simulation, use 8 or more atmosphere seeds;
  conclusions drawn from one seed, even across many configurations, describe that one
  realisation.

## Validation

`examples/slodar/` contains the validation scripts: synthetic slopes from the model and
from an independent FFT phase-screen simulator (`validate_synthetic.py`), and an
end-to-end LTAO simulation with [SPECULA](https://github.com/ArcetriAdaptiveOptics/SPECULA)
(`specula/`).
