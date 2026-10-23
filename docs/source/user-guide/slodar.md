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
({py:func}`~pyTomoAO.slodar.pseudo_open_loop_slopes`; check the delay on the telemetry).
WFS noise is white and only biases the zero-offset auto-covariance, which `fitNoise=True`
absorbs into per-WFS noise variances. Those variances can regularise the reconstructor
directly, giving the MMSE estimator for the measured noise:

```python
pol = pseudo_open_loop_slopes(slopes, commands, IM)          # same units as the slopes
res = prof.profile(pol, removeTT=True, fitL0=True)
rec.build_reconstructor(noiseCovariance=res.noise_covariance(d, rec.lgsWfsParams.nValidSubap))
```

`noise_covariance` converts the angle-of-arrival variances [rad²] to the units of the
reconstructor's slope covariance, rad² of phase difference across a subaperture at the
atmosphere wavelength.

## Validation

`examples/slodar/` contains the validation scripts: synthetic slopes from the model and
from an independent FFT phase-screen simulator (`validate_synthetic.py`), and an
end-to-end LTAO simulation with [SPECULA](https://github.com/ArcetriAdaptiveOptics/SPECULA)
(`specula/`).
