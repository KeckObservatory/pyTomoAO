# Examples

## `specula_ltao/`

Closes an LTAO loop in the SPECULA simulator with pyTomoAO's MMSE reconstructor:
- **Geometry:** 6 LGS on a 60″ ring, 20×20 Shack–Hartmann on the KAPA map, and a 349-actuator
  ground DM.
- **Atmosphere:** the 7-layer Maunakea median profile.
- **Control:** pseudo-open-loop control at 500 Hz.

Walkthrough in the [SPECULA closed-loop tutorial](https://keckobservatory.github.io/pyTomoAO/tutorials/specula-closed-loop.html).
Needs `pip install specula`. Add `--gpu` to `calibrate` and `run` if CuPy is installed.

```sh
cd examples/specula_ltao
python specula_ltao.py setup
python specula_ltao.py calibrate
python specula_ltao.py build --alpha 1e-4
python specula_ltao.py run --time 2
python specula_ltao.py analyse
```

## `slodar/`

Validation of `pyTomoAO.slodar`:
- **`validate_synthetic.py`:** synthetic slopes, from Gaussian draws and from FFT phase
  screens.
- **`specula/`:** an end-to-end SPECULA open-loop run, using `run_specula.py` then
  `analyze_specula.py`.

See the [SLODAR user guide](https://keckobservatory.github.io/pyTomoAO/user-guide/slodar.html).

## `benchmark/benchmark.py`

Times `build_reconstructor` on every configuration bundled with the package, on whichever
backends are available, and checks that the CPU and GPU results agree.

```sh
python examples/benchmark/benchmark.py
```

```
configuration          backend     min (s)   median (s)          shape
----------------------------------------------------------------------
kapa                   cpu           2.450        2.597      1312x2432
kapa                   gpu           0.081        0.083      1312x2432
                       agree      7.62e-05 (relative)
```

The CPU and GPU numbers are not expected to match exactly: the GPU kernels work in single
precision, so agreement at the 1e-4 level is the float32 floor rather than a defect.

### Baselines

`baseline.json` records a set of timings together with the processor they were measured on.
Compare a change against it with:

```sh
python examples/benchmark/benchmark.py --check-baseline --tolerance 0.25
```

which exits non-zero if any configuration is more than 25% slower. Record a new reference
with `--save-baseline`. Timings are machine-specific, so the script warns when the recorded
processor differs from the current one — treat a baseline from another machine as
indicative only.

### Why this replaced the previous scripts

`examples/benchmark/` used to hold `test_auto.py`, `test_auto_gpu.py` and
`compare_cpu_gpu.py` — around 1700 lines that carried their own **forked copies** of the
covariance kernels rather than calling the package. They drifted, and by the end they
contained none of the corrections made to the real kernels, so the numbers they reported
described code that was no longer shipped. Everything here goes through the public API.
