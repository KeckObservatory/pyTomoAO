# GPU acceleration

Building an MMSE reconstructor is dominated by two costs: forming the covariance matrices,
and inverting them. Both are well suited to a GPU, and pyTomoAO ships a CuPy implementation
of the covariance kernels alongside the Numba/NumPy CPU implementation.

## How the backend is selected

The choice happens once, at import time of
{py:mod}`pyTomoAO.tomographicReconstructor`:

```python
try:
    import cupy as cp
    from pyTomoAO.tomographyUtilsGPU import ...
except Exception:
    from pyTomoAO.tomographyUtilsCPU import ...
```

If CuPy imports successfully the GPU kernels are used; otherwise the CPU kernels are. You
will see one of these lines in the log:

```text
CUDA is available. Using GPU for computations.
CUDA is not available. Using CPU for computations.
```

The two backends expose the same functions — `_auto_correlation`, `_cross_correlation`,
`_build_reconstructor_model`, `_build_reconstructor_im` and
`_sparseGradientMatrixAmplitudeWeighted` — so nothing in your code changes.

## Installing CuPy

Install the wheel matching your CUDA toolkit:

```bash
pip install cupy-cuda12x     # CUDA 12.x
pip install cupy-cuda11x     # CUDA 11.x
```

Verify it works before expecting pyTomoAO to use it:

```bash
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

## Forcing the CPU

Pass `force_cpu=True` when constructing the reconstructor — useful for reproducing a result
exactly, for benchmarking, or when the GPU is busy:

```python
reconstructor = tomographicReconstructor("config.yaml", force_cpu=True)
```

:::{warning}
`force_cpu` sets a module-level flag, so it affects every reconstructor created afterwards
in the same process, not just this one. Keep CPU-forced work in its own process if you also
need GPU results.
:::

## Precision

`build_reconstructor(use_float32=True)` requests single precision. Two details are worth
knowing:

- On the **CPU** path the flag is honoured: `False` gives float64.
- On the **GPU** path the kernels are currently always invoked with `use_float32=True`,
  regardless of the argument, so GPU results are single precision.

Single precision is usually adequate — the reconstructor is a regularised inverse and the
regularisation dominates the conditioning — but if you are comparing CPU and GPU results
digit by digit, expect differences at the float32 level.

## Benchmarking

The repository includes scripts for measuring both paths:

```bash
python examples/benchmark/compare_cpu_gpu.py
python examples/benchmark/tomographicReconstructorBenchmarking.py
```

Speedup depends strongly on problem size. Small systems (a single WFS, a few hundred
subapertures) can be dominated by host-to-device transfer, while large multi-LGS
configurations are where the GPU pays off.

## Memory

The covariance matrices scale as the square of the number of valid phase points times the
number of guide stars. If you hit an out-of-memory error, in order of effectiveness:

1. Use `use_float32=True` on the CPU path, or accept the GPU's single precision.
2. Reduce `nFitSrc` — the optimisation grid multiplies the cross-covariance size.
3. Reduce the pupil sampling by using a coarser `validLLMap`.
4. Fall back to `force_cpu=True`, where you have far more RAM to work with.
