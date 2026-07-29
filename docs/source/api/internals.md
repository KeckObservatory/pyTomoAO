# Internal kernels

:::{warning}
These functions are private. They are documented for people working on pyTomoAO itself;
their names and signatures may change in any release. Use
{py:class}`~pyTomoAO.reconstructor.tomographicReconstructor` in application code.
:::

The two modules below are interchangeable implementations of the same set of kernels. The
CPU version uses NumPy with Numba-compiled inner loops; the GPU version uses CuPy.
{py:mod}`pyTomoAO.reconstructor` imports one or the other at module load, based
on whether CuPy is importable.

| Function                                | Role                                                       |
| --------------------------------------- | ----------------------------------------------------------- |
| `_covariance_matrix`                    | Von Kármán phase covariance between two point sets           |
| `_auto_correlation`                     | Slope-to-slope covariance across all guide star pairs        |
| `_cross_correlation`                    | Optimisation-direction-to-slope covariance                   |
| `_sparseGradientMatrixAmplitudeWeighted`| Phase-to-slope gradient operator for the lenslet array       |
| `_build_reconstructor_model`            | Assembles the model-based MMSE reconstructor                 |
| `_build_reconstructor_im`               | Assembles the interaction-matrix-based reconstructor         |

The GPU variants additionally accept a `use_float32` flag controlling the working precision
on the device.

## CPU implementation

```{eval-rst}
.. currentmodule:: pyTomoAO.tomographyUtilsCPU

.. autofunction:: _covariance_matrix
.. autofunction:: _auto_correlation
.. autofunction:: _cross_correlation
.. autofunction:: _sparseGradientMatrixAmplitudeWeighted
.. autofunction:: _build_reconstructor_model
.. autofunction:: _build_reconstructor_im
```

## GPU implementation

```{eval-rst}
.. currentmodule:: pyTomoAO.tomographyUtilsGPU

.. autofunction:: _covariance_matrix
.. autofunction:: _auto_correlation
.. autofunction:: _cross_correlation
.. autofunction:: _sparseGradientMatrixAmplitudeWeighted
.. autofunction:: _build_reconstructor_model
.. autofunction:: _build_reconstructor_im
```
