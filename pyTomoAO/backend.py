"""Selection of the CPU or GPU covariance kernels.

The two kernel modules, :mod:`pyTomoAO.tomographyUtilsCPU` and
:mod:`pyTomoAO.tomographyUtilsGPU`, implement the same five private functions. This module
picks one and hands back a namespace holding them, so that the choice is made per
reconstructor instead of being baked into module-level names at import time.
"""

import importlib
import logging
from types import SimpleNamespace

logger = logging.getLogger(__name__)

#: The kernels every backend must provide.
KERNEL_NAMES = (
    "_auto_correlation",
    "_build_reconstructor_im",
    "_build_reconstructor_model",
    "_cross_correlation",
    "_sparseGradientMatrixAmplitudeWeighted",
)

_MODULES = {
    "cpu": "pyTomoAO.tomographyUtilsCPU",
    "gpu": "pyTomoAO.tomographyUtilsGPU",
}

# Populated on first use by _probe_gpu().
_gpu_available = None
_gpu_error = None


def _probe_gpu():
    """Try to import the GPU kernels once, remembering why it failed if it did."""
    global _gpu_available, _gpu_error
    if _gpu_available is None:
        try:
            importlib.import_module(_MODULES["gpu"])
        except Exception as exc:
            _gpu_available = False
            _gpu_error = exc
        else:
            _gpu_available = True
            _gpu_error = None
    return _gpu_available


def cuda_available():
    """Whether the GPU kernels can be imported.

    Returns
    -------
    bool
    """
    return _probe_gpu()


def gpu_import_error():
    """The exception raised when importing the GPU kernels, if any.

    Returns
    -------
    Exception or None
    """
    _probe_gpu()
    return _gpu_error


def _load(kind):
    module = importlib.import_module(_MODULES[kind])
    kernels = {name: getattr(module, name) for name in KERNEL_NAMES}
    return SimpleNamespace(name=kind, is_gpu=kind == "gpu", module=module, **kernels)


def get_backend(prefer="auto"):
    """Resolve the kernel backend for one reconstructor.

    Parameters
    ----------
    prefer : {"auto", "cpu", "gpu"}, optional
        ``"auto"`` uses the GPU when CuPy imports and the CPU otherwise. ``"cpu"`` always
        uses the CPU kernels. ``"gpu"`` raises if the GPU kernels are unavailable, rather
        than silently falling back -- a caller who asked for the GPU wants to know.

    Returns
    -------
    types.SimpleNamespace
        Namespace with ``name``, ``is_gpu``, ``module`` and the entries of
        :data:`KERNEL_NAMES`.

    Raises
    ------
    ValueError
        If ``prefer`` is not one of the three accepted values.
    RuntimeError
        If ``prefer="gpu"`` and the GPU kernels cannot be imported.
    """
    if prefer not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"prefer must be 'auto', 'cpu' or 'gpu', got {prefer!r}")

    if prefer == "cpu":
        return _load("cpu")

    if prefer == "gpu":
        if not _probe_gpu():
            raise RuntimeError(
                "The GPU backend was requested but the CuPy kernels could not be imported: "
                f"{type(_gpu_error).__name__}: {_gpu_error}"
            )
        return _load("gpu")

    if _probe_gpu():
        return _load("gpu")

    # CuPy simply not being installed is the ordinary case. Anything else means the user
    # set up GPU support and is not getting it, at roughly 35x the cost on a reconstructor
    # build, so it is worth a warning carrying the underlying error.
    if isinstance(_gpu_error, ModuleNotFoundError):
        logger.info(
            "CuPy is not installed; using the CPU backend. "
            "Install pyTomoAO[gpu] for GPU acceleration."
        )
    else:
        logger.warning(
            "CuPy is installed but the GPU backend could not be loaded, so pyTomoAO is "
            "falling back to the CPU backend: %s: %s",
            type(_gpu_error).__name__,
            _gpu_error,
        )
    return _load("cpu")
