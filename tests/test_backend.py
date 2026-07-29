"""Tests for kernel backend selection (#112).

`force_cpu=True` used to log that it was forcing the CPU and then run on the GPU anyway: it
flipped a module-level flag, but the GPU functions were already bound at import. It also
changed the backend for every other reconstructor in the process.
"""

import logging

import pytest

from pyTomoAO import backend
from pyTomoAO import reconstructor as reconstructor_module

logger = logging.getLogger(__name__)


class TestGetBackend:
    def test_cpu_is_always_available(self):
        cpu = backend.get_backend("cpu")
        assert cpu.name == "cpu"
        assert cpu.is_gpu is False
        for name in backend.KERNEL_NAMES:
            assert callable(getattr(cpu, name)), f"{name} missing from the CPU backend"

    def test_auto_matches_availability(self):
        assert backend.get_backend("auto").name == ("gpu" if backend.cuda_available() else "cpu")

    def test_rejects_an_unknown_preference(self):
        with pytest.raises(ValueError, match="prefer must be"):
            backend.get_backend("cuda")

    def test_explicit_gpu_request_does_not_silently_fall_back(self):
        """Asking for the GPU and getting the CPU without being told is how a 35x slowdown
        goes unnoticed."""
        if backend.cuda_available():
            assert backend.get_backend("gpu").name == "gpu"
        else:
            with pytest.raises(RuntimeError, match="GPU backend was requested"):
                backend.get_backend("gpu")

    def test_both_backends_expose_the_same_kernels(self):
        cpu = backend.get_backend("cpu")
        if not backend.cuda_available():
            pytest.skip("requires CuPy and a CUDA device")
        gpu = backend.get_backend("gpu")
        for name in backend.KERNEL_NAMES:
            assert callable(getattr(gpu, name)), f"{name} missing from the GPU backend"
            assert callable(getattr(cpu, name))


class TestForceCPU:
    def test_force_cpu_selects_the_cpu_kernels(self, revolt_config):
        rec = reconstructor_module.tomographicReconstructor(revolt_config, force_cpu=True)
        assert rec.backend == "cpu"

    def test_default_follows_availability(self, revolt_config):
        rec = reconstructor_module.tomographicReconstructor(revolt_config)
        expected = "gpu" if backend.cuda_available() else "cpu"
        assert rec.backend == expected

    def test_force_cpu_does_not_leak_to_other_instances(self, revolt_config):
        """The old implementation mutated a process-global flag."""
        auto_before = reconstructor_module.tomographicReconstructor(revolt_config)
        forced = reconstructor_module.tomographicReconstructor(revolt_config, force_cpu=True)
        auto_after = reconstructor_module.tomographicReconstructor(revolt_config)

        expected = "gpu" if backend.cuda_available() else "cpu"
        assert forced.backend == "cpu"
        assert auto_before.backend == expected, "an existing reconstructor changed backend"
        assert auto_after.backend == expected, "a later reconstructor inherited force_cpu"
        assert backend.cuda_available() == reconstructor_module.CUDA

    @pytest.mark.skipif(not backend.cuda_available(), reason="requires CuPy and a CUDA device")
    def test_force_cpu_leaves_no_device_arrays_behind(self, revolt_config):
        """The discriminating check.

        The output dtype does *not* distinguish the two: the old code took the `else`
        branch and called the GPU kernel with `use_float32=False`, which also returns
        float64. What gives it away is the intermediates -- they came back as
        `cupy.ndarray`, because the work really had run on the device. Anything downstream
        expecting NumPy would then break, and `force_cpu` could not serve its main purpose
        of side-stepping a misbehaving GPU.
        """
        cpu = reconstructor_module.tomographicReconstructor(revolt_config, force_cpu=True)
        cpu.build_reconstructor()

        for name in ("Cxx", "Cox", "CnZ", "RecStatSA"):
            matrix = getattr(cpu, name)
            assert type(matrix).__module__.startswith("numpy"), (
                f"{name} is a {type(matrix).__module__}.{type(matrix).__name__}; "
                "force_cpu ran on the GPU"
            )

    @pytest.mark.skipif(not backend.cuda_available(), reason="requires CuPy and a CUDA device")
    def test_gpu_path_still_uses_single_precision(self, revolt_config):
        gpu = reconstructor_module.tomographicReconstructor(revolt_config)
        assert gpu.build_reconstructor().dtype == "float32"
