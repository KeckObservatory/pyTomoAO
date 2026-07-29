"""Guards on the package's import surface (#115).

`__init__.py` re-exports the main classes, which is convenient but means a module sharing a
name with one of its classes becomes unreachable: the class wins the package namespace.
That is what happened to `pyTomoAO.tomographicReconstructor` and `pyTomoAO.fitting` before
2.0, and it broke `unittest.mock` string targets on Python 3.9 and 3.10.
"""

import importlib
import inspect
import pkgutil

import pytest

import pyTomoAO

SUBMODULES = sorted(m.name for m in pkgutil.iter_modules(pyTomoAO.__path__))


def test_there_are_submodules_to_check():
    assert SUBMODULES, "no submodules discovered; the guard below would be vacuous"


def _import_or_skip(name):
    """Import a submodule, skipping the ones whose optional dependencies are absent.

    `tomographyUtilsGPU` needs CuPy, which CI does not have.
    """
    try:
        return importlib.import_module(f"pyTomoAO.{name}")
    except ImportError as exc:
        pytest.skip(f"pyTomoAO.{name} is not importable here: {exc}")


@pytest.mark.parametrize("name", SUBMODULES)
def test_module_name_does_not_collide_with_one_of_its_classes(name):
    module = _import_or_skip(name)
    classes = [
        obj_name
        for obj_name, obj in vars(module).items()
        if inspect.isclass(obj) and obj.__module__ == module.__name__
    ]
    assert name not in classes, (
        f"pyTomoAO.{name} defines a class also called {name!r}; if it is re-exported in "
        "__init__.py the class shadows the module in the package namespace"
    )


@pytest.mark.parametrize("name", SUBMODULES)
def test_attribute_access_yields_the_module(name):
    """`import pyTomoAO.x` must make `pyTomoAO.x` the module, not something else."""
    _import_or_skip(name)
    assert inspect.ismodule(getattr(pyTomoAO, name)), f"pyTomoAO.{name} is not a module"


def test_public_classes_are_importable_from_the_package():
    for name in pyTomoAO.__all__:
        assert hasattr(pyTomoAO, name), f"{name} is in __all__ but not importable"


def test_string_patch_targets_resolve():
    """The failure mode that made the old layout hurt: mock resolving a dotted path."""
    from unittest.mock import patch

    with patch("pyTomoAO.reconstructor.atmosphereParameters"), patch("pyTomoAO.dm_fitting.plt"):
        pass
