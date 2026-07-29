import logging
from importlib import resources

from .atmosphereParametersClass import atmosphereParameters
from .dm_fitting import fitting
from .dmParametersClass import dmParameters
from .lgsAsterismParametersClass import lgsAsterismParameters
from .lgsWfsParametersClass import lgsWfsParameters
from .reconstructor import tomographicReconstructor
from .tomographyParametersClass import tomographyParameters

__all__ = [
    "atmosphereParameters",
    "dmParameters",
    "example_config",
    "fitting",
    "lgsAsterismParameters",
    "lgsWfsParameters",
    "list_example_configs",
    "tomographicReconstructor",
    "tomographyParameters",
]

# Reference configurations shipped inside the package, so that a pip install is enough to
# run the documented examples. Writing one of these from scratch is not a realistic
# starting point: validLLMap and validActuators are hand-authored 2D maps of several
# hundred entries each.
_EXAMPLE_CONFIGS = {
    "kapa": "tomography_config_kapa.yaml",
    "kapa-single-channel": "tomography_config_kapa_single_channel.yaml",
    "revolt": "reconstructor_config_revolt.yaml",
    "keck": "tomography_config.yaml",
}


def list_example_configs():
    """Names accepted by :func:`example_config`.

    Returns
    -------
    list of str
        Sorted configuration names.
    """
    return sorted(_EXAMPLE_CONFIGS)


def example_config(name="kapa"):
    """Filesystem path to one of the bundled reference configurations.

    Parameters
    ----------
    name : str, optional
        One of the names returned by :func:`list_example_configs` (default ``"kapa"``):

        - ``"kapa"`` -- Keck/KAPA, four sodium LGS, 20x20 lenslets
        - ``"kapa-single-channel"`` -- KAPA reduced to a single WFS channel
        - ``"revolt"`` -- REVOLT, 1.2 m pupil, single WFS
        - ``"keck"`` -- Keck, four LGS, 7.9 m pupil

    Returns
    -------
    str
        Absolute path to the YAML file.

    Raises
    ------
    KeyError
        If ``name`` is not a bundled configuration.

    Examples
    --------
    >>> from pyTomoAO import example_config, tomographicReconstructor
    >>> rec = tomographicReconstructor(example_config("kapa"))  # doctest: +SKIP
    """
    try:
        filename = _EXAMPLE_CONFIGS[name]
    except KeyError:
        raise KeyError(
            f"Unknown example configuration {name!r}. Available: {list_example_configs()}"
        ) from None
    return str(resources.files(__name__) / "data" / filename)


# Libraries should not configure logging for the application that imports them.
# The NullHandler keeps "No handlers could be found" warnings away while leaving
# handler and level configuration to the caller:
#
#     logging.basicConfig(level=logging.INFO)
logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "2.0.0"
