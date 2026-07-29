"""Lazy access to matplotlib.

Plotting is a small part of the package -- two methods on the reconstructor and one
optional branch of the influence-function builder -- so matplotlib is an optional
dependency rather than something every ``import pyTomoAO`` drags in. That matters on a
real-time control machine, where the reconstructor is built but never plotted.

This lives in its own module so that both :mod:`pyTomoAO.reconstructor` and
:mod:`pyTomoAO.dm_fitting` can use it without an import cycle between them.
"""


def pyplot():
    """Import and return ``matplotlib.pyplot``.

    Returns
    -------
    module
        ``matplotlib.pyplot``.

    Raises
    ------
    ImportError
        If matplotlib is not installed, naming the extra that provides it.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Plotting requires matplotlib, which is an optional dependency of pyTomoAO. "
            'Install it with: pip install "pyTomoAO[plot]"'
        ) from exc
    return plt
