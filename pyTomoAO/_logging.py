"""Logging helpers shared by the pyTomoAO modules.

pyTomoAO logs through module-level loggers (``logging.getLogger(__name__)``), all children
of the ``pyTomoAO`` logger, and leaves handlers and levels to the application.

``resolve_logger`` exists only to deprecate the ``logger`` constructor arguments (#127). The
steps to remove them in 3.0 are listed in docs/source/development/releasing.md, under
"Pending removals".
"""

import warnings

#: Version in which the deprecated ``logger`` constructor arguments will be removed.
LOGGER_ARGUMENT_REMOVAL = "3.0"


def resolve_logger(logger, default, owner):
    """Logger for a constructor that still accepts the deprecated ``logger`` argument.

    Parameters
    ----------
    logger : logging.Logger or None
        The argument as passed; ``None`` (the default) selects ``default``.
    default : logging.Logger
        The owning module's logger.
    owner : str
        Name of the class, for the warning message.

    Returns
    -------
    logging.Logger
        ``default``, or the passed logger (with a ``DeprecationWarning``) for backward
        compatibility: it receives the constructor's own messages, as before.
    """
    if logger is None:
        return default
    warnings.warn(
        f"The 'logger' argument of {owner} is deprecated and will be removed in pyTomoAO "
        f"{LOGGER_ARGUMENT_REMOVAL}. It only ever received the constructor's messages; "
        "everything else is logged through the module loggers. Configure them with the "
        "logging module instead, e.g. logging.getLogger('pyTomoAO').setLevel(logging.INFO).",
        DeprecationWarning,
        stacklevel=3,
    )
    return logger
