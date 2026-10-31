# Logging

pyTomoAO reports what it is doing through Python's standard {mod}`logging` module and never
prints. Each module has its own logger, named after the module and a child of the
`pyTomoAO` logger:

| Logger | Logs |
|---|---|
| `pyTomoAO.reconstructor` | configuration loading, the reconstructor build, the fitting assembly |
| `pyTomoAO.dm_fitting` | influence functions and the fitting matrix |
| `pyTomoAO.slodar` | SLODAR geometry, covariance maps, profile fits, noise estimates |

As a library, pyTomoAO only attaches a {class}`logging.NullHandler` to the `pyTomoAO`
logger and sets no level. Until the application configures logging, **nothing is shown,
not even warnings and errors**: the NullHandler also stops Python's last-resort handler,
which would otherwise print them. The application decides what to display.

## Showing the messages

The simplest way, for a script or a notebook:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

To show only pyTomoAO's messages, or a different amount of detail per module:

```python
logging.basicConfig(level=logging.WARNING)                    # other libraries: quiet
logging.getLogger("pyTomoAO").setLevel(logging.INFO)          # pyTomoAO: progress
logging.getLogger("pyTomoAO.slodar").setLevel(logging.DEBUG)  # SLODAR: every L0 tried
```

A level is checked only on the logger that emits a message, not on its parents. A level
set on a module logger therefore overrides the `pyTomoAO` one for that module, whatever is
set on `pyTomoAO` later.

To keep only pyTomoAO's errors (with no level set on the module loggers):

```python
logging.getLogger("pyTomoAO").setLevel(logging.ERROR)
```

To silence pyTomoAO completely, whatever the module levels, stop its messages from reaching
the root logger's handlers:

```python
logging.getLogger("pyTomoAO").propagate = False
```

Handlers attached directly to a pyTomoAO logger still receive its messages.

Messages can be written to a file, formatted or filtered with the usual handlers, attached
to the `pyTomoAO` logger or to the root logger.

## The `logger` argument (deprecated)

`tomographicReconstructor` and `fitting` accept a `logger` argument. It is deprecated and
will be removed in pyTomoAO 3.0: the logger passed in only ever received the constructor's
own messages, while everything logged later (the reconstructor build, the fitting assembly)
went to the module loggers anyway. Passing it now emits a `DeprecationWarning`. To route
pyTomoAO's output somewhere, attach a handler to the `pyTomoAO` logger instead:

```python
handler = logging.FileHandler("pytomoao.log")
logging.getLogger("pyTomoAO").addHandler(handler)
logging.getLogger("pyTomoAO").setLevel(logging.INFO)
```
