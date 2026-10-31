# Releasing

pyTomoAO is published to [PyPI](https://pypi.org/project/pyTomoAO/) by
`.github/workflows/publish.yml`, which runs when a GitHub **release** is created. Publishing
uses PyPI's trusted publishing (OIDC), so no API token is stored in the repository.

## The changelog

`CHANGELOG.md` at the repository root follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and is rendered in the docs as
{doc}`../changelog`. Every user-visible change should land in the `Unreleased` section as
part of the pull request that makes it, not be reconstructed at release time.

## Version numbering

The version lives in a single place, `pyTomoAO/__init__.py`:

```python
__version__ = "2.0.0"
```

`pyproject.toml` declares `dynamic = ["version"]` and reads it from there at build time via
`[tool.setuptools.dynamic]`, and Sphinx reads it from the installed package metadata, so
bumping that one string is enough. Keep `CITATION.cff` in step with it.

pyTomoAO follows semantic versioning:

- **Patch** (`2.0.0 → 2.0.1`) — bug fixes, documentation, no interface change.
- **Minor** (`2.0.0 → 2.1.0`) — new features, new configuration keys, backwards compatible.
- **Major** (`1.0.1 → 2.0.0`) — changes that break existing configuration files or scripts.

Renaming a configuration key or changing a default that alters reconstructor output is a
breaking change, even though nothing in Python raises.

## Deprecations

An interface that is going away is first deprecated in a minor or patch release: it keeps
working, emits a `DeprecationWarning` that names the version removing it and what to use
instead, and is listed under *Deprecated* in the changelog. It is removed in the next major
release, and the removal is listed under *Breaking*, with a line in that version's migration
guide.

Each deprecation is recorded below until its removal, with the steps to remove it, so a
major release can work through the list.

### Pending removals

**`logger` argument of `tomographicReconstructor` and `fitting`** — deprecated in
2.1.0 (#127), to be removed in 3.0.

1. Delete the `logger` parameter, and its docstring entries (class and `__init__`), from
   `tomographicReconstructor.__init__` (`pyTomoAO/reconstructor.py`) and
   `fitting.__init__` (`pyTomoAO/dm_fitting.py`).
2. Delete the `logger = resolve_logger(...)` line and the import of `resolve_logger` in
   both modules. The constructors then log through the module logger like every other
   method.
3. Delete `pyTomoAO/_logging.py` (`resolve_logger`, `LOGGER_ARGUMENT_REMOVAL`) unless it
   has gained other helpers.
4. In `tests/test_logging.py`, replace
   `test_logger_argument_is_deprecated_but_still_honoured` by a test that `logger=` raises
   `TypeError`.
5. Remove the "The `logger` argument (deprecated)" section of
   `docs/source/user-guide/logging.md`, list the removal under *Breaking* in the changelog
   and add it to the 3.0 migration guide.
6. Remove this entry.

Searching the repository for `resolve_logger` or `LOGGER_ARGUMENT_REMOVAL` finds every
place to change.

## Cutting a release

1. Merge everything you want in the release into `dev`, with CI green. For a major
   release, this includes working through the pending removals listed under
   *Deprecations* above.
2. Bump `__version__` in `pyTomoAO/__init__.py` on `dev`.
3. Update `CHANGELOG.md`: rename the `Unreleased` heading to the new version with today's
   date, add a fresh empty `Unreleased` section above it, and update the comparison links at
   the bottom of the file.
4. Open a pull request from `dev` into `main` and merge it once CI passes. This also
   publishes the updated documentation to GitHub Pages.
5. Create a GitHub release targeting `main`, tagged `v<version>` (e.g. `v1.1.0`), using the
   changelog entry as the release notes.
6. Creating the release triggers `publish.yml`. It builds the sdist and wheel, verifies
   them with `twine check --strict`, installs the wheel into a clean virtualenv and imports
   it, then publishes through the `pypi` GitHub environment. Add a required reviewer to that
   environment in the repository settings if you want releases to need a second pair of
   eyes.
7. Verify the result:

   ```bash
   pip install --upgrade pyTomoAO
   python -c "import pyTomoAO; print(pyTomoAO.__version__)"
   ```

## If a release goes wrong

PyPI does not allow re-uploading a version that has already been published. If a broken
release ships, yank it on PyPI and publish a new patch version — do not attempt to reuse
the number.
