# Releasing

pyTomoAO is published to [PyPI](https://pypi.org/project/pyTomoAO/) by
`.github/workflows/publish.yml`, which runs when a GitHub **release** is created. Publishing
uses PyPI's trusted publishing (OIDC), so no API token is stored in the repository.

## Version numbering

The version lives in a single place, `pyTomoAO/__init__.py`:

```python
__version__ = "1.0.1"
```

`setup.py` reads it from there at build time, and Sphinx reads it from the installed
package metadata, so bumping that one string is enough.

pyTomoAO follows semantic versioning:

- **Patch** (`1.0.1 → 1.0.2`) — bug fixes, documentation, no interface change.
- **Minor** (`1.0.1 → 1.1.0`) — new features, new configuration keys, backwards compatible.
- **Major** (`1.0.1 → 2.0.0`) — changes that break existing configuration files or scripts.

Renaming a configuration key or changing a default that alters reconstructor output is a
breaking change, even though nothing in Python raises.

## Cutting a release

1. Merge everything you want in the release into `dev`, with CI green.
2. Bump `__version__` in `pyTomoAO/__init__.py` on `dev`.
3. Open a pull request from `dev` into `main` and merge it once CI passes. This also
   publishes the updated documentation to GitHub Pages.
4. Create a GitHub release targeting `main`, tagged `v<version>` (e.g. `v1.1.0`), with
   release notes covering user-visible changes and any migration steps.
5. Creating the release triggers `publish.yml`, which builds an sdist and wheel and uploads
   them to PyPI.
6. Verify the result:

   ```bash
   pip install --upgrade pyTomoAO
   python -c "import pyTomoAO; print(pyTomoAO.__version__)"
   ```

## If a release goes wrong

PyPI does not allow re-uploading a version that has already been published. If a broken
release ships, yank it on PyPI and publish a new patch version — do not attempt to reuse
the number.
