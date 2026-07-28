"""Sphinx configuration for the pyTomoAO documentation.

The rendered site is published to GitHub Pages by ``.github/workflows/docs.yml``.
Build it locally with ``make -C docs html`` (see ``development/documentation``).
"""

import os
import re
import sys
from datetime import date
from importlib import metadata

# Make the package importable for autodoc without requiring an editable install.
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "pyTomoAO"
author = "Jacob Taylor and Uriel Conod"
copyright = f"{date.today().year}, W. M. Keck Observatory"

def _get_release() -> str:
    """Version of the installed package, falling back to the source tree."""
    try:
        return metadata.version("pyTomoAO")
    except metadata.PackageNotFoundError:
        init = os.path.join(os.path.dirname(__file__), "..", "..", "pyTomoAO", "__init__.py")
        with open(init) as fh:
            match = re.search(r'__version__ = "(.*)"', fh.read())
        return match.group(1) if match else "0.0.0"


release = _get_release()
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST (Markdown) ---------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# -- Autodoc / autosummary ---------------------------------------------------

# CuPy is an optional GPU dependency and is never installed on the docs runner.
# Mock it (and its sibling namespaces) so tomographyUtilsGPU can still be imported.
autodoc_mock_imports = ["cupy", "cupyx", "cupy_backends"]

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autosummary_generate = True
autosummary_imported_members = False

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_preprocess_types = True

# sphinx-autodoc-typehints
typehints_defaults = "comma"

# -- Cross-project references ------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"pyTomoAO {version}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_copy_source = False
html_show_sourcelink = False

_repo_url = "https://github.com/KeckObservatory/pyTomoAO"

html_theme_options = {
    "source_repository": f"{_repo_url}/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#0b6e8f",
        "color-brand-content": "#0b6e8f",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5fc4e6",
        "color-brand-content": "#5fc4e6",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": _repo_url,
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16" width="1em" height="1em"><path fill-rule="evenodd" '
                'd="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
                "0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 "
                "1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 "
                "0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 "
                "1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 "
                "2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 "
                '1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}

# -- Misc --------------------------------------------------------------------

linkcheck_ignore = [
    r"https://github\.com/KeckObservatory/pyTomoAO/(issues|pull)/\d+",
]
linkcheck_timeout = 20
