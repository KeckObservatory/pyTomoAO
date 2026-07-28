# Documentation

This site is built with [Sphinx](https://www.sphinx-doc.org), written mostly in Markdown via
[MyST](https://myst-parser.readthedocs.io), themed with
[Furo](https://pradyunsg.me/furo/), and published to **GitHub Pages** by
`.github/workflows/docs.yml`.

## Building locally

```bash
pip install -e ".[docs]"
make -C docs html
```

Open `docs/build/html/index.html` in a browser. Other targets:

```bash
make -C docs clean       # remove build output and generated API stubs
make -C docs linkcheck   # verify external links resolve
make -C docs livehtml    # rebuild and reload on save (needs sphinx-autobuild)
```

`livehtml` requires `pip install sphinx-autobuild` and serves the docs at
<http://127.0.0.1:8000>.

To reproduce exactly what CI does, including failing on warnings:

```bash
sphinx-build -b html -W --keep-going docs/source docs/build/html
```

## Layout

```text
docs/
├── Makefile / make.bat
├── requirements.txt         # pinned toolchain used by CI
└── source/
    ├── conf.py
    ├── index.md             # landing page
    ├── _static/custom.css
    ├── _templates/autosummary/   # API page templates
    ├── figures/             # images referenced from the prose pages
    ├── getting-started/
    ├── user-guide/
    ├── tutorials/
    ├── api/                 # index + generated stubs (generated/ is not committed)
    └── development/
```

`docs/source/api/generated/` is produced by `sphinx.ext.autosummary` at build time and is
git-ignored — never edit or commit those files.

## Conventions

- **Markdown by default.** New pages are `.md` and parsed by MyST. Use `{eval-rst}` fenced
  blocks only where a directive has no MyST equivalent, as in the API pages.
- **Cross-reference, do not repeat.** Link to API objects with
  `` {py:class}`~pyTomoAO.fitting.fitting` `` and to pages with `` {doc}`../user-guide/gpu` ``.
  Both are checked at build time, so a rename that breaks a link fails CI.
- **Document units.** Every physical quantity in
  {doc}`../user-guide/configuration` states its unit; keep that up.
- **Say what is not true.** Where the code does not do what a reader would assume — an
  unused configuration section, a flag ignored on the GPU path — say so explicitly rather
  than leaving it out.
- **API docs come from docstrings.** To improve a class's reference page, improve its
  NumPy-style docstring in the source.

## Adding a page

1. Create the `.md` file in the appropriate directory.
2. Add it to the relevant `toctree` in `docs/source/index.md` — pages left out of every
   toctree raise a warning, and warnings fail the build.
3. Link to it from the section index so it is reachable by navigation, not just the sidebar.

## Publishing

`.github/workflows/docs.yml` handles both halves:

- **On pull requests and pushes to `dev`** the docs are built with `-W` (warnings are
  errors) and the HTML is uploaded as a workflow artifact. Nothing is published.
- **On pushes to `main`** the same build runs and the result is deployed to GitHub Pages via
  `actions/deploy-pages`.

The workflow can also be triggered manually from the Actions tab
(`workflow_dispatch`).

### One-time repository setup

GitHub Pages must be set to build from Actions for the deploy step to work:

1. **Settings → Pages → Build and deployment → Source**: select **GitHub Actions**.
2. The workflow already requests the required `pages: write` and `id-token: write`
   permissions; no personal access token or `gh-pages` branch is needed.

The site is served at <https://keckobservatory.github.io/pyTomoAO/>.

:::{note}
pyTomoAO previously targeted Read the Docs. That setup has been removed in favour of GitHub
Pages so that the docs build with the same Actions runner as the tests, and so that a
failing docs build blocks a pull request the same way a failing test does.
:::
