# Makefile for the pyTomoAO documentation.
#
#   make html       build the site into build/html
#   make strict     build the way CI does: warnings are errors
#   make livehtml   rebuild and reload in the browser on save
#   make linkcheck  verify external links
#   make clean      remove build output and generated API stubs

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
GENERATEDDIR  = $(SOURCEDIR)/api/generated

.PHONY: help strict livehtml clean Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Same invocation as .github/workflows/docs.yml.
strict:
	@$(SPHINXBUILD) -b html -W --keep-going "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Requires: pip install sphinx-autobuild
livehtml:
	@sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" \
		--watch ../pyTomoAO --ignore "$(GENERATEDDIR)/*" $(SPHINXOPTS) $(O)

clean:
	@rm -rf "$(BUILDDIR)" "$(GENERATEDDIR)"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
