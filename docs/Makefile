# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -W --keep-going
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = Optuna
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

doctest:
	sphinx-autogen source/**/*.rst
	@$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Copy from https://sphinx-gallery.github.io/stable/advanced.html#cleaning-the-gallery-files
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf source/reference/generated/
	rm -rf source/reference/visualization/generated/
	rm -rf source/tutorial/10_key_features
	rm -rf source/tutorial/20_recipes
	rm -f ../tutorial/**/*.db
	rm -rf ../tutorial/20_recipes/artifacts
	rm -rf ../tutorial/20_recipes/tmp
	rm -rf ../tutorial/20_recipes/best_atoms.png
