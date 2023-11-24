# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))


project = 'PINN4GPR '
copyright = '2023, Thomas Rigoni'
author = 'Thomas Rigoni'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.todo", 
              "sphinx.ext.viewcode", 
              "sphinx.ext.autodoc",
              "numpydoc",
              "sphinx.ext.intersphinx"
              ]

templates_path = ['_templates']
exclude_patterns = []

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

nitpicky = True
nitpick_ignore = [
    ("py:class", "pathlib.Path"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "pymunk.space.Space"),
    ("py:class", "numpy.random._generator.Generator"),
    ("py:class", "matplotlib.axes._axes.Axes")
]


numpydoc_class_members_toctree = False
numpydoc_validation_checks = {"all", "GL02", "SS01", "SS02", "SS03", "SS05", "SS06", "ES01", "PR08", "PR09", "RT04", "RT05", "SA01", "EX01", "RT02"}


# exclude classes and members that should not be documented
autodoc_default_options = {
    'exclude-members': "model_config, model_fields"
}
numpydoc_show_inherited_class_members = {
    "dataset_creation.configuration.GprMaxConfig": False
}

html_theme_options = {
    'globaltoc_collapse': False,  # True as default
    'globaltoc_maxdepth': -1,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_title = "PINN4GPR documentation"
html_last_updated_fmt = '%b %d, %Y'
