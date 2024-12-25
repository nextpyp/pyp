# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# add parent directory to $PATH
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "src"))
print("main project path is", MAIN_DIR)
sys.path.insert(0, MAIN_DIR)


# -- Project information -----------------------------------------------------

project = "nextPYP"
copyright = "2025, Bartesaghi Lab"
# author = "Alberto Bartesaghi"

# The full version, including alpha/beta/rc tags
release = "0.1"

html_favicon = '_static/favicon.png'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
#    "sphinx_immaterial",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx_tabs.tabs",
    "sphinx_panels",
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The master toctree document.
master_doc = "index"

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "sphinx_immaterial"
html_theme = "sphinx_book_theme"
# html_theme = "sphinx_rtd_theme"
# html_theme_options = {'page_width': 'auto'}
html_theme_options = { "repository_url": "https://github.com/nextpyp", "use_repository_button": True, "use_issues_button": False, "repository_branch": "master", "use_fullscreen_button": True, "navbar_end": ["theme-switcher", "navbar-icon-links"], "content_width": "1200px" }
html_logo = "_static/nextPYP_logo.svg"
html_title = ""

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css",
    "customizations.css",
    "versions.css"
]

html_js_files = [
    "version-info.js",
    "versions.js"
]

# Resolve function for linkcode extension.
# Thanks to https://github.com/materialsproject/pymatgen/blob/master/docs_rst/conf.py#L324
# Which is based on https://github.com/Lasagne/Lasagne/blob/master/docs/conf.py
def linkcode_resolve(domain, info):
    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        from inspect import getsourcefile, getsourcelines, unwrap
        import os

        obj = unwrap(obj) # unwrap obj nested in decorators
        fn = getsourcefile(obj)
        fn = os.path.relpath(fn, start=MAIN_DIR)
        source, lineno = getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None

    try:
        filename = "src/%s#L%d-%d" % find_source()
    except:
        filename = info["module"].replace(".", "/") + ".py"

    return "https://gitlab.cs.duke.edu/bartesaghilab/pyp/-/blob/master/%s" % filename

def setup(app):
    app.add_css_file('my_theme.css')