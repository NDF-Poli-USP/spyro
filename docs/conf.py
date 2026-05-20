# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'spyro'
copyright = '2026, Keith J. Roberts, Alexandre F. G. Olender, Ruben Andres Salas, Daiane I. Dolci, Eduardo Moscatelli de Souza, Thiago Dias dos Santos, Lucas Franceschini, Bruno S. Carmo'
author = 'Keith J. Roberts, Alexandre F. G. Olender, Ruben Andres Salas, Daiane I. Dolci, Eduardo Moscatelli de Souza, Thiago Dias dos Santos, Lucas Franceschini, Bruno S. Carmo, Usama Bin Qasim, Hassan Imran'
release = '0.9.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.extlinks',
	'sphinx.ext.mathjax',
	'sphinx.ext.intersphinx',
	'sphinx.ext.viewcode',
	'sphinx.ext.napoleon',
	'sphinx_copybutton',
]

mathjax_path = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js'

mathjax3_config = {
	'loader': {'load': ['[tex]/mathtools']},
	'tex': {'packages': {'[+]': ['mathtools']}}
}

autoclass_content = 'both'

autodoc_default_options = {
	'special-members': '__call__'
}

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'firedrake'
html_theme_path = ['_themes']
html_static_path = ['_static', 'images']
html_css_files = ['custom.css']
html_show_sphinx = False
html_show_copyright = False

copybutton_prompt_text = "$ "
