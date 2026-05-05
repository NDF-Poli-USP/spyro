.. spyro documentation master file, created by
   sphinx-quickstart on Sun May  3 00:07:40 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

spyro: seismic parallel inversion and reconstruction optimization framework
============================================

Wave modeling in Firedrake

spyro is a Python library for modeling waves. The main
functionality is a set of forward and adjoint wave propagators for solving the acoustic wave equation in the time domain.
These wave propagators can be used to form complete full waveform inversion (FWI) applications. See the [notebooks](https://github.com/Olender/spyro-1/tree/main/notebook_tutorials).
To implement these solvers, spyro uses the finite element package [Firedrake](https://www.firedrakeproject.org/index.html).

To use spyro, you'll need to have some knowledge of Python and some basic concepts in inverse modeling relevant to active-source seismology.

If you want to know more or cite our code please see our open access publication: https://gmd.copernicus.org/articles/15/8639/2022/gmd-15-8639-2022.html


.. toctree::
   :maxdepth: 2
   :caption: Contents:

