Installing spyro
================

For installation, spyro needs NumPy, SciPy, Matplotlib, segyio, Firedrake, and SeismicMesh.

For NumPy, SciPy, Matplotlib, and segyio you can use pip, such as:
::
    python -m pip install --user numpy scipy matplotlib segyio

In order to install Firedrake you should follow instructions on https://www.firedrakeproject.org/download.html, curretly for Ubuntu and MacOS X those are:
::
    curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
    python3 firedrake-install

Before using spyro you have to active the Firedrake venv 
::
    source firedrake/bin/activate

In order to install SeismicMesh you should follow the instructions on https://seismicmesh.readthedocs.io/en/par3d/install.html

After all the requirements are installed and running on the Firedrake venv you can just run
::
    pip install .
