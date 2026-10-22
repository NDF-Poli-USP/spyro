"""Meshing parameters for Spyro.

Mesh type selection, dimension handling, and the parameters used for
automatic mesh generation based on wavelength constraints.
"""

from .constants import cells_per_wavelength
from .gmsh_options import GmshOptions
from .parameters import MeshingParameters

__all__ = ["GmshOptions", "MeshingParameters", "cells_per_wavelength"]
