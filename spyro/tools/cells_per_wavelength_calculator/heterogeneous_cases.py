"""Heterogeneous cases of the cells-per-wavelength calculator.

Each case is a dataclass whose defaults describe it, given as the
calculator ``"heterogeneous_case"`` parameter. Any field can be replaced
when creating it, e.g. ``ElasticSyntheticCase(final_time=3.0)``. A case
read from files subclasses
:class:`~.heterogeneous_case.MaterialFileCase`, and a case defined by a
function of the coordinates subclasses
:class:`~.heterogeneous_case.MaterialModelCase`.
"""

from dataclasses import dataclass

import numpy as np

from .elastic_2d_material_model import elastic_2d_material
from .heterogeneous_case import MaterialModelCase


@dataclass(frozen=True, kw_only=True)
class ElasticSyntheticCase(MaterialModelCase):
    """Synthetic pre-salt section, the 2D isotropic elastic case.

    Its material is
    :func:`~.elastic_2d_material_model.elastic_2d_material`.

    Attributes
    ----------
    name : str
        Default is ``"elastic_synthetic"``.
    length_z : float
        Default is 5.0.
    length_x : float
        Default is 10.0.
    """

    name: str = "elastic_synthetic"
    length_z: float = 5.0
    length_x: float = 10.0

    def material(self, z: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate the pre-salt material.

        Parameters
        ----------
        z : np.ndarray
            Vertical coordinates, negative below the surface.
        x : np.ndarray
            Horizontal coordinates, broadcastable against ``z``.

        Returns
        -------
        dict
            P-wave velocity, S-wave velocity and density arrays.
        """
        return elastic_2d_material(z, x, self.length_x)
