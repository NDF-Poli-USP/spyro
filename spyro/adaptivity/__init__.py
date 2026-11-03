"""Automatic mesh adaptation driven by a two-level trace-error pilot.

Three strategies choose the next mesh of the loop: uniform cells per
wavelength (``cpw``), the smoothed velocity gradient (``gradient``) and the
dual-weighted residual of the trace error computed with spyro's automated
adjoint (``adjoint``). See :mod:`spyro.adaptivity.automatic_loop`.
"""

from .mesh_generation import (RectangularDomain, generate_mesh,
                              read_triangulation, local_mesh_size,
                              count_kmv_dofs)
from .size_fields import (VelocityGrid, AdaptiveSizeField, gradient_density,
                          adjoint_density)
from .pilot import TwoLevelPilot, PilotResult

__all__ = [
    "RectangularDomain",
    "generate_mesh",
    "read_triangulation",
    "local_mesh_size",
    "count_kmv_dofs",
    "VelocityGrid",
    "AdaptiveSizeField",
    "gradient_density",
    "adjoint_density",
    "TwoLevelPilot",
    "PilotResult",
]
