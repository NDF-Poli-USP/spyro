"""Velocity-adapted simplex meshes built with the Animate toolkit.

Animate (https://github.com/mesh-adaptation/animate) adapts an existing
Firedrake mesh to a Riemannian metric through PETSc's DMPlex interface to Mmg
(2D) and ParMmg (3D). It does not create meshes from a geometry, so the
workflow here starts from a uniform background rectangle and remeshes it so
that the local edge length follows ``h = vp / (frequency * cpw)``, the same
sizing rule used by :func:`spyro.meshing.meshing_utils.vp_to_sizing`.

Animate is an optional dependency and requires PETSc configured with
``--download-mmg`` (and ``--download-parmmg`` for 3D).
"""

import math

import firedrake as fire
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .firedrake_based_wrappers import rectangle_mesh

try:
    import animate
except ImportError:
    animate = None


def check_animate(func):
    """Raise ImportError when Animate is not available.

    Parameters
    ----------
    func : callable
        Function that depends on Animate.

    Returns
    -------
    callable
        Wrapped function that raises ImportError if Animate is missing.
    """
    def wrapper(*args, **kwargs):
        if animate is None:
            raise ImportError(
                "Animate is not available. Please install it (and PETSc "
                "with Mmg) to use this function."
            )
        return func(*args, **kwargs)

    return wrapper


def sample_grid_velocity(
    function_space: fire.FunctionSpace,
    vp_grid: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> fire.Function:
    """Sample a gridded 2D velocity model at the nodes of a function space.

    Parameters
    ----------
    function_space : firedrake.FunctionSpace
        Scalar function space on a 2D mesh with spyro's (z, x) coordinates.
    vp_grid : numpy.ndarray
        Velocity array of shape ``(nz, nx)`` in the convention returned by
        :func:`spyro.io.segy_io.read_segy_velocity_model`: row 0 is
        ``z = z_min`` (deepest sample) and column 0 is ``x = x_min``.
    bbox : tuple of float
        Grid extent ``(z_min, z_max, x_min, x_max)``.

    Returns
    -------
    firedrake.Function
        Velocity interpolated linearly at the function space nodes. Nodes
        outside the grid take the value of the nearest grid sample.
    """
    z_min, z_max, x_min, x_max = bbox
    nz, nx = vp_grid.shape
    z_axis = np.linspace(z_min, z_max, nz)
    x_axis = np.linspace(x_min, x_max, nx)
    interpolator = RegularGridInterpolator(
        (z_axis, x_axis), vp_grid, method="linear"
    )

    mesh = function_space.mesh()
    vector_space = fire.VectorFunctionSpace(mesh, function_space.ufl_element())
    coordinates = fire.Function(vector_space).interpolate(
        fire.SpatialCoordinate(mesh)
    )
    points = coordinates.dat.data_ro.copy()
    points[:, 0] = np.clip(points[:, 0], z_min, z_max)
    points[:, 1] = np.clip(points[:, 1], x_min, x_max)

    vp = fire.Function(function_space, name="velocity")
    vp.dat.data[:] = interpolator(points)
    return vp


@check_animate
def velocity_metric(
    vp: fire.Function,
    frequency: float,
    cpw: float,
    h_min: float,
    h_max: float,
    gradation_factor: float = 1.3,
) -> "animate.RiemannianMetric":
    """Build the isotropic metric that prescribes a cells-per-wavelength size.

    Parameters
    ----------
    vp : firedrake.Function
        P1 velocity field on the mesh that will be adapted.
    frequency : float
        Source frequency used to compute the wavelength.
    cpw : float
        Cells per wavelength.
    h_min : float
        Smallest edge length allowed in the adapted mesh.
    h_max : float
        Largest edge length allowed in the adapted mesh.
    gradation_factor : float, optional
        Maximum ratio between adjacent edge lengths (Mmg ``hgrad``).

    Returns
    -------
    animate.RiemannianMetric
        Metric ``M = h**-2 I`` with ``h = vp / (frequency * cpw)``.
    """
    mesh = vp.function_space().mesh()
    dim = fire.SpatialCoordinate(mesh).ufl_shape[0]
    tensor_space = fire.TensorFunctionSpace(mesh, "CG", 1)
    metric = animate.RiemannianMetric(tensor_space)
    metric.set_parameters({
        "dm_plex_metric": {
            "h_min": h_min,
            "h_max": h_max,
            "gradation_factor": gradation_factor,
        }
    })
    h = vp / (frequency * cpw)
    metric.interpolate(fire.Identity(dim) / h**2)
    return metric


@check_animate
def create_animate_velocity_mesh(
    vp_grid: np.ndarray,
    bbox: tuple[float, float, float, float],
    frequency: float,
    cpw: float,
    h_min: float | None = None,
    h_max: float | None = None,
    num_iterations: int = 3,
    gradation_factor: float = 1.3,
) -> fire.MeshGeometry:
    """Create a 2D triangular mesh adapted to a gridded velocity model.

    A uniform background mesh with edge length ``h_max`` is remeshed
    ``num_iterations`` times. Each pass resamples ``vp_grid`` on the current
    mesh (rather than projecting the previous field), so sharp velocity
    contrasts are not smeared between iterations.

    Parameters
    ----------
    vp_grid : numpy.ndarray
        Velocity array of shape ``(nz, nx)``, see :func:`sample_grid_velocity`.
    bbox : tuple of float
        Domain extent ``(z_min, z_max, x_min, x_max)``.
    frequency : float
        Source frequency used to compute the wavelength.
    cpw : float
        Cells per wavelength.
    h_min : float, optional
        Smallest edge length allowed. Defaults to the size required by the
        minimum velocity of ``vp_grid``.
    h_max : float, optional
        Largest edge length allowed. Defaults to the size required by the
        maximum velocity of ``vp_grid``.
    num_iterations : int, optional
        Number of resample-and-adapt passes.
    gradation_factor : float, optional
        Maximum ratio between adjacent edge lengths (Mmg ``hgrad``).

    Returns
    -------
    firedrake.MeshGeometry
        Adapted triangular mesh with spyro's (z, x) orientation.

    Raises
    ------
    ValueError
        If ``num_iterations`` is smaller than one.
    """
    if num_iterations < 1:
        raise ValueError(f"num_iterations must be positive, not {num_iterations}")
    if h_min is None:
        h_min = float(np.min(vp_grid)) / (frequency * cpw)
    if h_max is None:
        h_max = float(np.max(vp_grid)) / (frequency * cpw)

    z_min, z_max, x_min, x_max = bbox
    length_z = z_max - z_min
    length_x = x_max - x_min
    mesh = rectangle_mesh(
        math.ceil(length_z / h_max), math.ceil(length_x / h_max),
        length_z, length_x,
    )
    mesh.coordinates.dat.data[:, 0] += z_max
    mesh.coordinates.dat.data[:, 1] += x_min

    for _ in range(num_iterations):
        vp = sample_grid_velocity(fire.FunctionSpace(mesh, "CG", 1), vp_grid, bbox)
        metric = velocity_metric(
            vp, frequency, cpw, h_min, h_max, gradation_factor=gradation_factor
        )
        mesh = animate.adapt(mesh, metric)
    return mesh
