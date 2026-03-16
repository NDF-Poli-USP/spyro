from copy import deepcopy
import firedrake as fire
from ..meshing import MeshingParameters, AutomaticMesh
from ..io import write_function_to_grid


def velocity_to_grid(wave_obj, grid_spacing, output=False):
    """
    Convert velocity model to a regular grid with specified spacing.

    Interpolates the velocity model from a :class:`Wave` object onto a structured grid
    with uniform spacing. This is useful for visualization, creating SEG-Y
    outputs, or interfacing with other software that requires regular grids.

    Parameters
    ----------
    wave_obj : `spyro.Wave`
        Wave object containing the velocity model and mesh information.
    grid_spacing : float
        Desired uniform grid spacing in all dimensions (km).
    output : bool, optional
        If True, save the gridded velocity to a VTK file for visualization.
        Default is False.

    Returns
    -------
    grid_velocity_data : dict
        Dictionary containing:
        - 'vp_values' : ndarray
            Velocity values on the regular grid.
        - 'grid_spacing' : float
            The grid spacing used.
        - 'length_z' : float
            Domain length in z-direction.
        - 'length_x' : float
            Domain length in x-direction.
        - 'length_y' : float or None
            Domain length in y-direction (None for 2D).
        - 'abc_pad_length' : float
            Length of absorbing boundary padding.

    Notes
    -----
    The function creates a new structured mesh and interpolates the velocity
    model onto it using CG1 (continuous piecewise linear) elements. This
    allows conversion from any mesh type or polynomial degree to a regular grid.

    Examples
    --------
    >>> grid_data = velocity_to_grid(wave_obj, grid_spacing=0.01, output=True)
    >>> vp_gridded = grid_data['vp_values']
    """

    mesh_parameters_original = wave_obj.mesh_parameters
    u, V = change_scalar_field_resolution(
        wave_obj.initial_velocity_model,
        wave_obj,
        grid_spacing,
    )
    z = write_function_to_grid(u, V, grid_spacing, buffer=True)
    if output:
        output_file = fire.VTKFile("debug_velocity_to_grid.pvd")
        output_file.write(u)

    if mesh_parameters_original.abc_pad_length is None:
        pad_length = 0.0
    else:
        pad_length = deepcopy(mesh_parameters_original.abc_pad_length)

    grid_velocity_data = {
        "vp_values": z,
        "grid_spacing": grid_spacing,
        "length_z": deepcopy(mesh_parameters_original.length_z),
        "length_x": deepcopy(mesh_parameters_original.length_x),
        "length_y": deepcopy(mesh_parameters_original.length_y),
        "abc_pad_length": pad_length,
    }
    return grid_velocity_data


def change_scalar_field_resolution(scalar_field, wave_obj, grid_spacing):
    """
    Change a scalar field to a different resolution using a structured grid.

    Creates a new structured mesh with specified grid spacing and interpolates
    the scalar field onto this new mesh using continuous Galerkin (CG1)
    elements. The original field can be on any mesh type or polynomial degree.

    This is useful for visualization in ParaView and for generating SEG-Y
    outputs from high-order or unstructured simulations.

    Parameters
    ----------
    scalar_field : firedrake.Function
        The scalar field to be re-gridded (typically a velocity model).
    wave_obj : spyro.Wave
        Wave object containing the original mesh parameters.
    grid_spacing : float
        Desired grid spacing (edge length) for the new structured mesh (km).

    Returns
    -------
    u : firedrake.Function
        The scalar field interpolated onto the new structured mesh.
    V : firedrake.FunctionSpace
        The CG1 function space on the new structured mesh.

    Notes
    -----
    The interpolation uses allow_missing_dofs=True to handle cases where the
    original mesh extends beyond the new mesh boundaries. The new mesh is
    created using Firedrake's built-in mesh generation with the same domain
    dimensions as the original.

    Examples
    --------
    >>> u_grid, V_grid = change_scalar_field_resolution(
    ...     wave_obj.initial_velocity_model, wave_obj, grid_spacing=0.01
    ... )
    """
    mesh_parameters_original = wave_obj.mesh_parameters
    input_mesh_parameters_cg1 = {
        "dimension": mesh_parameters_original.dimension,
        "length_z": mesh_parameters_original.length_z,
        "length_x": mesh_parameters_original.length_x,
        "length_y": mesh_parameters_original.length_y,
        "mesh_type": "firedrake_mesh",
        "edge_length": grid_spacing,
        "abc_pad_length": mesh_parameters_original.abc_pad_length
    }
    meshing_parameters_cg1 = MeshingParameters(
        input_mesh_dictionary=input_mesh_parameters_cg1,
        comm=mesh_parameters_original.comm
    )
    meshing_obj = AutomaticMesh(meshing_parameters_cg1)
    mesh = meshing_obj.create_mesh()
    V = fire.FunctionSpace(mesh, "CG", 1)
    u = fire.Function(V).interpolate(scalar_field, allow_missing_dofs=True)

    return (u, V)
