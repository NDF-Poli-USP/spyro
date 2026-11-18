import firedrake as fire
import math
import numpy as np
import pytest
import spyro
from spyro.mpi.spyro_mpi import SpyroEnsemble

try:
    from SeismicMesh import write_velocity_model

    seismic_mesh = True
except ImportError:
    seismic_mesh = False


@pytest.mark.skipif(not seismic_mesh, reason="No SeismicMesh")
def test_read_and_write_segy():
    vp_name = "velocity_models/test"
    segy_file = vp_name + ".segy"
    hdf5_file = vp_name + ".hdf5"
    mesh = fire.UnitSquareMesh(10, 10)
    mesh.coordinates.dat.data[:, 0] *= -1

    V = fire.FunctionSpace(mesh, "CG", 3)
    x, y = fire.SpatialCoordinate(mesh)
    r = 0.2
    xc = -0.5
    yc = 0.5

    vp = fire.Function(V)

    c = fire.conditional((x - xc) ** 2 + (y - yc) ** 2 < r**2, 3.0, 1.5)

    vp.interpolate(c)

    spyro.io.create_segy(vp, V, 10.0 / 1000.0, segy_file)
    write_velocity_model(segy_file, vp_name)

    model = {}

    model["options"] = {
        "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
        "variant": "equispaced",  # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
        "degree": 3,  # p order
        "dimension": 2,  # dimension
    }
    model["mesh"] = {
        "length_z": 1.0,  # depth in km - always positive
        "length_x": 1.0,  # width in km - always positive
        "length_y": 0.0,  # thickness in km - always positive
        "user_mesh": mesh,
        "mesh_file": None,  # specify the mesh file
    }
    model["BCs"] = {
        "status": False,
    }
    model["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": 1.0,  # Final time for event
        "dt": 0.0005,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM
    }
    model["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-1.0, 1.0)],
        "frequency": 5.0,
        "delay": 1.5,
        "receiver_locations": [(-0.0, 0.5)],
    }

    wave = spyro.AcousticWave(dictionary=model)

    vp_read = spyro.io.interpolate(wave, hdf5_file, wave.function_space)

    SpyroEnsemble.write_vtk("velocity_models/test.pvd", vp_read)

    value_at_center = vp_read.at(xc, yc)
    test1 = math.isclose(value_at_center, 3.0)
    value_outside_circle = vp_read.at(xc + r + 0.1, yc)
    test2 = math.isclose(value_outside_circle, 1.5)
    assert all([test1, test2])


def test_saving_and_loading_shot_record():
    from ..inputfiles.model import dictionary

    dictionary["time_axis"]["final_time"] = 0.5
    wave_obj_saving = spyro.AcousticWave(dictionary=dictionary)
    wave_obj_saving.set_mesh(input_mesh_parameters={"edge_length": 0.02})
    wave_obj_saving.set_initial_velocity_model(constant=1.5)
    wave_obj_saving.forward_solve()
    spyro.io.save_shots(wave_obj_saving, file_name="test_shot_record")

    wave_obj_loading = spyro.AcousticWave(dictionary=dictionary)
    wave_obj_loading.set_mesh(input_mesh_parameters={"edge_length": 0.02})
    spyro.io.load_shots(wave_obj_loading, file_name="test_shot_record")


# Slope of the linear field used by the grid tests, along depth, x and y.
DEPTH_SLOPE, X_SLOPE, Y_SLOPE = 100.0, 10.0, 1.0


def _linear_field(
    lengths: tuple[float, ...], grid_spacing: float
) -> tuple[fire.Function, fire.FunctionSpace]:
    """Interpolate a field that grows with depth, x and y at different rates.

    Parameters
    ----------
    lengths : tuple of float
        `(length_z, length_x)` or `(length_z, length_x, length_y)`.
    grid_spacing : float
        Mesh edge length.

    Returns
    -------
    firedrake.Function
        The field on a CG1 space of a spyro `(z, x[, y])` mesh.
    firedrake.FunctionSpace
        That CG1 space.
    """
    dimension = len(lengths)
    mesh_parameters = spyro.meshing.MeshingParameters(
        input_mesh_dictionary={
            "length_z": lengths[0],
            "length_x": lengths[1],
            "length_y": lengths[2] if dimension == 3 else 0.0,
            "mesh_type": "firedrake_mesh",
            "edge_length": grid_spacing,
            "dimension": dimension,
        }
    )
    mesh = spyro.meshing.AutomaticMesh(mesh_parameters=mesh_parameters).create_mesh()
    coordinates = fire.SpatialCoordinate(mesh)
    field = DEPTH_SLOPE * (-coordinates[0]) + X_SLOPE * coordinates[1]
    if dimension == 3:
        field = field + Y_SLOPE * coordinates[2]
    V = fire.FunctionSpace(mesh, "CG", 1)
    return fire.Function(V).interpolate(field), V


@pytest.mark.parametrize("orientation", ["zx", "xz"])
@pytest.mark.parametrize(
    "lengths, grid_spacing",
    [((1.0, 2.0), 0.1), ((0.4, 0.6, 0.2), 0.1)],
    ids=["2d", "3d"],
)
def test_write_function_to_grid_orientation(
    lengths: tuple[float, ...], grid_spacing: float, orientation: str
) -> None:
    """Check the axis order of the grid on a non-square domain.

    With `"zx"` the axes are `(z, x[, y])`, with `"xz"` they are `(x, z[, y])`;
    z always runs from the deepest point up.

    Parameters
    ----------
    lengths : tuple of float
        Domain lengths, `(length_z, length_x[, length_y])`.
    grid_spacing : float
        Grid spacing, also used as the mesh edge length.
    orientation : str
        Axis order to request.
    """
    function, V = _linear_field(lengths, grid_spacing)
    grid = spyro.io.write_function_to_grid(
        function, V, grid_spacing, buffer=True, orientation=orientation
    )

    z_axis, x_axis = (0, 1) if orientation == "zx" else (1, 0)
    axes = {z_axis: (lengths[0], -DEPTH_SLOPE), x_axis: (lengths[1], X_SLOPE)}
    if len(lengths) == 3:
        axes[2] = (lengths[2], Y_SLOPE)
    assert grid.ndim == len(lengths)
    for axis, (length, slope) in axes.items():
        assert abs(grid.shape[axis] - (length / grid_spacing + 1)) <= 1
        steps = np.diff(grid, axis=axis)
        # The buffer keeps the grid a little inside the domain, so a step is
        # up to 5% shorter than grid_spacing; the slopes differ by a factor 10.
        assert np.allclose(steps, steps.flat[0])
        assert np.isclose(steps.flat[0], slope * grid_spacing, rtol=0.1)


def test_write_function_to_grid_rejects_unknown_orientation() -> None:
    """Reject an orientation other than `"zx"` and `"xz"`."""
    function, V = _linear_field((1.0, 2.0), 0.1)
    with pytest.raises(ValueError, match="orientation must be one of"):
        spyro.io.write_function_to_grid(function, V, 0.1, orientation="z_x")


if __name__ == "__main__":
    test_read_and_write_segy()
    test_saving_and_loading_shot_record()
