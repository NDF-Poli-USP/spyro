import importlib.util
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import firedrake as fire
import meshio
import numpy as np
import pytest

from spyro.io.segy_io import (
    create_grid_dictionary_from_segy,
    create_segy_from_grid,
    read_segy_velocity_model,
)
from spyro.meshing import meshing_2d, meshing_winslow2d
from spyro.meshing.meshing_functions import AutomaticMesh
from spyro.meshing.meshing_parameters import MeshingParameters
from spyro.meshing.meshing_utils import create_sizing_function
from spyro.mpi.spyro_mpi import SpyroEnsemble
from spyro.utils import scalar_conditional_to_grid

from .gmsh_mesh_checks import (
    VELOCITY_MODELS_DIR,
    assert_bounds,
    assert_cell_count,
    assert_cell_regions,
    assert_cell_type,
    assert_measure,
    assert_no_degenerate_cells,
    assert_region_above,
    create_gmsh_mesh,
    hyperellipse_area,
    measure,
    physical_groups,
    water_depths_at_edges,
    write_segy_traces,
)


HAS_NUMBA = importlib.util.find_spec("numba") is not None

LENGTH_Z = 3000.0
LENGTH_X = 5000.0
PADDING = 1000.0
HYPER_N = 4.0

# Structured meshes are transfinite, so their cell count is exact. Unstructured
# counts may change slightly between Gmsh versions; the baselines were made
# with Gmsh 4.13.1.
UNSTRUCTURED_CELL_COUNT_RTOL = 0.02
# The hyperelliptical boundary is a polyline, so its area is slightly smaller
# than the exact hyperellipse.
HYPERELLIPSE_AREA_RTOL = 0.01

WATER_INTERFACE_CASES = pytest.mark.parametrize(
    "water_interface", [True, False], ids=["water", "no_water"]
)
PADDING_CASES = pytest.mark.parametrize(
    "padding_type", [None, "rectangular", "hyperelliptical"]
)


# Winslow smoothing moves nodes but does not change the topology, so the
# structured cell count does not depend on the Winslow implementation.
STRUCTURED_EXPECTED_CELLS = {
    # (water_interface, padding_type): expected_cells
    (True, None): 770,
    (True, "rectangular"): 1421,
    (True, "hyperelliptical"): 1323,
    (False, None): 735,
    (False, "rectangular"): 1372,
    (False, "hyperelliptical"): 1225,
}


UNSTRUCTURED_EXPECTED_CELLS = {
    # (extend_segy, water_interface, padding_type): expected_cells
    (True, True, None): 17196,
    (True, True, "rectangular"): 24932,
    (True, True, "hyperelliptical"): 24526,
    (True, False, None): 17304,
    (True, False, "rectangular"): 25113,
    (True, False, "hyperelliptical"): 24653,
    (False, True, None): 17196,
    (False, True, "rectangular"): 18086,
    (False, True, "hyperelliptical"): 18115,
    (False, False, None): 17304,
    (False, False, "rectangular"): 18218,
    (False, False, "hyperelliptical"): 18215,
}


def _base_gmsh2d_parameters(output_filename):
    """Parameters for the 2-D velocity model."""
    return {
        "mesh_type": "gmsh_mesh",
        "dimension": 2,
        "length_z": LENGTH_Z,
        "length_x": LENGTH_X,
        "output_filename": str(output_filename),
        "segy_velocity_model": str(VELOCITY_MODELS_DIR / "vs_example_2D.segy"),
        "cells_per_wavelength": 2.0,
        "source_frequency": 10.0,
        "padding_x": PADDING,
        "padding_z": PADDING,
        "hyper_n": HYPER_N,
        "hmin_segy": 0.0,
        "grade": 0.1,
        "water_search_value": 0.0,
        "vp_water": 500.0,
        "min_element_size": 150.0,
        "winslow_iterations": 100,
        "winslow_omega": 0.5,
        "h_padding": 500.0,
    }


@pytest.mark.parametrize(
    "implementation",
    [
        "fast",
        pytest.param(
            "numba",
            marks=pytest.mark.skipif(not HAS_NUMBA, reason="Numba is not installed."),
        ),
    ],
)
@pytest.mark.parametrize("n_samples, n_traces", [(7, 13), (13, 7)])
def test_gmsh2d_winslow_non_square_segy(
    tmp_path: Path,
    implementation: str,
    n_samples: int,
    n_traces: int,
) -> None:
    """Match Winslow smoothing sampled at the actual SEG-Y coordinates.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the SEG-Y model and generated meshes.
    implementation : str
        Fast or Numba Winslow implementation to exercise.
    n_samples : int
        Number of depth samples per SEG-Y trace.
    n_traces : int
        Number of SEG-Y traces along x.
    """
    sample_z = np.linspace(0.0, 1.0, n_samples)[:, None]
    trace_x = np.linspace(0.0, 1.0, n_traces)[None, :]
    velocity = 1800.0 + 800.0 * (sample_z > 0.45) + 400.0 * (trace_x > 0.35)
    model_file = tmp_path / "model.segy"
    create_segy_from_grid(velocity, str(model_file), rotate=True)
    parameters = _base_gmsh2d_parameters(tmp_path / "initial.msh")
    parameters.update(
        segy_velocity_model=str(model_file),
        structured_mesh=True,
        apply_winslow=False,
        water_interface=False,
        grade=None,
        min_element_size=500.0,
        winslow_implementation=implementation,
        winslow_iterations=10,
        winslow_omega=0.2,
    )
    meshing_2d.generate_gmsh_2d_mesh(MeshingParameters(input_mesh_dictionary=parameters))
    initial_mesh = meshio.read(parameters["output_filename"])
    # Files use (z, x), whereas the smoothers operate on (x, z).
    points = initial_mesh.points[:, [1, 0]]
    quads = np.vstack([block.data for block in initial_mesh.cells if block.type == "quad"])
    x, z = points.T
    vertical_boundary = np.isclose(x, 0.0) | np.isclose(x, LENGTH_X)
    horizontal_boundary = np.isclose(z, -LENGTH_Z) | np.isclose(z, 0.0)
    sizing, _, _, sample_count, trace_count = create_sizing_function(
        str(model_file), bbox=(-LENGTH_Z, 0.0, 0.0, LENGTH_X),
        wl=parameters["cells_per_wavelength"], freq=parameters["source_frequency"],
        grade=None, vp_water=parameters["vp_water"],
    )
    assert (sample_count, trace_count) == (n_samples, n_traces)
    grid_x = np.linspace(0.0, LENGTH_X, n_traces)
    grid_z = np.linspace(-LENGTH_Z, 0.0, n_samples)
    grid_xx, grid_zz = np.meshgrid(grid_x, grid_z, indexing="ij")
    smoother = (
        meshing_winslow2d.winslow_smooth_vectorized if implementation == "fast"
        else meshing_winslow2d.winslow_smooth_numba
    )
    expected = smoother(
        points=points.copy(), quads=quads,
        segy_grid_x=grid_x, segy_grid_z=grid_z,
        segy_grid_vals=sizing(grid_xx.ravel(), grid_zz.ravel()).reshape(n_traces, n_samples),
        move_all=set(np.flatnonzero(~(vertical_boundary | horizontal_boundary))),
        move_x_only=set(np.flatnonzero(horizontal_boundary & ~vertical_boundary)),
        move_z_only=set(np.flatnonzero(vertical_boundary & ~horizontal_boundary)),
        iterations=parameters["winslow_iterations"], omega=parameters["winslow_omega"],
    )

    parameters.update(output_filename=str(tmp_path / "smoothed.msh"), apply_winslow=True)
    meshing_2d.generate_gmsh_2d_mesh(MeshingParameters(input_mesh_dictionary=parameters))
    smoothed_mesh = meshio.read(parameters["output_filename"])

    np.testing.assert_allclose(smoothed_mesh.points[:, [1, 0]], expected, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(
        np.vstack([block.data for block in smoothed_mesh.cells if block.type == "quad"]),
        quads,
    )


@pytest.mark.parametrize("use_grid", [True, False], ids=["grid", "segy"])
@pytest.mark.parametrize(
    "with_communicator", [False, True], ids=["serial", "root"]
)
@pytest.mark.parametrize("preexisting_files", [False, True], ids=["new", "existing"])
def test_gmsh2d_writes_valid_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_communicator: bool,
    preexisting_files: bool,
    use_grid: bool,
) -> None:
    """Generate readable mesh and velocity files with the real writers.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for generated files.
    monkeypatch : pytest.MonkeyPatch
        Fixture for isolating the working directory and global ensemble.
    with_communicator : bool
        Whether to initialize a real one-process ensemble instead of ``None``.
    preexisting_files : bool
        Whether to create placeholder output files before generation.
    use_grid : bool
        Whether to convert a grid instead of using an existing SEG-Y file.
    """
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "mesh.msh"
    temporary_model = tmp_path / "tmp_velocity_model.segy"
    if preexisting_files:
        output_file.write_bytes(b"existing mesh")
        temporary_model.write_bytes(b"existing velocity model")
    parameters = _base_gmsh2d_parameters(output_file)
    parameters.update(
        structured_mesh=True, apply_winslow=False, min_element_size=1000.0
    )
    velocity_grid = (2500.0 + np.arange(120).reshape(10, 12))[:, ::2]
    if use_grid:
        parameters.update(
            segy_velocity_model=None,
            velocity_model={"vp_values": velocity_grid},
        )
    comm = fire.Ensemble(fire.COMM_SELF, 1) if with_communicator else None
    monkeypatch.setattr(meshing_2d.SpyroEnsemble, "ensemble", comm)
    mesh_parameters = MeshingParameters(input_mesh_dictionary=parameters, comm=comm)

    velocity_model_file = meshing_2d.generate_gmsh_2d_mesh(mesh_parameters)

    expected_model_file = (
        "tmp_velocity_model.segy" if use_grid else parameters["segy_velocity_model"]
    )
    assert velocity_model_file == expected_model_file
    mesh = fire.Mesh(str(output_file), comm=fire.COMM_SELF)
    assert_cell_type(mesh, "quadrilateral")
    assert_cell_count(mesh, 15)
    assert_bounds(mesh, (-LENGTH_Z, 0.0), (0.0, LENGTH_X))
    if use_grid:
        # The grid is (nz, nx), deepest row first; read back in the same layout.
        written_grid, _, _ = read_segy_velocity_model(str(temporary_model))
        np.testing.assert_allclose(written_grid, velocity_grid)
    elif preexisting_files:
        assert temporary_model.read_bytes() == b"existing velocity model"
    else:
        assert not temporary_model.exists()


@pytest.mark.parametrize("use_grid", [True, False], ids=["grid", "segy"])
def test_gmsh2d_only_root_of_mesh_comm_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_grid: bool
) -> None:
    """Write nothing on a process that is not rank 0 of `mesh_parameters.comm`.

    `SpyroEnsemble` is left uninitialized, as when `AutomaticMesh` is used on
    its own: the writer must come from the communicator that
    `AutomaticMesh._load_mesh_file` waits on, not from `SpyroEnsemble`.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory where a writer would put the mesh and temporary SEG-Y file.
    monkeypatch : pytest.MonkeyPatch
        Used to run in `tmp_path` and to clear `SpyroEnsemble.ensemble`.
    use_grid : bool
        Whether to mesh a grid `velocity_model` instead of a SEG-Y file.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(SpyroEnsemble, "ensemble", None)
    comm = MagicMock()
    comm.ensemble_comm.rank = 0
    comm.comm.rank = 1
    parameters = _base_gmsh2d_parameters(tmp_path / "mesh.msh")
    parameters.update(
        structured_mesh=True, apply_winslow=False, min_element_size=1000.0
    )
    if use_grid:
        parameters.update(
            segy_velocity_model=None,
            velocity_model={"vp_values": 2500.0 + np.arange(60.0).reshape(10, 6)},
        )
    mesh_parameters = MeshingParameters(input_mesh_dictionary=parameters, comm=comm)

    velocity_model_file = meshing_2d.generate_gmsh_2d_mesh(mesh_parameters)

    expected_model_file = (
        "tmp_velocity_model.segy" if use_grid else parameters["segy_velocity_model"]
    )
    assert velocity_model_file == expected_model_file
    assert not (tmp_path / "mesh.msh").exists()
    assert not (tmp_path / "tmp_velocity_model.segy").exists()


def _check_mesh(mesh, msh_file, water_interface, padding_type) -> None:
    """Check the geometry and regions of a 2D Gmsh mesh.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        Mesh returned by ``AutomaticMesh``.
    msh_file : pathlib.Path
        Gmsh file the mesh was read from.
    water_interface : bool
        Whether the water layer is a separate region.
    padding_type : str or None
        Padding geometry.
    """
    box_area = LENGTH_Z * LENGTH_X
    if padding_type is None:
        lower, upper = (-LENGTH_Z, 0.0), (0.0, LENGTH_X)
        total_area, area_rtol, bounds_rtol = box_area, 1e-9, 1e-9
    else:
        # The hyperellipse reaches the same extremes as the rectangle.
        lower, upper = (-LENGTH_Z - PADDING, -PADDING), (0.0, LENGTH_X + PADDING)
        if padding_type == "rectangular":
            total_area, area_rtol, bounds_rtol = (
                (LENGTH_Z + PADDING) * (LENGTH_X + 2 * PADDING), 1e-9, 1e-9,
            )
        else:
            total_area = hyperellipse_area(
                LENGTH_Z, LENGTH_X, PADDING, PADDING, HYPER_N
            )
            area_rtol, bounds_rtol = HYPERELLIPSE_AREA_RTOL, 1e-3

    assert_bounds(mesh, lower, upper, rtol=bounds_rtol)
    assert_measure(mesh, total_area, rtol=area_rtol)
    assert_no_degenerate_cells(mesh)

    groups = physical_groups(msh_file)
    core = ("WaterSurface", "SubSurface") if water_interface else ("SubSurface",)
    expected_measures = {core: box_area}
    if padding_type is not None:
        # The total was checked above; the box edges are straight, so the
        # split between model and padding is exact.
        expected_measures[("Padding",)] = measure(mesh) - box_area
    assert_cell_regions(mesh, groups[2], expected_measures)
    if water_interface:
        assert_region_above(
            mesh, groups[2]["WaterSurface"], groups[2]["SubSurface"]
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "winslow_implementation",
    [
        None,
        "default",
        "fast",
        pytest.param(
            "numba",
            marks=pytest.mark.skipif(
                not HAS_NUMBA, reason="Numba is not installed."
            ),
        ),
    ],
    ids=["no_winslow", "default", "fast", "numba"],
)
@PADDING_CASES
@WATER_INTERFACE_CASES
def test_gmsh2d_structured(
    tmp_path, water_interface, padding_type, winslow_implementation
):
    """Structured quadrilateral meshes, with and without Winslow smoothing."""
    parameters = _base_gmsh2d_parameters(tmp_path / "mesh.msh")
    parameters.update(
        {
            "padding_type": padding_type,
            "water_interface": water_interface,
            "structured_mesh": True,
            "apply_winslow": winslow_implementation is not None,
            "winslow_implementation": winslow_implementation or "default",
            "extend_segy": False,
        }
    )

    mesh = create_gmsh_mesh(parameters)

    assert_cell_type(mesh, "quadrilateral")
    assert_cell_count(mesh, STRUCTURED_EXPECTED_CELLS[(water_interface, padding_type)])
    _check_mesh(mesh, parameters["output_filename"], water_interface, padding_type)


@pytest.mark.slow
@PADDING_CASES
@WATER_INTERFACE_CASES
@pytest.mark.parametrize(
    "extend_segy", [True, False], ids=["extend_segy", "no_extend_segy"]
)
def test_gmsh2d_unstructured(
    tmp_path, extend_segy, water_interface, padding_type
):
    """Unstructured triangular meshes."""
    parameters = _base_gmsh2d_parameters(tmp_path / "mesh.msh")
    parameters.update(
        {
            "padding_type": padding_type,
            "water_interface": water_interface,
            "structured_mesh": False,
            "apply_winslow": False,
            "winslow_implementation": "default",
            "extend_segy": extend_segy,
        }
    )

    mesh = create_gmsh_mesh(parameters)

    assert_cell_type(mesh, "triangle")
    assert_cell_count(
        mesh,
        UNSTRUCTURED_EXPECTED_CELLS[(extend_segy, water_interface, padding_type)],
        rtol=UNSTRUCTURED_CELL_COUNT_RTOL,
    )
    _check_mesh(mesh, parameters["output_filename"], water_interface, padding_type)


# A model in km whose sea floor deepens from left to right, used to check the
# layout of grid velocity models: a transposed or mirrored grid gives the wrong
# water depths at the edges, or a water layer Gmsh cannot build.
WATER_MODEL_LENGTH_Z = 1.0
WATER_MODEL_LENGTH_X = 2.0
WATER_MODEL_SPACING = 0.02
WATER_MODEL_SAMPLES = round(WATER_MODEL_LENGTH_Z / WATER_MODEL_SPACING) + 1
SEA_FLOOR_LEFT = 0.2
SEA_FLOOR_RIGHT = 0.4
ROCK_INTERFACE_DEPTH = 0.6


def _water_model_velocity(depth, x, where: Callable):
    """Return the water-model velocity in km/s: water over two rock layers.

    Below the sea floor the velocity is 1.5 km/s above `ROCK_INTERFACE_DEPTH` and
    2.5 km/s below it, plus the depth in km.

    Parameters
    ----------
    depth : numpy.ndarray or ufl.core.expr.Expr
        Positive depth in km.
    x : numpy.ndarray or ufl.core.expr.Expr
        Lateral position in km.
    where : callable
        `numpy.where` or `firedrake.conditional`.

    Returns
    -------
    numpy.ndarray or ufl.core.expr.Expr
        Velocity in km/s, 0 in the water.
    """
    sea_floor = SEA_FLOOR_LEFT + (SEA_FLOOR_RIGHT - SEA_FLOOR_LEFT) * x / (
        WATER_MODEL_LENGTH_X
    )
    rock = where(depth < ROCK_INTERFACE_DEPTH, 1.5, 2.5) + depth
    return where(depth < sea_floor, 0.0, rock)


def _write_water_model_segy(filename: Path, n_traces: int) -> None:
    """Write the water model to SEG-Y with segyio only.

    Parameters
    ----------
    filename : pathlib.Path
        SEG-Y file to write.
    n_traces : int
        Number of traces, evenly spaced from `x = 0` to `x = WATER_MODEL_LENGTH_X`.
    """
    depth = np.linspace(0.0, WATER_MODEL_LENGTH_Z, WATER_MODEL_SAMPLES)[:, None]
    x = np.linspace(0.0, WATER_MODEL_LENGTH_X, n_traces)[None, :]
    vp = _water_model_velocity(depth, x, np.where)
    write_segy_traces(filename, np.broadcast_to(vp, (WATER_MODEL_SAMPLES, n_traces)))


def _water_model_conditional_grid(orientation: str) -> dict:
    """Sample the water model with `scalar_conditional_to_grid`.

    Parameters
    ----------
    orientation : str
        `orientation` passed to `scalar_conditional_to_grid`.

    Returns
    -------
    dict
        Grid velocity dictionary with `grid_spacing`.
    """
    mesh_parameters = MeshingParameters(
        input_mesh_dictionary={
            "length_z": WATER_MODEL_LENGTH_Z,
            "length_x": WATER_MODEL_LENGTH_X,
            "length_y": 0.0,
            "mesh_type": "firedrake_mesh",
            "edge_length": WATER_MODEL_SPACING,
            "dimension": 2,
        }
    )
    mesh = AutomaticMesh(mesh_parameters=mesh_parameters).create_mesh()
    z, x = fire.SpatialCoordinate(mesh)
    return scalar_conditional_to_grid(
        conditional=_water_model_velocity(-z, x, fire.conditional),
        domain_dimensions=(WATER_MODEL_LENGTH_Z, WATER_MODEL_LENGTH_X),
        grid_spacing=WATER_MODEL_SPACING,
        orientation=orientation,
    )


def _water_model_parameters(
    velocity_key: str, velocity_value, output_filename: Path
) -> dict:
    """Return the input of an unstructured Gmsh mesh of the water model.

    Parameters
    ----------
    velocity_key : str
        `"segy_velocity_model"` or `"velocity_model"`.
    velocity_value : str or dict
        SEG-Y path or grid velocity dictionary.
    output_filename : pathlib.Path
        Mesh file to write.

    Returns
    -------
    dict
        Input mesh dictionary for `MeshingParameters`.
    """
    return {
        "mesh_type": "gmsh_mesh",
        "dimension": 2,
        "length_z": WATER_MODEL_LENGTH_Z,
        "length_x": WATER_MODEL_LENGTH_X,
        "output_filename": str(output_filename),
        velocity_key: velocity_value,
        "cells_per_wavelength": 2.0,
        "source_frequency": 5.0,
        "padding_type": None,
        "padding_x": 0.0,
        "padding_z": 0.0,
        "hyper_n": 4.0,
        "hmin_segy": 0.0,
        "grade": None,
        "water_interface": True,
        "water_search_value": 0.0,
        "vp_water": 1.5,
        "structured_mesh": False,
        "apply_winslow": False,
        "extend_segy": False,
        "min_element_size": 0.1,
        "h_padding": 0.5,
    }


@pytest.mark.parametrize(
    "source, n_traces",
    [
        ("segy_file", 101),
        ("segy_dictionary", 101),
        ("segy_dictionary", WATER_MODEL_SAMPLES),
        ("conditional", None),
    ],
    ids=["segy_file", "segy_dictionary", "square_segy_dictionary", "conditional"],
)
def test_gmsh2d_grid_velocity_model_meshes_like_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, n_traces: int | None
) -> None:
    """Mesh the water model from each velocity input and check the water depths.

    The `segy_file` case meshes the SEG-Y directly and checks the test itself. The
    square case has `nz == nx`, where a transposed grid used to give no error.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the model, the mesh and the mesher's temporary SEG-Y file.
    monkeypatch : pytest.MonkeyPatch
        Used to run the mesher in `tmp_path`.
    source : str
        `"segy_file"`, `"segy_dictionary"` (grid from
        `create_grid_dictionary_from_segy`) or `"conditional"` (grid from
        `scalar_conditional_to_grid` with its default orientation).
    n_traces : int or None
        Number of SEG-Y traces; None for the conditional grid.
    """
    monkeypatch.chdir(tmp_path)
    if source == "conditional":
        velocity_key = "velocity_model"
        velocity_value = _water_model_conditional_grid(orientation="zx")
    else:
        segy_file = tmp_path / "model.segy"
        _write_water_model_segy(segy_file, n_traces)
        if source == "segy_file":
            velocity_key, velocity_value = "segy_velocity_model", str(segy_file)
        else:
            velocity_key = "velocity_model"
            velocity_value = create_grid_dictionary_from_segy(
                str(segy_file), WATER_MODEL_LENGTH_Z, WATER_MODEL_LENGTH_X
            )
            assert velocity_value["vp_values"].shape == (WATER_MODEL_SAMPLES, n_traces)

    msh_file = tmp_path / "model.msh"
    create_gmsh_mesh(_water_model_parameters(velocity_key, velocity_value, msh_file))

    left, right = water_depths_at_edges(msh_file, WATER_MODEL_LENGTH_X)
    tolerance = 1.5 * WATER_MODEL_SPACING
    assert abs(left - SEA_FLOOR_LEFT) <= tolerance, f"water depth {left} at x = 0"
    assert abs(right - SEA_FLOOR_RIGHT) <= tolerance, f"water depth {right} at x = L"


@pytest.mark.parametrize("source", ["segy_dictionary", "conditional"])
def test_gmsh2d_grid_velocity_model_rejects_nx_nz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """Reject a non-square grid laid out as `(nx, nz)`, before any meshing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the model and any file the mesher might write.
    monkeypatch : pytest.MonkeyPatch
        Used to run the mesher in `tmp_path`.
    source : str
        `"segy_dictionary"` (a transposed `create_grid_dictionary_from_segy`
        grid, with `grid_spacing_z` and `grid_spacing_x`) or `"conditional"`
        (`scalar_conditional_to_grid` with `orientation="xz"`, with
        `grid_spacing`).
    """
    monkeypatch.chdir(tmp_path)
    if source == "conditional":
        grid = _water_model_conditional_grid(orientation="xz")
    else:
        segy_file = tmp_path / "model.segy"
        _write_water_model_segy(segy_file, 101)
        grid = create_grid_dictionary_from_segy(
            str(segy_file), WATER_MODEL_LENGTH_Z, WATER_MODEL_LENGTH_X
        )
        grid["vp_values"] = grid["vp_values"].T
    msh_file = tmp_path / "model.msh"
    parameters = _water_model_parameters("velocity_model", grid, msh_file)

    with pytest.raises(ValueError, match=r"expects \(nz, nx\)"):
        meshing_2d.generate_gmsh_2d_mesh(
            MeshingParameters(input_mesh_dictionary=parameters)
        )
    assert not msh_file.exists()
    assert not (tmp_path / "tmp_velocity_model.segy").exists()
