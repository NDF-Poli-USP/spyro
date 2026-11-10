"""Elastic wave propagation with Clayton-Engquist ABCs on a 2D Gmsh mesh.

A Ricker source in the middle of a homogeneous 1 x 1 km domain radiates
outwards. Absorption is measured as the final mechanical energy divided by
its peak. The same simulation is run on:

* a Firedrake ``RectangleMesh`` with CE_A1, the reference (ratio ~2e-4);
* the Gmsh mesh with CE_A1, which must absorb like the reference;
* the Gmsh mesh with no ABC, which must keep most of its energy (~0.9), so
  that a low ratio really comes from the boundaries.

If the Gmsh boundary ids differ from the ``RectangleMesh`` ones, the ABC terms
are applied with the wrong normals and the Gmsh ratio rises to ~6e-2.
Lengths are in km, velocities in km/s and density in g/cm^3.
"""

import firedrake as fire
import numpy as np
import pytest
import segyio

import spyro
from spyro.meshing import AutomaticMesh, MeshingParameters


LENGTH = 1.0
VP = 3.0
VS = 1.732
DENSITY = 2.7
FREQUENCY = 5.0
ELEMENT_SIZE = 0.1
GRID_SPACING = 0.05
# The Gmsh ratio may exceed the reference by at most this factor.
REFERENCE_FACTOR = 3.0
NO_ABC_MIN_RATIO = 0.5


def _gmsh_mesh(tmp_path) -> fire.MeshGeometry:
    """Mesh a homogeneous SEG-Y model with Gmsh.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the SEG-Y and mesh files.

    Returns
    -------
    firedrake.MeshGeometry
        Triangular mesh of ``[-LENGTH, 0] x [0, LENGTH]`` in ``(z, x)``.
    """
    n = int(round(LENGTH / GRID_SPACING)) + 1
    segy_file = tmp_path / "homogeneous.segy"
    spec = segyio.spec()
    spec.format = 1
    spec.samples = range(n)
    spec.tracecount = n
    with segyio.create(str(segy_file), spec) as f:
        for i in range(n):
            f.trace[i] = np.full(n, VP, dtype=np.float32)

    parameters = {
        "mesh_type": "gmsh_mesh",
        "dimension": 2,
        "length_z": LENGTH,
        "length_x": LENGTH,
        "output_filename": str(tmp_path / "homogeneous.msh"),
        "segy_velocity_model": str(segy_file),
        "cells_per_wavelength": VP / (FREQUENCY * ELEMENT_SIZE),
        "source_frequency": FREQUENCY,
        "grade": 0.0,
        "hmin_segy": 0.0,
        "padding_type": None,
        "water_interface": False,
        "structured_mesh": False,
        "apply_winslow": False,
    }
    mesh_parameters = MeshingParameters(input_mesh_dictionary=parameters)
    return AutomaticMesh(mesh_parameters=mesh_parameters).create_mesh()


def _rectangle_mesh() -> fire.MeshGeometry:
    """Build the reference Firedrake mesh with the same size and element size.

    Returns
    -------
    firedrake.MeshGeometry
        Triangular mesh of ``[-LENGTH, 0] x [0, LENGTH]`` in ``(z, x)``.
    """
    n = int(round(LENGTH / ELEMENT_SIZE))
    return fire.RectangleMesh(n, n, 0.0, LENGTH, originX=-LENGTH)


def _energy_ratio(mesh: fire.MeshGeometry, energy_file, abc: bool) -> float:
    """Run the elastic simulation and return final over peak energy.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        Mesh of ``[-LENGTH, 0] x [0, LENGTH]`` in ``(z, x)``.
    energy_file : pathlib.Path
        Where spyro saves the mechanical energy history.
    abc : bool
        Whether to apply CE_A1 absorbing boundary conditions.

    Returns
    -------
    float
        Final mechanical energy divided by its peak value.
    """
    dictionary = {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 3,
            "dimension": 2,
        },
        "parallelism": {"type": "automatic"},
        "mesh": {"user_mesh": mesh, "length_z": LENGTH, "length_x": LENGTH},
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [[-LENGTH / 2, LENGTH / 2]],
            "frequency": FREQUENCY,
            "delay": 1.5,
            "delay_type": "multiples_of_minimum",
            "amplitude": np.array([0.0, 1.0]),
            "receiver_locations": [],
        },
        "synthetic_data": {
            "type": "object",
            "density": DENSITY,
            "p_wave_velocity": VP,
            "s_wave_velocity": VS,
            "real_velocity_file": None,
        },
        "time_axis": {
            "initial_time": 0,
            "final_time": 1.2,
            "dt": 5e-4,
            "output_frequency": 20,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "time": False,
            "mechanical_energy": True,
            "mechanical_energy_filename": str(energy_file),
        },
    }
    if abc:
        dictionary["absorving_boundary_conditions"] = {
            "status": True,
            "abc_type": "nrbc",
            "nrbc": {"type": "CE_A1", "dt_scheme": "backward"},
        }

    wave = spyro.IsotropicWave(dictionary)
    wave.set_mesh(user_mesh=mesh, input_mesh_parameters={})
    wave.forward_solve()

    energy = np.load(energy_file)
    return energy[-1] / energy.max()


@pytest.fixture(scope="module")
def module_tmp_path(tmp_path_factory):
    """Provide one temporary directory shared by the tests in this module.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Pytest factory for temporary directories.

    Returns
    -------
    pathlib.Path
        Temporary directory.
    """
    return tmp_path_factory.mktemp("gmsh_clayton_engquist")


@pytest.fixture(scope="module")
def gmsh_mesh(module_tmp_path) -> fire.MeshGeometry:
    """Build the Gmsh mesh once for all tests in this module.

    Parameters
    ----------
    module_tmp_path : pathlib.Path
        Shared temporary directory.

    Returns
    -------
    firedrake.MeshGeometry
        The Gmsh mesh.
    """
    return _gmsh_mesh(module_tmp_path)


def test_gmsh_2d_clayton_engquist_absorbs_like_rectangle_mesh(
    gmsh_mesh, module_tmp_path
) -> None:
    """Check that CE_A1 absorbs as well on the Gmsh mesh as on a RectangleMesh.

    Parameters
    ----------
    gmsh_mesh : firedrake.MeshGeometry
        Mesh built by the Gmsh mesher.
    module_tmp_path : pathlib.Path
        Shared temporary directory.
    """
    reference_ratio = _energy_ratio(
        _rectangle_mesh(), module_tmp_path / "energy_rectangle.npy", abc=True
    )
    gmsh_ratio = _energy_ratio(
        gmsh_mesh, module_tmp_path / "energy_gmsh.npy", abc=True
    )
    assert gmsh_ratio < REFERENCE_FACTOR * reference_ratio, (
        f"Final/peak energy is {gmsh_ratio:.2e} on the Gmsh mesh but "
        f"{reference_ratio:.2e} on a RectangleMesh. The Clayton-Engquist "
        "boundaries are not absorbing correctly on the Gmsh mesh; check that "
        "its boundary ids match RectangleMesh (1 z=-Lz, 2 z=0, 3 x=0, 4 x=Lx)."
    )


def test_gmsh_2d_without_abc_keeps_energy(gmsh_mesh, module_tmp_path) -> None:
    """Check that, without ABCs, the Gmsh domain keeps most of its energy.

    This makes sure the absorption measured above comes from the boundaries.

    Parameters
    ----------
    gmsh_mesh : firedrake.MeshGeometry
        Mesh built by the Gmsh mesher.
    module_tmp_path : pathlib.Path
        Shared temporary directory.
    """
    ratio = _energy_ratio(
        gmsh_mesh, module_tmp_path / "energy_gmsh_no_abc.npy", abc=False
    )
    assert ratio > NO_ABC_MIN_RATIO, (
        f"Final/peak energy without ABCs is {ratio:.2e}; expected above "
        f"{NO_ABC_MIN_RATIO}. Energy is lost for reasons other than the "
        "absorbing boundaries."
    )
