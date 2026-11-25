"""Velocity at the nodes shared by cells of different velocities.

A velocity model defined cell by cell (DG0) is converted to the continuous space
of the problem twice: in the preamble (``creating_velocity_profile``) and after
the extension of the infinite model (``velocity_abc``). A node shared by cells of
different velocities takes the minimum of them, the rule of ``Minimum_Eikonal``
for the velocity at the critical point, so that the result does not depend on
the order in which the cells are visited, and therefore on the mesh partition.
The helpers of this module are also used by
``tests/parallel/test_reference_model_parallel.py``.
"""

import firedrake as fire
import numpy as np
import pytest

from spyro.mpi.spyro_mpi import SpyroEnsemble
from spyro.solvers.acoustic_wave import AcousticWave
from tests.on_one_core import test_eikonal_new as ten
from tests.on_one_core import test_reference_model_abc as trm

# Integral of the velocity in km**3/s (km/s times km**2) of the 2D
# reference-model waves in serial, after the preamble ("original", unit square)
# and after the extension of the infinite model ("extended", 7 km x 4 km). The
# integrals use the quadrature of the solver, a weighted sum of the nodal values,
# so that they do not depend on the default quadrature of Firedrake, which differs
# between versions (2025.4.1 and 2026.10.0 give the same values with it).
SERIAL_INTEGRAL = {
    "original": {"T": 2.2336309523809526, "Q": 2.2312499999999953},
    "extended": {"T": 62.93452380952376, "Q": 62.92500000000008},
}


def cell_minimum_violations(wave) -> int:
    """Count the nodes whose velocity is not the minimum of their cells.

    Parameters
    ----------
    wave : `AcousticWave`
        Wave after the preamble: ``wave.initial_velocity_model`` is the DG0
        model and ``wave.c`` its conversion to ``wave.function_space``.

    Returns
    -------
    `int`
        Number of nodes, over every rank, whose velocity differs from the
        minimum of the velocities of the cells that contain them.
    """
    model = wave.initial_velocity_model
    cell_dofs = model.function_space().cell_node_map().values_with_halo[:, 0]
    cell_nodes = wave.function_space.cell_node_map().values_with_halo
    expected = np.full(wave.c.dat.data_ro_with_halos.shape, np.inf)
    np.minimum.at(
        expected,
        cell_nodes.ravel(),
        np.repeat(model.dat.data_ro_with_halos[cell_dofs], cell_nodes.shape[1]),
    )
    owned = wave.function_space.dof_dset.size
    wrong = ~np.isclose(wave.c.dat.data_ro[:owned], expected[:owned])
    return SpyroEnsemble.ensemble.comm.allreduce(int(wrong.sum()))


def velocity_on_jump(wave) -> tuple[int, int, int]:
    """Count the nodes on x = 0.5 with each velocity over every rank.

    Parameters
    ----------
    wave : `AcousticWave`
        Wave whose velocity ``wave.c`` is in ``wave.function_space``.

    Returns
    -------
    n_fast : `int`
        Number of nodes on x = 0.5 with 3.0 km/s.
    n_slow : `int`
        Number of nodes on x = 0.5 with 1.5 km/s.
    n_other : `int`
        Number of nodes on x = 0.5 with any other velocity.
    """
    V = wave.function_space
    coords = fire.Function(fire.VectorFunctionSpace(wave.mesh, V.ufl_element()))
    coords.interpolate(fire.SpatialCoordinate(wave.mesh))
    owned = V.dof_dset.size
    on_jump = np.isclose(coords.dat.data_ro[:owned, 1], 0.5)
    values = wave.c.dat.data_ro[:owned][on_jump]
    fast = np.isclose(values, 3.0)
    slow = np.isclose(values, 1.5)
    comm = SpyroEnsemble.ensemble.comm
    return (
        comm.allreduce(int(fast.sum())),
        comm.allreduce(int(slow.sum())),
        comm.allreduce(int((~fast & ~slow).sum())),
    )


def velocity_integral(wave) -> float:
    """Integrate the velocity with the quadrature of the solver.

    Parameters
    ----------
    wave : `AcousticWave`
        Wave whose velocity ``wave.c`` is in ``wave.function_space``.

    Returns
    -------
    `float`
        Integral of the velocity over the mesh, with ``wave.quadrature_rule``.
    """
    return fire.assemble(wave.c * fire.dx(**wave.quadrature_rule))


def velocity_profile(element_geometry: str) -> dict:
    """Build the velocity of the 2D reference model before and after the extension.

    Parameters
    ----------
    element_geometry : `str`
        Geometry of the finite element, "T" or "Q".

    Returns
    -------
    `dict`
        For "original" (after the preamble) and "extended" (after
        ``velocity_abc`` of the infinite model): the node counts of
        `velocity_on_jump` under "jump" and the integral of the velocity under
        "integral"; for "original" also the result of `cell_minimum_violations`
        under "violations".
    """
    wave = trm.wave_instance(element_geometry, 2, "hybrid", False, "ARNOLDI")
    profile = {
        "original": {
            "violations": cell_minimum_violations(wave),
            "jump": velocity_on_jump(wave),
            "integral": velocity_integral(wave),
        }
    }

    layer = wave.layer_ops
    layer.geometry_infinite_model(wave)
    layer.create_mesh_with_layer(wave, inf_model=True)
    layer.velocity_abc(wave, inf_model=True)
    profile["extended"] = {
        "jump": velocity_on_jump(wave),
        "integral": velocity_integral(wave),
    }
    return profile


def check_velocity_profile(profile: dict, element_geometry: str) -> None:
    """Check the cell minimum at the nodes and the serial integrals.

    Parameters
    ----------
    profile : `dict`
        Result of `velocity_profile`.
    element_geometry : `str`
        Geometry of the finite element, "T" or "Q".

    Returns
    -------
    None
    """
    assert profile["original"]["violations"] == 0
    for stage in ("original", "extended"):
        n_fast, n_slow, n_other = profile[stage]["jump"]
        SpyroEnsemble.print(
            f"{element_geometry} {stage}: nodes on x = 0.5 with 3.0 km/s: {n_fast}, "
            f"1.5 km/s: {n_slow}, other: {n_other}; integral of c: "
            f"{profile[stage]['integral']:.12f}"
        )
        assert n_fast == 0 and n_other == 0 and n_slow > 0
        assert np.isclose(
            profile[stage]["integral"],
            SERIAL_INTEGRAL[stage][element_geometry],
            rtol=1e-12,
        )


@pytest.mark.parametrize("element_geometry", ["T", "Q"])
def test_velocity_on_jump_is_minimum(element_geometry: str) -> None:
    """Check the velocity of the reference model before and after the extension.

    Parameters
    ----------
    element_geometry : `str`
        Geometry of the finite element, "T" or "Q".

    Returns
    -------
    None
    """
    check_velocity_profile(velocity_profile(element_geometry), element_geometry)


def test_velocity_is_cell_minimum_with_straddling_cells() -> None:
    """Check the cell minimum where cells straddle the velocity jump.

    The mesh of ``test_eikonal_new`` T-2D underintegrated with edge 0.2 km
    (KMV degree 4) has cells on both sides of x = 0.5 and nodes on it, so
    many nodes are shared by cells of both velocities.

    Returns
    -------
    None
    """
    dictionary = ten.wave_dict("T", 2, 3, "underintegrated", "nearest_node")
    wave = AcousticWave(dictionary=dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.2})
    wave.set_initial_velocity_model(
        conditional=fire.conditional(wave.mesh_x < 0.5, 3.0, 1.5)
    )
    wave.mesh_ops.preamble_mesh_operations(
        wave, ele_type_eik="underintegrated", f_est=0.07, source_bc_eik="nearest_node"
    )

    assert cell_minimum_violations(wave) == 0
