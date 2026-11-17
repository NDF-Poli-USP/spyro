"""Homogeneous SeismicMesh mesh generated over two ranks.

Run with ``mpiexec -n 2 pytest tests/parallel/test_seismicmesh_homogeneous_parallel.py``.
SeismicMesh gathers a parallel mesh onto rank 0 and returns placeholders on
the other ranks, which must neither post-process nor write it. The two ranks
are split in every way the mesh wrapper distinguishes: without an ensemble,
as one ensemble member spanning both ranks, and as two ensemble members with
one rank each, where only the first member generates the mesh file.
"""

import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
from firedrake import COMM_WORLD, Function, FunctionSpace, SpatialCoordinate

from spyro.mpi.spyro_mpi import (
    AutomaticParallelism,
    ParallelismConfig,
    SpatialParallelism,
    SpyroEnsemble,
)
from spyro.meshing.seismic_mesh_based_wrappers import (
    SeismicMesh,
    create_seismicmesh_2D_mesh_homogeneous,
)


@pytest.mark.skipif(SeismicMesh is None, reason="SeismicMesh is not installed")
@pytest.mark.parallel(2)
@pytest.mark.parametrize(
    "parallelism, spatial_size",
    [
        (None, 2),
        (SpatialParallelism(number_of_sources=1), 2),
        (AutomaticParallelism(number_of_sources=2), 1),
    ],
    ids=["no_ensemble", "one_ensemble_member", "two_ensemble_members"],
)
def test_seismicmesh_homogeneous_mesh_in_parallel(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    parallelism: ParallelismConfig | None,
    spatial_size: int,
) -> None:
    """The mesh spans the padded domain on every rank of the spatial comm.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Per-rank pytest temporary directory; rank 0's is shared.
    monkeypatch : pytest.MonkeyPatch
        Restores the class-level `SpyroEnsemble` state after the test, so
        that one case's ensemble does not leak into the next.
    parallelism : ParallelismConfig or None
        Configuration `SpyroEnsemble` is initialized with, or None to leave
        it uninitialized.
    spatial_size : int
        Expected size of the communicator the mesh is distributed over.
    """
    assert COMM_WORLD.size == 2
    length_z, length_x, pad = 1.0, 2.0, 0.3
    monkeypatch.setattr(SpyroEnsemble, "ensemble", None)
    monkeypatch.setattr(SpyroEnsemble, "config", None)
    if parallelism is not None:
        SpyroEnsemble.initialize(parallelism)
    output_file_name = COMM_WORLD.bcast(str(tmp_path / "homogeneous.msh"), root=0)
    mesh_parameters = SimpleNamespace(
        length_z=length_z,
        length_x=length_x,
        abc_pad=pad,
        edge_length=0.1,
        output_file_name=output_file_name,
    )

    mesh = create_seismicmesh_2D_mesh_homogeneous(mesh_parameters)

    assert mesh.comm.size == spatial_size
    V = FunctionSpace(mesh, "CG", 1)
    z, x = SpatialCoordinate(mesh)
    uz = Function(V).interpolate(z)
    ux = Function(V).interpolate(x)
    with uz.dat.vec_ro as vz, ux.dat.vec_ro as vx:
        assert np.isclose(vz.min()[1], -length_z - pad)
        assert np.isclose(vz.max()[1], 0.0)
        assert np.isclose(vx.min()[1], -pad)
        assert np.isclose(vx.max()[1], length_x + pad)
