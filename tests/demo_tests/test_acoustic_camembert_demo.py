import firedrake as fire
import numpy as np
import spyro
import pytest


# creating dummy model for comm generation
class DummyModel():
    def __init__(self):
        self.parallelism_type = "automatic"
        self.number_of_sources = 6


@pytest.mark.parallel(6)
def test_acoustic_camembert_fwi():
    from demos.acoustic_camembert_fwi import run_fwi
    model = DummyModel()
    comm = spyro.utils.mpi_init(model)

    run_fwi(load_real_shot=False)

    comm.global_comm.Barrier()
    length_z = 2.0
    length_x = 2.0
    grid_vp_data = spyro.io.segy_io.create_grid_dictionary_from_segy(
            "camembert.segy",
            length_z,
            length_x,
        )

    mesh = fire.RectangleMesh(200, 200, length_z, length_x, comm=comm.comm)
    mesh.coordinates.dat.data[:, 0] *= -1.0
    V = fire.FunctionSpace(mesh, "CG", 1)
    u = spyro.io.project_grid_velocity_data(grid_vp_data, V, comm=comm)

    camembert_vp = float(u.at((-1.0, 1.0)))
    outside_camembert_vp = float(u.at((-1.7, 1.7)))

    assert camembert_vp == pytest.approx(3.0, rel=1e-2)
    assert outside_camembert_vp == pytest.approx(2.5, rel=1e-2)
