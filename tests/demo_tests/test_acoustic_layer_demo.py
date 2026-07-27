import firedrake as fire
import numpy as np
import spyro
import pytest


# creating dummy model for comm generation
class DummyModel():
    def __init__(self):
        self.parallelism_type = "automatic"
        self.number_of_sources = 6


@pytest.mark.parallel(8)
def test_acoustic_camembert_fwi():
    from demos.acoustic_layers_fwi import setting_up_fwi,  run_fwi

    setting_up_fwi()
    run_fwi()
    length_z = 2.0
    length_x = 2.0
    grid_vp_data = spyro.io.segy_io.create_grid_dictionary_from_segy(
            "layers.segy",
            length_z,
            length_x,
        )

    model = DummyModel()
    comm = spyro.utils.mpi_init(model)
    mesh = fire.RectangleMesh(200, 200, length_z, length_x, comm=comm.comm)
    mesh.coordinates.dat.data[:, 0] *= -1.0
    V = fire.FunctionSpace(mesh, "CG", 1)
    u = spyro.io.project_grid_velocity_data(grid_vp_data, V, comm=comm)

    switch_vp = float(u.at((-1.05, 1.0)))
    before_switch_vp = float(u.at((-0.7, 1.7)))

    assert switch_vp == pytest.approx(3.0)
    assert before_switch_vp == pytest.approx(2.5)
