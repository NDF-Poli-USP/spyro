import os

import firedrake as fire

import spyro
from spyro.mpi.spyro_mpi import AutomaticParallelism, SpyroEnsemble

# from mpi4py.MPI import COMM_WORLD
# debugpy.listen(3000 + COMM_WORLD.rank)
# debugpy.wait_for_client()


def test_saving_and_loading_shot_record():
    from ..inputfiles.model import dictionary

    dictionary["parallelism"]["type"] = "custom"
    dictionary["parallelism"]["shot_ids_per_propagation"] = [[0, 1]]
    dictionary["time_axis"]["final_time"] = 0.5
    dictionary["acquisition"]["source_locations"] = [(-0.5, 0.4), (-0.5, 0.6)]
    dictionary["acquisition"]["receiver_locations"] = spyro.create_transect(
        (-0.55, 0.1), (-0.55, 0.9), 200
    )

    wave = spyro.AcousticWave(dictionary=dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.02})
    wave.set_initial_velocity_model(constant=1.5)
    wave.forward_solve()
    spyro.io.save_shots(wave, file_name="test_shot_record")
    shots1 = wave.forward_solution_receivers

    wave2 = spyro.AcousticWave(dictionary=dictionary)
    wave2.set_mesh(input_mesh_parameters={"edge_length": 0.02})
    spyro.io.load_shots(wave2, file_name="test_shot_record")
    shots2 = wave.forward_solution_receivers

    assert (shots1 == shots2).all()


def test_write_vtk_from_every_ensemble_member():
    """Writing replicated fields from all ensemble members must not race.

    Each of the two ensemble members has a one-rank spatial communicator, so
    a ``VTKFile`` built on it makes every member try to create the same
    output directory and files.
    """
    SpyroEnsemble.initialize(
        AutomaticParallelism(number_of_sources=fire.COMM_WORLD.size)
    )
    mesh = fire.UnitSquareMesh(4, 4, comm=SpyroEnsemble.ensemble.comm)
    u = fire.Function(fire.FunctionSpace(mesh, "CG", 1)).assign(1.0)

    for i in range(20):
        SpyroEnsemble.write_vtk(f"results/write_vtk/control_{i}.pvd", u, name="u")
    fire.COMM_WORLD.barrier()

    assert os.path.isfile("results/write_vtk/control_19.pvd")


if __name__ == "__main__":
    test_saving_and_loading_shot_record()
