import shutil
import tempfile

import numpy as np
import pytest
from firedrake import conditional
from mpi4py.MPI import COMM_WORLD

import spyro


def _build_wave(parallelism: str, output_dir: str) -> spyro.AcousticWave:
    """Build a small two-shot acoustic solver that logs its energy and time.

    The velocity model has two layers with one source in each, so that the two shots
    log different energies.

    Parameters
    ----------
    parallelism : str
        Parallelism type, `automatic` for one shot per ensemble member or `spatial`
        for both shots in serial on a distributed mesh.
    output_dir : str
        Directory, shared by all ranks, where the energy and time are saved.

    Returns
    -------
    spyro.AcousticWave
        Solver with its mesh and velocity model set.
    """
    dictionary = {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 2,
            "dimension": 2,
        },
        "parallelism": {"type": parallelism},
        "mesh": {
            "length_z": 1.0,
            "length_x": 1.0,
            "length_y": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(-0.3, 0.3), (-0.6, 0.7)],
            "frequency": 5.0,
            "delay": 1.5,
            "delay_type": "multiples_of_minimum",
            "receiver_locations": spyro.create_transect((-0.8, 0.2), (-0.8, 0.8), 3),
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.1,
            "dt": 0.001,
            "output_frequency": 10,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "acoustic_energy": True,
            "acoustic_energy_filename": f"{output_dir}/energy.npy",
            "time_filename": f"{output_dir}/time.npy",
        },
    }
    wave = spyro.AcousticWave(dictionary=dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    wave.set_initial_velocity_model(
        conditional=conditional(wave.mesh_z < -0.5, 3.0, 1.5)
    )
    return wave


@pytest.mark.parametrize("parallelism", ["automatic", "spatial"])
def test_energy_logging_in_parallel(parallelism):
    # With `spatial`, the energy assembly is collective over 2 ranks: logging it on
    # one rank only used to hang. With `automatic`, each shot has its own ensemble
    # member: the energy of the second shot used to be lost.
    output_dir = COMM_WORLD.bcast(
        tempfile.mkdtemp() if COMM_WORLD.rank == 0 else None, root=0
    )
    wave = _build_wave(parallelism, output_dir)
    wave.forward_solve()
    COMM_WORLD.barrier()

    time = np.load(f"{output_dir}/time.npy")
    energies = [np.load(f"{output_dir}/energysn[{shot}].npy") for shot in (0, 1)]
    for energy in energies:
        assert energy.shape == time.shape
        assert energy[-1] > 0.0
    assert not np.allclose(energies[0], energies[1], rtol=1e-3, atol=0.0)

    # The last propagated shot of this rank
    shot = wave.current_sources[0]
    assert wave.simulation_logger.get("acoustic_energy") == energies[shot][-1]

    spyro.io.delete_tmp_files(wave)
    COMM_WORLD.barrier()
    if COMM_WORLD.rank == 0:
        shutil.rmtree(output_dir)
