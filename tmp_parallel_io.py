import spyro
# import debugpy
# from mpi4py.MPI import COMM_WORLD
# debugpy.listen(3000 + COMM_WORLD.rank)
# debugpy.wait_for_client()


def test_saving_and_loading_supershot_record():
    from test.inputfiles.model import dictionary

    dictionary["parallelism"]["type"] = "custom"
    dictionary["parallelism"]["shot_ids_per_propagation"] = [[0, 1]]
    dictionary["time_axis"]["final_time"] = 0.5
    dictionary["acquisition"]["source_locations"] = [(-0.5, 0.4), (-0.5, 0.6)]
    dictionary["acquisition"]["receiver_locations"] = spyro.create_transect((-0.55, 0.1), (-0.55, 0.9), 200)

    wave = spyro.AcousticWave(dictionary=dictionary)
    wave.set_mesh(mesh_parameters={"dx": 0.02})
    wave.set_initial_velocity_model(constant=1.5)
    wave.forward_solve()
    spyro.io.save_shots(wave, filename="test_shot_record")
    shots1 = wave.forward_solution_receivers

    wave2 = spyro.AcousticWave(dictionary=dictionary)
    wave2.set_mesh(mesh_parameters={"dx": 0.02})
    spyro.io.load_shots(wave2, filename="test_shot_record")
    shots2 = wave.forward_solution_receivers

    test_pass = (shots1 == shots2).all()
    print(f"Passes supershot io test:{test_pass}", flush=True)

    assert test_pass


def test_saving_and_loading_shot_records_in_ensemble_serial_with_spatial_parallelism():
    from test.inputfiles.model import dictionary

    dictionary["parallelism"]["type"] = "spatial"
    dictionary["time_axis"]["final_time"] = 0.5
    dictionary["acquisition"]["source_locations"] = [(-0.5, 0.4), (-0.5, 0.6)]
    dictionary["acquisition"]["receiver_locations"] = spyro.create_transect((-0.55, 0.1), (-0.55, 0.9), 200)

    wave = spyro.AcousticWave(dictionary=dictionary)
    wave.set_mesh(mesh_parameters={"dx": 0.02})
    wave.set_initial_velocity_model(constant=1.5)
    wave.forward_solve()
    spyro.io.save_shots(wave, filename="test_shot_record")
    shots1 = wave.forward_solution_receivers

    wave2 = spyro.AcousticWave(dictionary=dictionary)
    wave2.set_mesh(mesh_parameters={"dx": 0.02})
    spyro.io.load_shots(wave2, filename="test_shot_record")
    shots2 = wave.forward_solution_receivers

    test_pass = (shots1 == shots2).all()
    print(f"Passes shot in ensemble serial io test:{test_pass}", flush=True)

    assert test_pass


if __name__ == "__main__":
    # test_saving_and_loading_supershot_record()
    test_saving_and_loading_shot_records_in_ensemble_serial_with_spatial_parallelism()
