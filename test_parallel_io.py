import firedrake as fire
import math
import pytest
import spyro


def test_saving_shot_record():
    from test.inputfiles.model import dictionary

    dictionary["parallelism"]["type"] = "custom"
    dictionary["parallelism"]["shot_ids_per_propagation"] = [[0, 1]]
    dictionary["time_axis"]["final_time"] = 0.5
    dictionary["acquisition"]["source_locations"] = [(-0.5, 0.4), (-0.5, 0.6)]
    dictionary["acquisition"]["receiver_locations"] = spyro.create_transect((-0.55, 0.1), (-0.55, 0.9), 200)

    Wave_obj = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj.set_mesh(mesh_parameters={"dx": 0.02})
    Wave_obj.set_initial_velocity_model(constant=1.5)
    Wave_obj.forward_solve()
    spyro.io.save_shots(Wave_obj, file_name="test_shot_record")


def test_loading_shot_record():
    from test.inputfiles.model import dictionary

    dictionary["parallelism"]["type"] = "custom"
    dictionary["parallelism"]["shot_ids_per_propagation"] = [[0, 1]]
    dictionary["time_axis"]["final_time"] = 0.5
    dictionary["acquisition"]["source_locations"] = [(-0.5, 0.4), (-0.5, 0.6)]
    dictionary["acquisition"]["receiver_locations"] = spyro.create_transect((-0.55, 0.1), (-0.55, 0.9), 200)

    Wave_obj = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj.set_mesh(mesh_parameters={"dx": 0.02})
    spyro.io.load_shots(Wave_obj, file_name="test_shot_record")


if __name__ == "__main__":
    test_saving_shot_record()
    test_loading_shot_record()
