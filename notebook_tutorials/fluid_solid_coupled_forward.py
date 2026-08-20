import spyro
from spyro.solvers.acoustic_elastic_wave import AcousticElasticWave

dictionary = {}

dictionary["options"] = {
    "cell_type": "T",
    "variant": "lumped",
    "degree": 2,
    "dimension": 2,
}

dictionary["parallelism"] = {
    "type": "automatic",
}

dictionary["mesh"] = {
    "length_z": 1.0,
    "length_x": 1.0,
    "length_y": 0.0,
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
    "edge_lenght": 0.005,
    "interface_x": 0.5,
    "absorb_left": False,
    "absorb_right": False,
    "absorb_top": False,
    "absorb_bottom": False,
}

dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.5, 0.6)],
    "frequency": 25.0,
    "delay": 1.0/25.0,
    "delay_type": "time",
    "receiver_locations": [(-0.51, 0.51)],
    "user_vertex_only_mesh": True,
}

dictionary["time_axis"] = {
    "initial_time": 0.0,
    "final_time": 0.22,
    "dt": 0.001,
    "output_frequency": 10,
    "gradient_sampling_frequency": 1,
}

dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": "results/forward_outpu_fluid_solid_coupled.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "graadient_output": False,
    "gradient_filename": None,
    "debug_output": False,
    "displacement_output": True,
    "displacement_output_filename": "results/displacement_output.pvd",
    "snapshot_frequency": 20,
    "snapshot_output_dir": "results/snapshots",
}

dictionary["synthetic_data"] = {
    "type": "object",
    "velocity_fluid": 1.5,
    "density_solid": 1.0,
    "p_wave_velocity": 1.5,
    "s_wave_velocity": 0.0,
    "real_velocity_file": None,
}

Wave_obj = AcousticElasticWave(dictionary=dictionary)
Wave_obj.forward_solve()