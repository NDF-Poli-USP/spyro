import spyro
import matplotlib.pyplot as plt
import numpy as np

from spyro.solvers.acoustic_elastic_wave import AcousticElasticWave
from spyro.plots.receiver_plots import plot_receiver_response, plot_displacement_components

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
    "edge_length": 0.005,
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
    "receiver_locations": [(-0.51, 0.5025)],
    "solid_receiver_locations": [(-0.49, 0.4975)], 
    "user_vertex_only_mesh": True,
}

dictionary["time_axis"] = {
    "initial_time": 0.0,
    "final_time": 0.22,
    "dt": 0.0001,
    "output_frequency": 10,
    "gradient_sampling_frequency": 1,
}

dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": "results/pressure.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "graadient_output": False,
    "gradient_filename": None,
    "debug_output": False,
    "displacement_output": True,
    "displacement_output_filename": "results/displacement.pvd",
    "snapshot_frequency": 20,
    "snapshot_output_dir": "results/snapshots",
    "p_equivalent_output": True,
    "p_equivalent_output_filename": "results/p_equivalent.pvd",
    "interface_error_frequency": 20,
}

dictionary["synthetic_data"] = {
    "type": "object",
    "velocity_fluid": 1.5,
    "density_solid": 2.0,
    "p_wave_velocity": 2.0,
    "s_wave_velocity": 1.2,
    "real_velocity_file": None,
}

Wave_obj = AcousticElasticWave(dictionary=dictionary)
Wave_obj.forward_solve()

import numpy as np
np.savez(
    "results/spyro_receiver_data.npz",
    p_spyro=np.asarray(Wave_obj.forward_solution_receivers)[:, 0],
    u_solid=np.array(Wave_obj.solid_receiver_history)[:, 0, :],
    dt=dictionary["time_axis"]["dt"],
    final_time=dictionary["time_axis"]["final_time"],
)

receiver_data = Wave_obj.forward_solution_receivers[:, 0]
plot_receiver_response(
    receiver_data,
    final_time=dictionary["time_axis"]["final_time"],
    filename="results/receiver_fluid.png",
    receiver_id_for_title=0,
)

solid_data = np.array(Wave_obj.solid_receiver_history)[:, 0, :]
plot_displacement_components(
    time_vector=np.linspace(0, dictionary["time_axis"]["final_time"], len(solid_data)),
    receiver_results=solid_data,
    source_type="Ricker",
    filename="results/receiver_solid.png",
)