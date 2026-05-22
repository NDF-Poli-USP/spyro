import numpy as np
import spyro
from numpy import abs, array, max, sqrt
from spyro.solvers.elastic_wave.elastic_wave import (PropISO, PropVTI, PropTTI)


def resolve_source_delay(source_delay, frequency, source_delay_mode):
    if source_delay_mode == "explicit":
        return source_delay
    if source_delay_mode == "specfem-single-force-ricker":
        return 1.2 / frequency
    raise ValueError(f"Unsupported source delay mode: {source_delay_mode}")


source_z = -1.1
source_x = 1.5
source_y = 1.5
receiver_z = -1.9
receiver_y = 1.5
edge_length = 0.25

source_direction_xyz = (1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 0.0)
source_delay_requested = 0.2
source_delay_mode = "specfem-single-force-ricker"
effective_source_delay = resolve_source_delay(source_delay_requested,
                                              frequency=5.0,
                                              source_delay_mode=source_delay_mode)

receiver_locations = spyro.create_transect((receiver_z, 1.2, receiver_y),
                                           (receiver_z, 1.8, receiver_y), 300)
source_dir_x, source_dir_y, source_dir_z = source_direction_xyz

dM = 0.1  # Mass density
vP = 1.5  # P-wave velocity
vS = 1.0  # S-wave velocity

dictionary = {
    "options": {
        "cell_type": "Q",
        "variant": "lumped",
        "degree": 4,
        "dimension": 3,
    },
    "parallelism": {
        "type": "automatic",
    },
    "mesh": {
        "length_z": 3.0,
        "length_x": 3.0,
        "length_y": 3.0,
        "mesh_type": "firedrake_mesh",
    },
    "acquisition": {
        "source_type": "ricker",
        "source_locations": [(source_z, source_x, source_y)],
        "frequency": 5.0,
        "delay": effective_source_delay,
        "delay_type": "time",
        "receiver_locations": receiver_locations,
        "amplitude": array([source_dir_z, source_dir_x, source_dir_y], dtype=float),
        "use_vertex_only_mesh": True,
    },
    "time_axis": {
        "initial_time": 0.0,
        "final_time": 1.5,
        "dt": 0.001,
        "output_frequency": 100,
    },
    "synthetic_data": {
        "type": "object",
        "density": dM,
        "p_wave_velocity": vP,
        "s_wave_velocity": vS,
    },
}

iso_constants = (vP, vS, dM)
epsilon = 0.
gamma = 0.
delta = 0.
vti_constants = (epsilon, gamma, delta)
theta = 0.
phi = 0.
tti_constants = (theta, phi)

print("Running Isotropic Wave Propagation", flush=True)
wave_iso = spyro.IsotropicWave(dictionary)
wave_iso.set_mesh(input_mesh_parameters={"edge_length": edge_length, "periodic": False})
wave_iso.forward_solve()
last_displacement_iso = wave_iso.u_n.dat.data_ro.copy()
u_max_iso = max(abs(last_displacement_iso), axis=0)

print("Running VTI Wave Propagation", flush=True)
wave_vti = spyro.ElasticWave(dictionary, anisotropy="VTI")
wave_vti.set_mesh(input_mesh_parameters={"edge_length": edge_length, "periodic": False})
wave_vti.get_anisotropy_properties(iso_constants, vti_constants=vti_constants)
wave_vti.forward_solve()
last_displacement_vti = wave_vti.u_n.dat.data_ro.copy()
u_max_vti = max(abs(last_displacement_vti), axis=0)

print("Running TTI Wave Propagation", flush=True)
wave_tti = spyro.ElasticWave(dictionary, anisotropy="TTI")
wave_tti.set_mesh(input_mesh_parameters={"edge_length": edge_length, "periodic": False})
wave_tti.get_anisotropy_properties(iso_constants, vti_constants=vti_constants,
                                   tti_constants=tti_constants)
wave_tti.forward_solve()
last_displacement_tti = wave_tti.u_n.dat.data_ro.copy()
u_max_tti = max(abs(last_displacement_tti), axis=0)
