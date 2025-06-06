from mpi4py.MPI import COMM_WORLD
from mpi4py import MPI
import numpy as np
import firedrake as fire
import spyro
import pytest


def error_calc(p_numerical, p_analytical, nt):
    norm = np.linalg.norm(p_numerical, 2) / np.sqrt(nt)
    error_time = np.linalg.norm(p_analytical - p_numerical, 2) / np.sqrt(nt)
    div_error_time = error_time / norm
    return div_error_time


def test_forward():
    final_time = 1.0

    dictionary = {}
    dictionary["options"] = {
        "cell_type": "Q",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
        "variant": "lumped",  # lumped, equispaced or DG, default is lumped
        "degree": 4,  # p order
        "dimension": 2,  # dimension
    }
    dictionary["parallelism"] = {
        "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
    }
    dictionary["mesh"] = {
        "Lz": 1.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
        "Lx": 1.0,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",
    }
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-0.1, 0.5)],
        "frequency": 5.0,
        "delay": 0.2,
        "delay_type": "time",
        "receiver_locations": spyro.create_transect((-0.9, 0.2), (-0.9, 0.8), 301),
    }
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": final_time,  # Final time for event
        "dt": 0.001,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds - Perguntar Daiane ''post_processing_frequnecy'
        "gradient_sampling_frequency": 1,
    }
    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "results/forward_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
    }
    dictionary["absorving_boundary_conditions"] = {
        "status": True,
        "damping_type": "local",
        "absorb_top": True,
        "absorb_bottom": True,
        "absorb_right": True,
        "absorb_left": True,
    }

    Wave_obj = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj.set_mesh(mesh_parameters={"dx": 0.1})

    mesh_z = Wave_obj.mesh_z
    cond = fire.conditional(mesh_z < -0.5, 3.5, 1.5)
    Wave_obj.set_initial_velocity_model(conditional=cond, output=True)

    Wave_obj.forward_solve()


if __name__ == "__main__":
    test_forward()
