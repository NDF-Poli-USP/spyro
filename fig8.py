import os
import spyro
from sys import float_info
import firedrake as fire
import math
import numpy as np
import ipdb
os.environ["OMP_NUM_THREADS"] = "1"
fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}


def test_eikonal_values_fig8():
    dictionary = {}
    dictionary["options"] = {
        # Simplexes: triangles or tetrahedra (T) or quadrilaterals (Q)
        "cell_type": "T",
        "variant": "lumped",  # Options: lumped, equispaced or DG.
        # Default is lumped "method":"MLT"
        # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral)
        # You can either specify a cell_type+variant or a method
        # accepted_variants = ["lumped", "equispaced", "DG"]
        "degree": 4,  # p order
        "dimension": 2,  # dimension
    }

    # Number of cores for the shot. For simplicity, we keep things serial.
    # spyro however supports both spatial parallelism and "shot" parallelism.
    # Options: automatic (same number of cores for evey processor) or spatial
    dictionary["parallelism"] = {
        "type": "spatial",
    }

    # Define the domain size without the PML or AL. Here we'll assume a
    # 1.00 x 1.00 km domain and compute the size for the Absorbing Layer (AL)
    # to absorb outgoing waves on boundries (-z, +-x sides) of the domain.
    dictionary["mesh"] = {
        "Lz": 1.0,  # depth in km - always positive
        "Lx": 1.0,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "mesh_type": "firedrake_mesh",
    }

    # Create a source injection operator. Here we use a single source with a
    # Ricker wavelet that has a peak frequency of 5 Hz injected at a specified
    # point of the mesh. We also specify to record the solution at a microphone
    # near the top of the domain. This transect of receivers is created with
    # the helper function `create_transect`.
    dictionary["acquisition"] = {
        "source_type": "ricker",
        # "source_locations": [(-0.5, 0.2), (-0.5, 0.25), (-0.5, 0.3)],
        "source_locations": [(-0.5, 0.25)],
        "frequency": 5.0,
        "delay": 1.5,
        "receiver_locations": spyro.create_transect(
            (-0.10, 0.1), (-0.10, 0.9), 20),
    }

    # Simulate for 1.0 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": 1.00,  # Final time for event
        "dt": 0.0005,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 100,  # how frequently to save solution to RAM
    }

    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "results/fd_forward_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
    }

    # Create the acoustic wave object
    Wave_obj = spyro.AcousticWave(dictionary=dictionary)

    # Using SeismicMesh:
    # cpw = 5.0
    # lba = 1.5 / 5.0
    # edge_length = lba / cpw
    edge_length = 0.01

    Wave_obj.set_mesh(mesh_parameters={"edge_length": edge_length})
    cond = fire.conditional(Wave_obj.mesh_x < 0.5, 3.0, 1.5)

    # Rest of setup
    Wave_obj.set_initial_velocity_model(conditional=cond)
    Wave_obj.c = Wave_obj.initial_velocity_model

    outfile = fire.VTKFile("/mnt/d/spyro/output/output.pvd")
    outfile.write(Wave_obj.c)

    # ipdb.set_trace()
    yp = eikonal(Wave_obj)


class dir_point_bc(fire.DirichletBC):
    '''
    Class for Eikonal boundary conditions at a point.
    '''

    def __init__(self, V, value, nodes):
        # Call superclass init
        # We provide a dummy subdomain id.
        super(dir_point_bc, self).__init__(V, value, 0)
        # Override the "nodes" property which says where the boundary
        # condition is to be applied.
        self.nodes = nodes


def define_bcs(Wave):
    '''
    BCs for eikonal
    '''

    print('Defining Eikonal BCs')

    # Extract node positions
    z_f = fire.Function(Wave.function_space).interpolate(Wave.mesh_z)
    x_f = fire.Function(Wave.function_space).interpolate(Wave.mesh_x)
    z_data = z_f.dat.data_with_halos[:]
    x_data = x_f.dat.data_with_halos[:]

    # Identify source locations
    possou = Wave.sources.point_locations
    sou_ids = [np.where(np.isclose(z_data, z_s) & np.isclose(
        x_data, x_s))[0] for z_s, x_s in possou]

    # Define BCs for eikonal
    bcs_eik = [dir_point_bc(Wave.function_space,
                            fire.Constant(0.0), ids) for ids in sou_ids]

    # Mark source locations
    sou_marker = fire.Function(Wave.function_space, name="source_marker")
    sou_marker.assign(0)
    sou_marker.dat.data_with_halos[sou_ids] = 1

    # Save source marker
    outfile = fire.VTKFile("/mnt/d/spyro/output/souEik.pvd")
    outfile.write(sou_marker)

    return bcs_eik


def linear_eik(Wave, u, vy, dx):
    '''
    Linear Eikonal
    '''
    f = fire.Constant(1.0)
    FL = fire.inner(fire.grad(u), fire.grad(vy)) * dx - f / Wave.c * vy * dx
    return FL


def assemble_eik(Wave, u, vy, dx):
    '''
    Eikonal with stabilizer term
    '''
    eps = fire.Constant(1.) * fire.CellDiameter(Wave.mesh)  # Stabilizer

    # delta = fire.Constant(float_info.min)
    delta = fire.Constant(float_info.epsilon)
    f = fire.Constant(1.0)
    grad_u_norm = fire.sqrt(fire.inner(fire.grad(u), fire.grad(u))) + delta
    F = (grad_u_norm * vy * dx - f / Wave.c * vy * dx + eps * fire.inner(
        fire.grad(u), fire.grad(vy)) * dx)
    return F


def solve_eik(Wave, bcs_eik):
    '''
    Solve nonlinear eikonal
    '''

    # Functions
    yp = fire.Function(Wave.function_space, name='Eikonal (Time [s])')
    vy = fire.TestFunction(Wave.function_space)
    u = fire.TrialFunction(Wave.function_space)

    # Linear Eikonal
    print('Solving Pre-Eikonal')
    FeikL = linear_eik(Wave, u, vy, fire.dx)
    fire.solve(fire.lhs(FeikL) == fire.rhs(FeikL), yp, bcs=bcs_eik)

    # Nonlinear Eikonal
    print('Solving Post-Eikonal')
    Feik = assemble_eik(Wave, yp, vy, fire.dx)
    J = fire.derivative(Feik, yp)
    user_tol = 1e-16
    fire.solve(Feik == 0, yp, bcs=bcs_eik, solver_parameters={
        'snes_type': 'vinewtonssls',
        'snes_max_it': 1000,
        'snes_atol': user_tol,  # Increase the tolerance
        'snes_rtol': 1e-20,
        'snes_linesearch_type': 'l2',
        'snes_linesearch_damping': 1.00,
        'snes_linesearch_maxstep': 0.50,
        'snes_linesearch_order': 2,
        'pc_type': 'lu',
        'ksp_type': 'gmres',
        'ksp_max_it': 1000,
        'ksp_atol': user_tol,  # Increase the tolerance
    }, J=J)

    return yp


def eikonal(Wave):
    '''
    Eikonal solver
    '''

    # Boundary conditions
    bcs_eik = define_bcs(Wave)

    # Solving Eikonal
    yp = solve_eik(Wave, bcs_eik)

    eikonal_file = fire.VTKFile('/mnt/d/spyro/output/Eik.pvd')
    eikonal_file.write(yp)

    # Extract node positions
    z_f = fire.Function(Wave.function_space).interpolate(Wave.mesh_z)
    x_f = fire.Function(Wave.function_space).interpolate(Wave.mesh_x)
    z_data = z_f.dat.data_with_halos[:]
    x_data = x_f.dat.data_with_halos[:]
    poseik = [(-0.5, 0)]
    eik_ids = [np.where(np.isclose(z_data, z_s) & np.isclose(
        x_data, x_s))[0] for z_s, x_s in poseik]

    for pos in eik_ids:
        print('Eikonal min:', round(1e3 * yp.dat.data_with_halos[
            pos].item(), 3), 'ms')

    print('min:', round(1e3*yp.dat.data_with_halos[:].min(), 3), 'ms')
    print('max:', round(yp.dat.data_with_halos[:].max(), 5), 's')

    return yp


# Verify distant values as lref and velocities
if __name__ == "__main__":
    test_eikonal_values_fig8()
