import os
import spyro
from sys import float_info
import firedrake as fire
import math
import numpy as np
import ipdb
import warnings
import time             # For runtime
import tracemalloc      # For memory usage
os.environ["OMP_NUM_THREADS"] = "1"
fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}
warnings.filterwarnings("ignore", category=RuntimeWarning)


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
        "degree": 1,  # p order p=4 ok
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
        # "source_locations": [(-0.5, 0.25), (-0.5, 0.35), (-0.5, 0.5)],
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
        "gradient_sampling_frequency": 100,  # how frequently to save to RAM
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
    edge_length = 0.05

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


def assemble_eik(Wave, u, vy, dx, f_est=1.0):
    '''
    Eikonal with stabilizer term
    '''
    eps = fire.Constant(f_est) * fire.CellDiameter(Wave.mesh)  # Stabilizer
    delta = fire.Constant(float_info.min)
    # delta = fire.Constant(float_info.epsilon)
    grad_u_norm = fire.sqrt(fire.inner(fire.grad(u), fire.grad(u))) + delta
    f = fire.Constant(1.0)
    F = (grad_u_norm * vy * dx - f / Wave.c * vy * dx + eps * fire.inner(
        fire.grad(u), fire.grad(vy)) * dx)
    return F


def solve_prop(nl_solver='newtonls', l_solver='preonly',
               user_atol=1e-16, user_iter=50, monitor=False):
    '''
    Solver Parameters
    https://petsc.org/release/manualpages/SNES/SNESType/
    https://petsc.org/release/manualpages/KSP/KSPType/
    https://petsc.org/release/manualpages/PC/PCType/

    atol: F(x) ≤ atol
    rtol: F(x) ≤ rtol∗F(x0)
    stol: || delta x || < stol*|| x ||
    haptol: lhs - rhs < haptol
    haptol < atol < rtol < stol
    '''

    # Tolerances and iterations
    user_rtol = user_atol * 1e2
    user_stol = user_atol * 1e3
    ksp_max_it = user_iter

    param_solver = {'snes_type': nl_solver, 'ksp_type': l_solver}

    if nl_solver == 'newtontr':  # newton, cauchy, dogleg
        param_solver.update({'snes_tr_fallback_type': 'newton'})

    if nl_solver == 'ngmres':  # difference, none, linesearch
        param_solver.update({'snes_ngmres_select_type': 'linesearch'})

    if nl_solver == 'qn':
        param_solver.update({'snes_qn_m_type': 5,
                             'snes_qn_powell_descent': True,
                             # lbfgs, broyden, badbroyden
                             'snes_qn_type': 'badbroyden',
                             # diagonal, none, scalar, jacobian
                             'snes_qn_scale_type': 'jacobian'})

    if nl_solver == 'ngs':
        param_solver.update({'snes_ngs_sweeps': 2,
                             'snes_ngs_atol': user_atol,
                             'snes_ngs_rtol': user_rtol,
                             'snes_ngs_stol': user_stol,
                             'snes_ngs_max_it': user_iter})

    if nl_solver == 'ncg':
        # fr, prp, dy, hs, cd
        param_solver.update({'snes_ncg_type': 'cd'})

    if l_solver == 'preonly':
        ig_nz = False
        pc_type = 'lu'  # lu, cholesky

        param_solver.update({'pc_factor_mat_solver_type': 'umfpack'})  # mumps
    else:
        ig_nz = True
        pc_type = 'ilu'  # ilu, icc, lu, cholesky

    if l_solver == 'gmres':
        param_solver.update({'ksp_gmres_restart': 3,
                             'ksp_gmres_haptol': user_atol * 1e-4})

    param_solver.update({
        'snes_linesearch_type': 'l2',  # l2, cp, basic
        'snes_linesearch_damping': 1.0,  # 0.1-41 0.3-30 0.5-21 0.9-13 1.0-7
        'snes_linesearch_maxstep': 1.0,
        'snes_max_funcs': 1000,
        'snes_linesearch_order': 2,
        'snes_linesearch_alpha': 1e-4,
        'snes_max_it': user_iter,
        'snes_linesearch_rtol': user_rtol,
        'snes_linesearch_atol': user_atol,
        'snes_rtol': user_atol,
        'snes_atol': user_rtol,
        'snes_stol': user_stol,
        'ksp_max_it': ksp_max_it,
        'ksp_rtol': user_rtol,
        'ksp_atol': user_atol,
        'ksp_initial_guess_nonzero': ig_nz,
        'pc_type': pc_type,
        'pc_factor_reuse_ordering': True,
        'snes_monitor': None,
    })

    if monitor:  # For debugging
        param_solver.update({
            'snes_view': None,
            'snes_converged_reason': None,
            'snes_linesearch_monitor': None,
            'ksp_monitor_true_residual': None,
            'ksp_converged_reason': None,
            'report': True,
            'error_on_nonconvergence': True})
    return param_solver


def clean_inst_num(data_arr):
    ''''
    Clean data: Set NaNs and negative values to zero
    '''
    data_arr[np.where(np.isnan(data_arr) | np.isinf(
        data_arr) | (data_arr < 0.0))] = 0.0
    return data_arr


def solve_eik(Wave, bcs_eik, tol=1e-16, f_est=0.21):
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
    J = fire.derivative(FeikL, yp)

    # Initial guess
    cell_diameter_function = fire.Function(Wave.function_space)
    cell_diameter_function.interpolate(fire.CellDiameter(Wave.mesh))
    yp.assign(cell_diameter_function.dat.data_with_halos.max()
              / Wave.c.dat.data_with_halos.min())

    # Linear Eikonal
    user_atol = tol**0.75
    # newtontr, qn, ngs, ncg, ngmres
    nl_solver = 'newtontr'
    l_solver = 'gmres'  # gmres, preonly, bcgs
    while True:
        try:
            pL = solve_prop(nl_solver=nl_solver, l_solver=l_solver,
                            user_atol=user_atol, user_iter=50)
            fire.solve(fire.lhs(FeikL) == fire.rhs(FeikL), yp,
                       bcs=bcs_eik, solver_parameters=pL, J=J)
            print(f"\nSolver Executed Successfully. AbsTol: {user_atol:.1e}")
            break
        except Exception as e:
            print(f"Error Solving: {e}")
            user_atol = user_atol * 10 if user_atol < 1e-5 \
                else round(user_atol + 1e-5, 5)
            if user_atol > 1e-4:
                print("\nTolerance too high. Exiting.")
                break

    # Clean data: Set NaNs and negative values to zero
    data_eikL = clean_inst_num(yp.dat.data_with_halos[:])

    # Nonlinear Eikonal
    print('Solving Post-Eikonal')
    user_atol = tol
    # vinewtonrsls, vinewtonssls, newtonls, newtontr, qn, ngs, ncg, ngmres
    nl_solver = 'vinewtonrsls'
    l_solver = 'preonly'  # gmres, preonly, bcgs
    user_est = f_est
    while True:
        try:
            print(f"Iteration for Festab: {user_est:.2f}")
            yp = fire.Function(Wave.function_space, name='Eikonal (Time [s])')
            yp.dat.data_with_halos[:] = data_eikL
            Feik = assemble_eik(Wave, yp, vy, fire.dx, f_est=user_est)
            J = fire.derivative(Feik, yp)
            pNL = solve_prop(nl_solver=nl_solver, l_solver=l_solver,
                             user_atol=user_atol, user_iter=50)
            fire.solve(Feik == 0, yp, bcs=bcs_eik, solver_parameters=pNL, J=J)
            print(f"\nSolver Executed Successfully. AbsTol: {user_atol:.1e}")
            print(f"Solver Executed Successfully. Festab: {user_est:.2f}")
            break
        except Exception as e:
            print(f"Error Solving: {e}")
            user_est += 0.01
            if user_est > 1.0:
                user_est = f_est
                print('\nHigh Stabilizing Factor. Increasing Tolerance!')
                user_atol = user_atol * 10 if user_atol < 1e-5 \
                    else round(user_atol + 1e-5, 5)
                if user_atol > 1e-4:
                    print('\nHigh Tolerance. Exiting!')
                    break

    return yp


def eikonal(Wave):
    '''
    Eikonal solver
    '''

    tracemalloc.start()
    tRef = time.perf_counter()  # Reference time. Don't move

    # Boundary conditions
    bcs_eik = define_bcs(Wave)

    # Solving Eikonal
    yp = solve_eik(Wave, bcs_eik)

    # Eikonal results
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

    # Total time
    print(67*'*')
    print('Estimating Runtime and Used Memory')
    # tTotal = time.clock() - tRef
    tTotal = time.perf_counter() - tRef
    print('tTotal(s):{:3.3f}, tTotal(m):{:3.3f}, tTotal(h):{:3.3f}'.format(
        tTotal, tTotal/60, tTotal/3600))

    # Memory usage
    curr, peak = tracemalloc.get_traced_memory()
    print('UdMem(MB):{:3.3f}, PkMem(MB):{:3.3f}'.format(
        curr/1024**2, peak/1024**2))
    tracemalloc.stop()
    print(67*'*')

    # Save file for resource usage
    np.savetxt('/mnt/d/spyro/output/Eik_time.txt',
               (tTotal, tTotal/60, tTotal/3600, curr/1024**2,
                peak/1024**2), delimiter='\t')

    return yp


# Verify distant values as lref and velocities
if __name__ == "__main__":
    test_eikonal_values_fig8()
