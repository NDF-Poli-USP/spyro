import spyro
import firedrake as fire
import math
import ipdb


def test_eikonal_values_fig8():
    dictionary = {}
    dictionary["options"] = {
        # Simplexes: triangles or tetrahedra (T) or quadrilaterals (Q)
        "cell_type": "T",
        # Options: lumped, equispaced or DG. Default is lumped "method":"MLT"
        # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral)
        # You can either specify a cell_type+variant or a method
        "variant": "equispaced",
        # accepted_variants = ["lumped", "equispaced", "DG"]
        "degree": 4,  # p order
        "dimension": 2,  # dimension
    }

    # Number of cores for the shot. For simplicity, we keep things serial.
    # spyro however supports both spatial parallelism and "shot" parallelism.
    # Options: automatic (same number of cores for evey processor) or spatial
    dictionary["parallelism"] = {
        "type": "automatic",
    }

    # Define the domain size without the PML or AL. Here we'll assume a
    # 1.00 x 1.00 km domain and compute the size for the Absorbing Layer (AL)
    # to absorb outgoing waves on boundries (-z, +-x sides) of the domain.
    dictionary["mesh"] = {
        "Lz": 1.0,  # depth in km - always positive
        "Lx": 1.0,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "mesh_type": "SeismicMesh",
    }

    # Create a source injection operator. Here we use a single source with a
    # Ricker wavelet that has a peak frequency of 5 Hz injected at a specified
    # point of the mesh. We also specify to record the solution at a microphone
    # near the top of the domain. This transect of receivers is created with
    # the helper function `create_transect`.
    dictionary["acquisition"] = {
        "source_type": "ricker",
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
    edge_length = 0.05
    Wave_obj.set_mesh(mesh_parameters={"edge_length": edge_length})
    cond = fire.conditional(Wave_obj.mesh_x < 0.5, 3.0, 1.5)

    # Rest of setup
    Wave_obj.set_initial_velocity_model(conditional=cond)
    # Wave_obj._get_initial_velocity_model()

    Wave_obj.c = Wave_obj.initial_velocity_model
    # Wave_obj.forward_solve()

    outfile = fire.VTKFile("output.pvd")
    outfile.write(Wave_obj.c)

    eikonal(Wave_obj)

    ipdb.set_trace()


def define_bcs(Wave):
    pass
    return bcs_eik


def assemble_eik(Wave, yp, vy, dx):
    f = fire.Constant(1.0)
    eps = fire.CellDiameter(Wave.mesh)  # Stabilizer
    F = (fire.sqrt(fire.inner(fire.grad(yp), fire.grad(yp))) * vy * dx
         - f / Wave.c * vy * dx + eps * fire.inner(
        fire.grad(yp), fire.grad(vy)) * dx)
    return F


def solve_eik(Wave, bcs_eik, yp):

    # Initialize mesh function for boundary domains
    elreg_marker = fire.Function(Wave.function_space, name="elreg_marker")
    elreg_marker.assign(0)
    dx = fire.Measure('dx', domain=Wave.mesh, subdomain_data=elreg_marker)

    # Functions

    vy = fire.TestFunction(Wave.function_space)
    u = fire.TrialFunction(Wave.function_space)

    print('Solve Pre-Eikonal')
    f = fire.Constant(1.0)
    F1 = fire.inner(fire.grad(u), fire.grad(vy)) * dx - f / Wave.c * vy * dx
    fire.solve(fire.lhs(F1) == fire.rhs(F1), yp, bcs=bcs_eik)

    print('Solve Post-Eikonal')
    Feik = assemble_eik(Wave, yp, vy, dx)
    fire.solve(Feik == 0, yp, bcs=bcs_eik, solver_parameters={
        'snes_type': 'vinewtonssls',
        'snes_max_it': 1000,
        'snes_atol': 5e-6,
        'snes_rtol': 1e-20,
        'snes_linesearch_type': 'l2',
        'snes_linesearch_damping': 1.00,
        'snes_linesearch_maxstep': 0.50,
        'snes_linesearch_order': 2,
        'ksp_type': 'gmres',
        'pc_type': 'lu'
    })

    return yp


def eikonal(Wave):  # (Mes, Dom, pH):

    print('Defining Eikonal BCs')
    bcs_eik = define_bcs(Wave)

    # Solving Eikonal
    yp = fire.Function(Wave.function_space, name='Eikonal (Time [ms])')
    yp.assign(solve_eik(Wave, bcs_eik, yp))

    # if pH['saveFile']:
    #     eikonal_file = fire.File(pH['FolderCase'] + '/out/Eik.pvd')
    #     eikonal_file.write(yp)

    # return yp


# Verify distant values as lref and velocities
if __name__ == "__main__":
    test_eikonal_values_fig8()


# def define_boundaries(Wave, edge_length):
#     class Sigma:
#         def __init__(self, possou, tolz, tolx):
#             self.possou = possou
#             self.tolz = tolz
#             self.tolx = tolx

#         def inside(self, x):
#             cond = False
#             for p in self.possou:
#                 cond = cond or (p[0] - self.tolz[0] <= x[0] <= p[0]
#                                 + self.tolz[1] and p[1] - self.tolx[0]
#                                 <= x[1] <= p[1] + self.tolx[1])
#             return cond

#     # Initialize mesh function for boundary domains
#     elreg_marker = fire.Function(Wave.function_space, name="elreg_marker")
#     elreg_marker.assign(0)

#     element = Wave.function_space.ufl_element()
#     function_space_fs = fire.FunctionSpace(
#         Wave.mesh, element.reconstruct(degree=element.degree() - 1))
#     facet_marker = fire.Function(function_space_fs, name="facet_marker")
#     facet_marker.assign(0)

#     # Initialize sub-domain instances
#     possou = Wave.sources.point_locations

#     if Wave.sources.number_of_points > 1:
#         zl = 0.25 * edge_length
#         zu = 0.25 * edge_length
#         xl = 0.25 * edge_length
#         xu = 1.25 * edge_length
#     else:
#         zl = 0.6 * edge_length
#         zu = 0.6 * edge_length
#         xl = 0.6 * edge_length
#         xu = 0.6 * edge_length

#     sigma = Sigma(possou, [zl, zu], [xl, xu])
#     for i in range(len(Wave.mesh.coordinates.dat.data_with_halos)):
#         if sigma.inside(Wave.mesh.coordinates.dat.data_with_halos[i]):
#             facet_marker.dat.data_with_halos[i] = 1

#     bcs = [fire.DirichletBC(Wave.function_space,
#                             fire.Constant(0.0), facet_marker)]

#     dx = fire.Measure('dx', domain=Wave.mesh, subdomain_data=elreg_marker)

#     souEik_file = fire.VTKFile('SouEik.pvd')
#     souEik_file.write(facet_marker)

#     ipdb.set_trace()

#     return dx, bcs
