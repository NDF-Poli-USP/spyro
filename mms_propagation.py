import spyro
import firedrake as fire
import numpy as np
import matplotlib.pyplot as plt
import sys

final_time = 1.0

def error_norm(u, u_an):
    L2 = fire.assemble((u-u_an)**2*fire.dx) #L2 norm
    L2_initial = fire.assemble(u_an**2*fire.dx)
    return np.sqrt(L2/L2_initial)
# dt = float(sys.argv[1])

def get_error(dt):
    # dt = 0.0001

    dictionary = {}
    dictionary["options"] = {
        "cell_type": "Q",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
        "variant": 'lumped',  # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
        "degree": 4,  # p order
        "dimension": 2,  # dimension
    }

    # Number of cores for the shot. For simplicity, we keep things serial.
    # spyro however supports both spatial parallelism and "shot" parallelism.
    dictionary["parallelism"] = {
        "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
    }

    # Define the domain size without the PML. Here we'll assume a 1.00 x 1.00 km
    # domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML) to absorb
    # outgoing waves on three sides (eg., -z, +-x sides) of the domain.
    dictionary["mesh"] = {
        "Lz": 1.0,  # depth in km - always positive
        "Lx": 1.0,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "mesh_type": "firedrake_mesh",  # options: firedrake_mesh or user_mesh
        "mesh_file": None,  # specify the mesh file
    }

    # Create a source injection operator. Here we use a single source with a
    # Ricker wavelet that has a peak frequency of 5 Hz injected at the center of the mesh.
    # We also specify to record the solution at a microphone near the top of the domain.
    # This transect of receivers is created with the helper function `create_transect`.
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-1.0, 1.0)],#, (-0.605, 1.7), (-0.61, 1.7), (-0.615, 1.7)],#, (-0.1, 1.5), (-0.1, 2.0), (-0.1, 2.5), (-0.1, 3.0)],
        "frequency": 5.0,
        "delay": 1.5,
        "receiver_locations": [(-0.0, 0.5)],
    }

    # Simulate for 2.0 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": final_time,  # Final time for event
        "dt": dt,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM
    }

    dictionary["visualization"] = {
        "forward_output" : True,
        "output_filename": "results/forward_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
    }

    Wave_obj = spyro.AcousticWaveMMS(dictionary=dictionary)
    Wave_obj.set_mesh(dx=0.02)

    # Wave_obj.set_initial_velocity_model(constant = 1.0)
    Wave_obj.set_initial_velocity_model(expression = "1 + sin(pi*-z)*sin(pi*x)")
    Wave_obj.forward_solve()

    time = np.linspace(0.0, final_time, int(final_time/dt)+1)

    rec_out = Wave_obj.receivers_output
    np.save("mms_quads_rec_out"+str(dt)+".npy", rec_out)

    # plt.plot(time, Wave_obj.receivers_output)
    # plt.show()

    u_an = Wave_obj.analytical

    fire.File("u_analytical.pvd").write(u_an)
    fire.File("u_numerical.pvd").write(Wave_obj.u_n)
    error = error_norm(Wave_obj.u_n, u_an)

    print(f"Error norm for dt = {dt} is: {error}")

    print("END")

    return error

dts = [
    # 0.002,
    # 0.0015,
    0.001,
    0.0008,
    0.0005,
    0.0003,
    0.0001,
]

errors = []
for dt in dts:
    errors.append(get_error(dt))

for dt in dts:
    print(f"dt = {dt}, error = {errors[dts.index(dt)]}")

plt.loglog(dts, errors)

# theory = [t for t in dts]
# theory = [errors[0]*th/theory[0] for th in theory]

# plt.loglog(dts, theory, '-.', label='1st order in time')

theory = [t**2 for t in dts]
theory = [errors[0]*th/theory[0] for th in theory]

plt.loglog(dts, theory, '--', label='2nd order in time')

# theory = [t**3 for t in dts]
# theory = [errors[0]*th/theory[0] for th in theory]

# plt.loglog(dts, theory, '-.', label='3rd order in time')
plt.legend()
plt.title(f"Convergence for quads with final time = {final_time}")
plt.show()