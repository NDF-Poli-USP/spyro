import firedrake as fire
import math
import spyro


u1 = lambda x, t: (x[0]**2 + x[0])*(x[1]**2 - x[1])*t
u2 = lambda x, t: (2*x[0]**2 + 2*x[0])*(-x[1]**2 + x[1])*t
u = lambda x, t: fire.as_vector([u1(x, t), u2(x, t)])

b1 = lambda x, t: -(2*x[0]**2 + 6*x[1]**2 - 16*x[0]*x[1] + 10*x[0] - 14*x[1] + 4)*t
b2 = lambda x, t: -(-12*x[0]**2 - 4*x[1]**2 + 8*x[0]*x[1] - 16*x[0] + 8*x[1] - 2)*t
b = lambda x, t: fire.as_vector([b1(x, t), b2(x, t)])

dt = 1e-4
h = 0.1
fo = int(0.1/dt)

dictionary = {}
dictionary["options"] = {
    "cell_type": "Q",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}
dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}
dictionary["mesh"] = {
    "Lz": 1.0,  # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_type": "firedrake_mesh",  # options: firedrake_mesh or user_mesh
    "mesh_file": None,  # specify the mesh file
}
dictionary["acquisition"] = {
    "source_type": "MMS",
    "source_locations": [(-1.0, 1.0)],
    "frequency": 5.0,
    "delay": 1.5,
    "receiver_locations": [(-0.0, 0.5)],
    "body_forces": b,
}
dictionary["time_axis"] = {
    "initial_condition": u,
    "initial_time": 0.0,  # Initial time for event
    "final_time": 1.0,  # Final time for event
    "dt": dt,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": fo,  # how frequently to output solution to pvds
    "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM
}

dictionary["visualization"] = {
    "forward_output": False,
    "output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
}

dictionary["synthetic_data"] = {
    "type": "object",
    "density": 1,
    "lambda": 1,
    "mu": 1,
    "real_velocity_file": None,
}
dictionary["boundary_conditions"] = [
    ("uz", "on_boundary", 0),
    ("ux", "on_boundary", 0),
]

wave = spyro.IsotropicWave(dictionary)
wave.set_mesh(input_mesh_parameters={"edge_length": h})
wave.forward_solve()

u_an = fire.Function(wave.function_space)
x = wave.get_spatial_coordinates()
t = wave.current_time
u_an.interpolate(u(x, t))

e1 = fire.errornorm(wave.u_n.sub(0), u_an.sub(0))
e2 = fire.errornorm(wave.u_n.sub(1), u_an.sub(1))

print(f"For h: {h}, e1 = {e1}, e2 = {e2}")

assert math.isclose(e1, 0.0, abs_tol=1e-7)
assert math.isclose(e2, 0.0, abs_tol=1e-7)