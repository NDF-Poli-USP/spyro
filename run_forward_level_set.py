import time

from firedrake import *

import spyro

# 10 outside
# 11 inside

model = {}

model["opts"] = {
    "method": "KMV",
    "degree": 1,  # p order
    "dimension": 2,  # dimension
}

model["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor), custom, off
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    # input is a list of integers with the length of the number of shots.
}

model["mesh"] = {
    "Lz": 1.50,  # depth in km - always positive
    "Lx": 1.50,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/immersed_disk_v2.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

model["PML"] = {
    "status": False,  # true,  # true or false
    "outer_bc": "non-reflective",  #  dirichlet, neumann, non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,
    "cmax": 4.5,  # maximum acoustic wave velocity in pml - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 0.25,  # thickness of the pml in the z-direction (km) - always positive
    "lx": 0.25,  # thickness of the pml in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 2,
    "source_pos": [
        (-0.10, 0.25),
        (-0.10, 0.75),
    ],  # spyro.create_receiver_transect((-0.10, 0.30), (-0.10, 1.20), 4),
    "frequency": 10.0,
    "delay": 1.0,
    "amplitude": 1.0,
    "num_receivers": 200,
    "receiver_locations": spyro.create_receiver_transect(
        (-0.10, 0.30), (-0.10, 1.20), 200
    ),
}

model["timeaxis"] = {
    "t0": 0.0,  #  initial time for event
    "tf": 0.70,  # final time for event
    "dt": 0.0001,  # timestep size
    "nspool": 200,  # how frequently to output solution to pvds
    "fspool": 1000,  # how frequently to save solution to ram
}


comm = spyro.utils.mpi_init(model)

mesh, V = spyro.io.read_mesh(model, comm)

# assign values [4.5 and 2.0]
u = TrialFunction(V)
v = TestFunction(V)
q = Function(V)
# make the assumption that subdomains are named 10 and 11
solve(u * v * dx == 1 * v * dx(10) + -1 * v * dx(11), q)

vp = Function(V)
sd1 = SubDomainData(q < 0)
sd2 = SubDomainData(q > 0)

vp.interpolate(Constant(4.5), subset=sd1)
vp.interpolate(Constant(2.0), subset=sd2)
File("exact_vp.pvd").write(vp)

qr_x, _, _ = spyro.domains.quadrature.quadrature_rules(V)


sources = spyro.Sources(model, mesh, V, comm).create()

receivers = spyro.Receivers(model, mesh, V, comm).create()

for sn in range(model["acquisition"]["num_sources"]):
    if spyro.io.is_owner(comm, sn):
        t1 = time.time()
        p_field, p_field_dt, p_recv = spyro.solvers.Leapfrog_level_set(
            model, mesh, comm, vp, sources, receivers, source_num=sn
        )
        print(time.time() - t1)
        spyro.io.save_shots("forward_exact_level_set" + str(sn) + ".dat", p_recv)
        # spyro.plots.plot_shotrecords(
        #    model,
        #    p_recv,
        #    name="level_set_" + str(sn),
        #    vmin=-1e-1,
        #    vmax=1e-1,
        #    appear=False,
        # )
