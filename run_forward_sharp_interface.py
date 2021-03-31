import time

from firedrake import *

import spyro

model = {}
model["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor), custom, off
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    # input is a list of integers with the length of the number of shots.
}
model["opts"] = {
    "method": "KMV",
    "degree": 1,  # p order
    "quadrature": "KMV",
    "dimension": 2,  # dimension
}
model["mesh"] = {
    "Lz": 0.65,  # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/immersed_disk_true_vp.msh",
    "initmodel": "velocity_models/immersed_disk_guess_vp.hdf5",
    "truemodel": "velocity_models/immersed_disk_true_vp.hdf5",
}
model["PML"] = {
    "status": True,  # true,  # true or false
    "outer_bc": "non-reflective",  #  dirichlet, neumann, non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,
    "cmax": 4.5,  # maximum acoustic wave velocity in pml - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 0.50,  # thickness of the pml in the z-direction (km) - always positive
    "lx": 0.50,  # thickness of the pml in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
}
recvs = spyro.create_transect((-0.01, 0.01), (-0.01, 0.99), 100)
sources = spyro.create_transect((-0.01, 0.01), (-0.01, 0.99), 4)
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": len(sources),
    "source_pos": sources,
    "frequency": 5.0,
    "delay": 1.0,
    "amplitude": 1.0,
    "num_receivers": len(recvs),
    "receiver_locations": recvs,
}
model["timeaxis"] = {
    "t0": 0.0,  #  initial time for event
    "tf": 1.0,  # final time for event
    "dt": 0.00025,  # timestep size
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 9999,  # how frequently to save solution to ram
}


comm = spyro.utils.mpi_init(model)

mesh, V = spyro.io.read_mesh(model, comm)

vp_exact = spyro.io.interpolate(model, mesh, V, guess=False)

File("exact_vp.pvd").write(vp_exact, name="true_velocity")

sources = spyro.Sources(model, mesh, V, comm).create()

receivers = spyro.Receivers(model, mesh, V, comm).create()

for sn in range(model["acquisition"]["num_sources"]):
    if spyro.io.is_owner(comm, sn):
        t1 = time.time()
        p_field, p_field_dt, p_recv = spyro.solvers.Leapfrog_level_set(
            model,
            mesh,
            comm,
            vp_exact,
            sources,
            receivers,
            source_num=sn,
            output=True,
        )
        print(time.time() - t1)
        spyro.io.save_shots("shots/forward_exact_level_set" + str(sn) + ".dat", p_recv)
        spyro.plots.plot_shotrecords(
            model,
            p_recv,
            name="level_set_" + str(sn),
            vmin=-1e-3,
            vmax=1e-3,
            appear=False,
        )
