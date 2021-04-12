import time

from firedrake import *

import spyro

model = {}
model["parallelism"] = {
    "type": "off",  # options: automatic (same number of cores for evey processor), custom, off
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
}
model["opts"] = {
    "method": "KMV",
    "degree": 1,  # p order
    "quadrature": "KMV",
    "dimension": 2,  # dimension
}
# bbox = (-9.15162, 0.0, 0.0, 27.43962)
model["mesh"] = {
    "Lz": 9.15162,  # depth in km - always positive
    "Lx": 27.43962,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/sigsbee2b_true.msh",
    "initmodel": "velocity_models/sigsbee2b_guess.hdf5",
    "truemodel": "velocity_models/sigsbee2b_true.hdf5",
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
"""The three surveys shared the same acquisition geometry. Each receiver recorded data every .008 seconds for 1 500 timesteps resulting in 12 seconds of data. A 7 950 m (26 100 ft) long streamer cable was deployed with 348 hydrophones spaced 22.86 m (75 ft) apart. Shots were fired every 45.72 m (150 ft) starting at 3 330 m (10 925 ft). Table 5 shows the values that Sigsbee shot headers should contain."""
# We do 1/10 of the total number of true shots
sources = spyro.create_transect((-0.01, 3.33), (-0.01, 19.4896), 50)
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": len(sources),
    "source_pos": sources,
    "frequency": 5.0,
    "delay": 1.0,
    "amplitude": 1.0,
    "num_receivers": 348,
    "receiver_locations": None,
}
model["timeaxis"] = {
    "t0": 0.0,  #  initial time for event
    "tf": 6.0,  # final time for event
    "dt": 0.001,  # timestep size
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 9999,  # how frequently to save solution to ram
}


comm = spyro.utils.mpi_init(model)

mesh, V = spyro.io.read_mesh(model, comm)

vp_exact = spyro.io.interpolate(model, mesh, V, guess=True)

File("exact_vp.pvd").write(vp_exact, name="true_velocity")

sources = spyro.Sources(model, mesh, V, comm).create()

XMIN = 0.01
XMAX = 27.43
for sn in range(model["acquisition"]["num_sources"]):
    if spyro.io.is_owner(comm, sn):
        # calculate receivers for each source
        offset = 7.950  # offset of 7.950 km
        _, xsrc = model["acquisition"]["source_pos"][sn]
        xmin = xsrc - offset if xsrc - offset > XMIN else 0.01
        xmax = xsrc + offset if xsrc + offset < XMAX else 27.43
        model["acquisition"]["receiver_locations"] = spyro.create_transect(
            (-0.01, xmin), (-0.01, xmax), 348
        )
        receivers = spyro.Receivers(model, mesh, V, comm).create()
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
