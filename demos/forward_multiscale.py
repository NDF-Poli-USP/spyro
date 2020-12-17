from firedrake import File
import time
import spyro

model = {}

model["opts"] = {
    "method": "KMV",
    "variant": None,
    "type": "SIP",  # for DG only - SIP, NIP and IIP
    "element": "tria",  # tria or tetra
    "degree": 1,  # p order
    "quadrature": "KMV",  # # GLL, GL, Equi
    "dimension": 2,  # dimension
    "beta": 0.0,  # for Newmark time integration only
    "gamma": 0.5,  # for Newmark time integration only
}


model["mesh"] = {
    "Lz": 4.0,  # depth in km - always positive
    "Lx": 18.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/mm_exact.msh",
    "initmodel": "velocity_models/mm_init.hdf5",
    "truemodel": "velocity_models/mm_exact.hdf5",
}


model["PML"] = {
    "status": True,  # True or false
    "outer_bc": "non-reflective",  #  neumann, non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial. hyperbolic, shifted_hyperbolic
    "exponent": 1,
    "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 0.5,  # thickness of the pml in the z-direction (km) - always positive
    "lx": 0.5,  # thickness of the pml in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 4,
    "source_pos": spyro.create_receiver_transect((-0.15, 0.1), (-0.15, 16.9), 4),
    "frequency": 10.0,
    "delay": 1.0,
    "num_receivers": 301,
    "receiver_locations": spyro.create_receiver_transect(
        (-0.15, 0.1), (-0.15, 16.9), 301
    ),
}


# Perform each simulation for 2 seconds.
model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 3.0,  # Final time for event
    "dt": 0.001,  # timestep size
    "nspool": 200,  # how frequently to output solution to pvds
    "fspool": 1,  # how frequently to save solution to RAM
}  # how freq. to output to files and screen


# Use one core per shot.
model["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor), custom, off
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    # input is a list of integers with the length of the number of shots.
}

comm = spyro.utils.mpi_init(model)

mesh, V = spyro.io.read_mesh(model, comm)

vp_exact = spyro.io.interpolate(model, mesh, V, guess=False)

File("vp_exact.pvd").write(vp_exact)

sources = spyro.Sources(model, mesh, V, comm).create()
receivers = spyro.Receivers(model, mesh, V, comm).create()

src_freq = model["acquisition"]["frequency"]

for sn in range(model["acquisition"]["num_sources"]):
    if spyro.io.is_owner(comm, sn):
        t1 = time.time()
        p_field, p_exact_recv = spyro.solvers.Leapfrog(
            model, mesh, comm, vp_exact, sources, receivers, source_num=sn
        )
        print(time.time() - t1)

        spyro.plots.plot_shotrecords(
            model, p_exact_recv, name=str(sn + 1), vmin=-1e-5, vmax=1e-5
        )

        spyro.io.save_shots(
            "shots/mm_exact_" + str(src_freq) + "_Hz_source_" + str(sn) + ".dat",
            p_exact_recv,
        )
