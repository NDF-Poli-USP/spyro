import firedrake
import spyro

model = spyro.io.load_model()

# Create the computational environment
comm = spyro.utils.mpi_init(model)

# Create mesh
mesh, V = spyro.utils.create_mesh(model, comm, quad=False)

# Load data
vp_exact = spyro.utils.load_velocity_model(
    model, V, source_file=model["data"]["initfile"]
)

# Check initial data
firedrake.File("vp_exact.pvd").write(vp_exact)

# Acquisition geometry
sources, receivers = spyro.Geometry(model, mesh, V, comm).create()

for sn in range(model["acquisition"]["num_sources"]):
    if spyro.io.is_owner(comm, sn):
        # Generate shots
        p_field, p_exact_recv = spyro.solvers.Leapfrog(
            model, mesh, comm, vp_exact, sources, receivers, source_num=sn
        )

        # Plotting
        spyro.plots.plot_shotrecords(
            model, p_exact_recv, name=str(sn), vmin=-1e-4, vmax=1e-4
        )

        # Save'em
        shotfile = (model["data"]["shots"]
                    + str(model["acquisition"]["frequency"]) 
                    + "Hz_sn_" 
                    + str(sn) 
                    + ".dat")
        spyro.io.save_shots(shotfile, p_exact_recv)
