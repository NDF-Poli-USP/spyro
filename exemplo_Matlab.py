from firedrake import File
import firedrake as fire
import spyro

model = {}

model["opts"] = {
    "method": "CG",  # either CG or KMV
    "quadrature": "GLL",
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}
model["parallelism"] = {
    "type": "automatic",
}
model["mesh"] = {
    "Lz": 3.0,  # depth in km - always positive
    "Lx": 7.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": None,
    "initmodel": None,
    "truemodel": None,
}
model["BCs"] = {
    "status": False,  # True or false
    "outer_bc": None,  #  None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.5,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.9,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.9,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
}
model["acquisition"] = {
    "source_type": "Ricker",
    "source_pos": [(-0.1,3.5)],
    "frequency": 5.0,
    "delay": 1.0,
    "receiver_locations": spyro.create_transect((-0.10, 0.1), (-0.10, 7.0), 20),
}
model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 1.00,  # Final time for event
    "dt": 0.00025,
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 99999,  # how frequently to save solution to RAM
}
comm = spyro.utils.mpi_init(model)
mesh = fire.RectangleMesh(60,140,model["mesh"]["Lz"],model["mesh"]["Lx"],quadrilateral = True)
z,x = fire.SpatialCoordinate(mesh)
mesh.coordinates.dat.data[:,0] *= -1.0
V = fire.FunctionSpace(mesh,model["opts"]["method"],model["opts"]["degree"])
vp_temp = fire.conditional(z<-1.5,3.5,1.5)
vp = fire.Function(V)
vp.interpolate(vp_temp)
if comm.ensemble_comm.rank == 0:
    File("true_velocity.pvd", comm=comm.comm).write(vp)
sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(
    dt=model["timeaxis"]["dt"],
    tf=model["timeaxis"]["tf"],
    freq=model["acquisition"]["frequency"],
)
p, p_r = spyro.solvers.forward(model, mesh, comm, vp, sources, wavelet, receivers, output=True)
spyro.plots.plot_shots(model, comm, p_r, vmin=-1e-3, vmax=1e-3)
spyro.io.save_shots(model, comm, p_r)
