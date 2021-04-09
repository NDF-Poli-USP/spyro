import SeismicMesh as sm
import meshio

fname = "eage_slice_ps_true.segy"
bbox = (-4200.0, 0.0, 0.0, 13520.0)

# Desired minimum mesh size in domain
wl = 10
freq = 5
hmin = 1500.0 / (wl * freq)

rectangle = sm.Rectangle(bbox)

ef = sm.get_sizing_function_from_segy(
    fname,
    bbox,
    hmin=hmin,
    wl=wl,
    freq=freq,
    dt=0.001,
    grade=0.15,
    domain_pad=500,
    pad_style="edge",
)

sm.plot_sizing_function(ef)

points, cells = sm.generate_mesh(domain=rectangle, edge_length=ef)

sm.write_velocity_model(
    fname,
    ofname="eage_true_ps_slice",
    bbox=bbox,
    domain_pad=500.0,
    pad_style="edge",
    units="m-s",
)


meshio.write_points_cells(
    "eage_true_ps_slice.msh",
    points / 1000.0,
    [("triangle", cells)],
    file_format="gmsh22",
    binary=False,
)


fname = "eage_slice_ps_guess.segy"
# Bounding box describing domain extents (corner coordinates)
# top right corner is at (0.0, 13520.0, 13520.0)
bbox = (-4200.0, 0.0, 0.0, 13520.0)

# Desired minimum mesh size in domain
wl = 10
freq = 5
hmin = 1500.0 / (wl * freq)

rectangle = sm.Rectangle(bbox)

# Construct mesh sizing object from velocity model
ef = sm.get_sizing_function_from_segy(
    fname,
    bbox,
    hmin=hmin,
    wl=wl,
    freq=freq,
    dt=0.001,
    grade=0.15,
    domain_pad=500,
    pad_style="edge",
)

sm.plot_sizing_function(ef)

points, cells = sm.generate_mesh(domain=rectangle, edge_length=ef)

sm.write_velocity_model(
    fname,
    ofname="eage_guess_ps_slice",
    bbox=bbox,
    domain_pad=500.0,
    pad_style="edge",
    units="m-s",
)
meshio.write_points_cells(
    "eage_guess_ps_slice.msh",
    points / 1000.0,
    [("triangle", cells)],
    file_format="gmsh22",
    binary=False,
)
