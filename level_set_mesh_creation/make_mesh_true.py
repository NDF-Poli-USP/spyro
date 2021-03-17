import meshio
from SeismicMesh import *

# velocity model for true problem
fname = (
    "immersed_disk_true_vp.segy"  # generated via create_immersed_disk_velocity_models.m
)
bbox = (-1500.0, 0.0, 0.0, 1500.0)
wl = 5
freq = 5
hmin = 1500 / (wl * freq)

rectangle = Rectangle(bbox)

ef = get_sizing_function_from_segy(
    fname,
    bbox,
    hmin=hmin,
    hmax=300,
    wl=wl,
    freq=freq,
    dt=0.001,
    cr_max=0.5,
    grade=0.35,
    domain_pad=500,
    pad_style="edge",
    units="km-s",
)

write_velocity_model(
    fname,
    ofname="immersed_disk_true_vp",
    bbox=bbox,
    domain_pad=500.0,
    pad_style="edge",
    units="km-s",
)

points, cells = generate_mesh(domain=rectangle, edge_length=ef, max_iter=200)

meshio.write_points_cells(
    "immersed_disk_true_vp.msh",
    points / 1000.0,
    [("triangle", cells)],
    file_format="gmsh22",
    binary=False,
)
