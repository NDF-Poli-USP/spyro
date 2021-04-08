import meshio
from SeismicMesh import *

# velocity model for guess problem
fname = "immersed_disk_guess_vp.segy"  # generated via create_immersed_disk_velocity_models.m
bbox = (-650.0, 0.0, 0.0, 1000.0)
wl = 40
freq =5
hmin = 1500 / (wl * freq)
rectangle = Rectangle(bbox)

ef = get_sizing_function_from_segy(
    fname,
    bbox,
    hmin=hmin,
    wl=wl,
    freq=freq,
    grade=0.05,
    units="km-s",
    domain_pad=250,
    pad_style="edge",
)

write_velocity_model(
    fname,
    ofname="immersed_disk_guess_vp",
    bbox=bbox,
    units="km-s",
    domain_pad=250,
    pad_style="edge",
)

points, cells = generate_mesh(domain=rectangle, edge_length=ef, max_iter=200)

meshio.write_points_cells(
    "immersed_disk_guess_vp.msh",
    points / 1000.0,
    [("triangle", cells)],
    file_format="gmsh22",
    binary=False,
)
