import os                                                                       
import multiprocessing as mp
import numpy as np
from mpi4py import MPI
import meshio

from SeismicMesh import get_sizing_function_from_segy, generate_mesh, Rectangle
from SeismicMesh.sizing.mesh_size_function import write_velocity_model
comm = MPI.COMM_WORLD

# Name of SEG-Y file containg velocity model.
fname = "Mar2_Vp_1.25m.segy"
def createMesh(hmin, wl, freq, dt, grade):
    # Bounding box describing domain extents (corner coordinates)
    bbox = (-3500.0, 0.0, 0., 17000.0)
   
    write_velocity_model(fname,bbox=bbox,domain_pad=1e3, pad_style='edge')
    # write_velocity_model(fname)
    rectangle = Rectangle(bbox)

    # Construct mesh sizing object from velocity model
    ef = get_sizing_function_from_segy(
        fname,
        bbox,
        hmin=hmin,
        wl=wl,
        freq=freq,
        dt=dt,
        grade=grade,
        domain_pad=1e3,
        pad_style="edge",
    )


    points, cells = generate_mesh(domain=rectangle, edge_length=ef)

    if comm.rank == 0:
        meshio.write_points_cells(
            "mm.msh",
            points / 1000,
            [("triangle", cells)],
            file_format="gmsh22",
            binary=False
        )

def run_python(process):                                                             
    os.system('python3 {}'.format(process)) 

# Frequencies cut
freqs = [5, 8, 10]

hmin  = 40
wl    = 2.41
dt    = 0.001
grade = 0.2

for f in freqs:
    
    createMesh(hmin,wl,f,dt,grade)
    processes = [] 
    processes.append("run_fwi.py --freq_cut " + str(f))                   
    
    pool = mp.Pool(processes=1)                                                        
    pool.map(run_python, processes)  
    pool.close()
    pool.join()