from firedrake import *
import spyro
from scipy.interpolate import griddata
import numpy as np

def saving_segy_from_function(vp, V, filename):

    print("Generating new segy", flush = True)
    segy_fname = filename


    # write a new file to be used in the re-meshing

    grid_spacing = 10.0 / 1000.0
    m = V.ufl_domain()
    W = VectorFunctionSpace(m, V.ufl_element())
    coordinates = interpolate(m.coordinates, W)
    x, y = coordinates.dat.data[:, 0], coordinates.dat.data[:, 1]    
    
    # add buffer to avoid NaN when calling griddata
    min_x = np.amin(x) + 0.01
    max_x = np.amax(x) - 0.01
    min_y = np.amin(y) + 0.01
    max_y = np.amax(y) - 0.01  
    z = vp.dat.data[:]   
    
    # target grid to interpolate to
    xi = np.arange(min_x, max_x, grid_spacing)
    yi = np.arange(min_y, max_y, grid_spacing)
    xi, yi = np.meshgrid(xi, yi)    # interpolate
    vp_i = griddata((x, y), z, (xi, yi), method="linear")
    print("creating new velocity model...", flush=True)
    spyro.io.create_segy(vp_i, segy_fname)