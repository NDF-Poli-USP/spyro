import os
from scipy.ndimage import gaussian_filter
import segyio
import numpy as np
import matplotlib.pyplot as plt

# smooth field function from Alexandre {{{
def smooth_field(input_filename, output_filename, show = False, sigma = 100, v_water=1501):
    f, filetype = os.path.splitext(input_filename)    
    
    if filetype == ".segy":
        with segyio.open(input_filename, ignore_geometry=True) as f:
            nz, nx = len(f.samples), len(f.trace)
            vp = np.zeros(shape=(nz, nx))
            for index, trace in enumerate(f.trace):
                vp[:, index] = trace    
        
    vp_smooth = gaussian_filter(vp, sigma)
    ni, nj = np.shape(vp)    
    
    for i in range(ni):
        for j in range(nj):
            if vp[i,j] < v_water: # check units (default is m/s) 
                vp_smooth[i,j] = vp[i,j]    
    
    spec = segyio.spec()
    spec.sorting = 2 # not sure what this means
    spec.format = 1 # not sure what this means
    spec.samples = range(vp_smooth.shape[0])
    spec.ilines = range(vp_smooth.shape[1])
    spec.xlines = range(vp_smooth.shape[0])    
    
    assert np.sum(np.isnan(vp_smooth[:])) == 0    
    
    with segyio.create(output_filename, spec) as f:
        for tr, il in enumerate(spec.ilines):
            f.trace[tr] = vp_smooth[:, tr]    
            
    if show == True:
        with segyio.open(output_filename, ignore_geometry=True) as f:
            nz, nx = len(f.samples), len(f.trace)
            show_vp = np.zeros(shape=(nz, nx))
            for index, trace in enumerate(f.trace):
                show_vp[:, index] = trace
            
        fig, ax = plt.subplots()
        plt.pcolormesh(show_vp, shading="auto")
        plt.title("Guess model")
        plt.colorbar(label="P-wave velocity (km/s)")
        plt.xlabel("x-direction (m)")
        plt.ylabel("z-direction (m)")
        ax.axis("equal")
        plt.show()    
    
    return None
#}}}

fname = "./velocity_models/elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy"  # in m/s
#fname = "./velocity_models/elastic-marmousi-model/model/MODEL_S-WAVE_VELOCITY_1.25m.segy"   # in m/s
#fname = "./velocity_models/elastic-marmousi-model/model/MODEL_DENSITY_1.25m.segy"          # in g/cm3

v_water = 1500.1 # for Vp (m/s)
#v_water = 0.1 # for Vs (m/s)

show=True
sigma=300

input_filename=fname
output_filename=fname+".smoothed_sigma="+str(sigma)+".segy"
smooth_field(input_filename, output_filename, show = show, sigma = sigma, v_water=v_water)


