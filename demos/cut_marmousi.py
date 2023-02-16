from firedrake import *
from scipy.optimize import *
from movement import *
import spyro
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import h5py
import meshio
#import SeismicMesh
import weakref
import finat
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL
from scipy.ndimage import gaussian_filter
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from spyro.io import write_function_to_grid, create_segy

# cut a small domain from the original Marmousi model {{{
def _cut_marmousi(minz, maxz, minx, maxx, smooth=False, field="velocity_model"):

    from SeismicMesh.sizing.mesh_size_function import write_velocity_model
    import segyio
    import math

    path = "./velocity_models/elastic-marmousi-model/model/"
    if smooth:
        #fname_marmousi = path + "MODEL_P-WAVE_VELOCITY_1.25m.segy.smoothed_sigma=300.segy.hdf5"
        #fname_marmousi = path + "MODEL_S-WAVE_VELOCITY_1.25m.segy.smoothed_sigma=300.segy.hdf5"
        fname_marmousi = path + "MODEL_DENSITY_1.25m.segy.smoothed_sigma=300.segy.hdf5"
    else:
        #fname_marmousi = path + "MODEL_P-WAVE_VELOCITY_1.25m.segy.hdf5"
        #fname_marmousi = path + "MODEL_S-WAVE_VELOCITY_1.25m.segy.hdf5"
        fname_marmousi = path + "MODEL_DENSITY_1.25m.segy.hdf5"

    with h5py.File(fname_marmousi, "r") as f:
        Zo = np.asarray(f.get(field)[()]) # original Marmousi data/domain
        nrow, ncol = Zo.shape
        Lz = 3.5  # original depth of Marmousi model
        Lx = 17.0 # original length of Marmousi model
        zo = np.linspace(-Lz, 0.0, nrow) # original Marmousi data/domain
        xo = np.linspace(0.0,  Lx, ncol) # original Marmousi data/domain
        interpolant = RegularGridInterpolator((xo, zo), np.transpose(Zo))
        #interpolant = RegularGridInterpolator((zo, xo), Zo)

        nrowq = math.ceil( nrow * (maxz-minz) / Lz )
        ncolq = math.ceil( ncol * (maxx-minx) / Lx )
        assert nrowq > 0
        assert ncolq > 0

        zq = np.linspace(minz, maxz, nrowq)
        xq = np.linspace(minx, maxx, ncolq)
        #zq, xq = np.meshgrid(zq, xq)
        xq, zq = np.meshgrid(xq, zq)

        #Zq = interpolant((zq, xq))
        Zq = interpolant((xq, zq))

        if smooth:
            #fname = path + "MODEL_P-WAVE_VELOCITY_1.25m_small_domain_smoothed_sigma=300.segy"
            #fname = path + "MODEL_S-WAVE_VELOCITY_1.25m_small_domain_smoothed_sigma=300.segy"
            fname = path + "MODEL_DENSITY_1.25m_small_domain_smoothed_sigma=300.segy"
        else:
            #fname = path + "MODEL_P-WAVE_VELOCITY_1.25m_small_domain.segy"
            #fname = path + "MODEL_S-WAVE_VELOCITY_1.25m_small_domain.segy"
            fname = path + "MODEL_DENSITY_1.25m_small_domain.segy"

        # save to segy format
        create_segy(Zq, fname)
        # save to hdf5 format
        #write_velocity_model(fname)
        hfname = fname +".hdf5"
        print(f"Writing velocity model: {hfname}", flush=True)
        with h5py.File(hfname, "w") as fh:
            #fh.create_dataset("velocity_model", data=Zq, dtype="f")
            fh.create_dataset("density", data=Zq, dtype="f")
            fh.attrs["shape"] = Zq.shape
            #fh.attrs["units"] = "m/s"
            fh.attrs["units"] = "g/cm3"

    if True: # plot vg? {{{
        with segyio.open(fname, ignore_geometry=True) as f:
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
    #}}}
#}}}

# 4x2 middle of the domain
minz = -2.0
maxz =  0.0
minx = 7.5
maxx = 11.5
# 2x1.5 middle of the domain
#minz = -1.95
#maxz = -0.45
#minx = 8.5
#maxx = 10.5
_cut_marmousi(minz, maxz, minx, maxx, smooth=False)
#_cut_marmousi(minz, maxz, minx, maxx, smooth=True)

