import os
import sys
import spyro

"""remove program name from argv (pytest complains about it)"""
for i, argv in enumerate(sys.argv):
    if ".py" in argv:
        sys.argv.pop(i)

"""Check save/load functions for model parameters are working"""

original_model = {
    "opts": {
        "method": "CG",
        "variant": "spectral",
        "degree": 1,
        "dimension": 2,
        "cmin": 1,
        "cmax": 3.5,
        "rmin": 1,
        "timestepping": "explicit"
    },
    "mesh": {
        "nz": 60,
        "nx": 60,
        "Lz": 2.0,
        "Lx": 2.0,
        "Ly": 0.0,
        "meshfile": None,
        "truemodel": None
    },
    "PML": {
        "status": False,
        "outer_bc": "reflective",
        "damping_type": "polynomial",
        "exponent": 1,
        "cmax": 4.7,
        "R": 0.001,
        "lz": 0.5,
        "lx": 0.5,
        "ly": 0.0
    },
    "acquisition": {
        "source_type": "Ricker",
        "amplitude": 10.0,
        "num_sources": 1,
        "src_XMIN": 0.999,
        "src_XMAX": 1.001,
        "rec_XMIN": 0.05,
        "rec_XMAX": 1.95,
        "src_depth": [
            1.95
        ],
        "frequency": 2,
        "delay": 1.0,
        "num_receivers": 21,
        "rec_depth": [
            1.95
        ]
    },
    "water": {
        "depth": 1.85,
        "velocity": 1.51
    },
    "timeaxis": {
        "t0": 0.0,
        "tf": 2.6,
        "dt": 0.001,
        "nspool": 5,
        "fspool": 1
    },
    "material": {
        "type": "simp",
        "vp_min": 1,
        "vp_max": 3.5,
        "penal": 1
    },
    "parallelism": {
        "type": "off"
    },
    "cplex": {
        "beta": 0.01,
        "mul_beta": 25,
        "use_rmin": True,
        "mul_rmin": 30,
        "lim_rmin": 50,
	"gamma_m": 0.9,
	"gamma_v": 0.9
    },
    "inversion": {
        "freq_bands": [
            None
        ],
        "optimizer": "cplex"
    },
    "data": {
        "shots": "shots/ls_simple",
        "record": "record",
        "initfile": None,
        "resultfile": "vp_ls_simple_3_d_cplex.hdf5"
    },
    "output": {
        "outdir": "output_fig3_d"
    }
}

def test_json_io(tmpdir):

    # Save original model
    p = tmpdir.mkdir("sub")
    jsonfile = os.path.join(p, "model.json")
    spyro.io.save_model(original_model, jsonfile=jsonfile)
    # Load it
    loaded_model = spyro.io.load_model(jsonfile=jsonfile)

    assert original_model == loaded_model
