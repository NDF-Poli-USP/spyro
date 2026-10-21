These files demonstrate how to perform a synthetic full waveform inversion (FWI) using the forward and adjoint wave propagators developed in spyro.

Some demos require a specific number of MPI processes, as indicated at the beginning of each file. These demos should be run using:

```bash
mpiexec -n X_CORES python3 demoname.py
```

where X_CORES is the required number of MPI processes and demo_name.py is the name of the demo script.

## Literate demos

[Elastic FWI with the automated adjoint](elastic_fwi_automated_adjoint/elastic_fwi_automated_adjoint.py.rst)
runs an FWI of an isotropic elastic medium with the
automated adjoint, recovering P- and S-wave velocities with density fixed.
It covers synthetic observations, a Taylor test, the joint inversion of
both velocities and an exercise inverting them in stages. We recommend the
latest Firedrake release.
Extract the script with
[pylit](https://pypi.org/project/pylit/) and run it with three processes:

```bash
cd demos/elastic_fwi_automated_adjoint
pylit --code-block-marker ".. code-block:: python" elastic_fwi_automated_adjoint.py.rst
mpiexec -n 3 python3 elastic_fwi_automated_adjoint.py
```

## Cells-per-wavelength calculator

[acoustic_homogeneous_cpw.py](acoustic_homogeneous_cpw.py) finds the
minimum cells-per-wavelength of a discretization for the 2D homogeneous
acoustic case, against the analytical solution. Edit the arguments of
`run()` to evaluate another method or degree:

```bash
python3 demos/acoustic_homogeneous_cpw.py
```

[elastic_homogeneous_cpw.py](elastic_homogeneous_cpw.py) does the same
for the 2D homogeneous isotropic elastic case, sizing the mesh by the
S-wavelength and comparing against the analytical point-force solution:

```bash
python3 demos/elastic_homogeneous_cpw.py
```

[acoustic_marmousi2_cpw.py](acoustic_marmousi2_cpw.py) finds the minimum
cells-per-wavelength of a discretization on the Marmousi2 model with its
water layer (Martin et al., 2006, doi: 10.1190/1.2172306), against a
numerical reference. A single towed-streamer shot is used: an air gun 6 m
deep and a streamer 8 m deep with receiver groups every 12.5 m at offsets
from 2 km to 10 km. The P-wave velocity SEG-Y is
not distributed with spyro; save it as
`velocity_models/vp_marmousi-ii.segy` first. The run is long, so use MPI:

```bash
mpiexec -n N_CORES python3 demos/acoustic_marmousi2_cpw.py
```
