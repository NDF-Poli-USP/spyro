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
It covers synthetic observations, a Taylor test and the joint inversion of
both velocities. We recommend the latest Firedrake release.
Extract the script with
[pylit](https://pypi.org/project/pylit/) and run it with three processes:

```bash
cd demos/elastic_fwi_automated_adjoint
pylit --code-block-marker ".. code-block:: python" elastic_fwi_automated_adjoint.py.rst
mpiexec -n 3 python3 elastic_fwi_automated_adjoint.py
```
