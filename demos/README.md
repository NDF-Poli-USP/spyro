These files demonstrate how to perform a synthetic full waveform inversion (FWI) using the forward and adjoint wave propagators developed in spyro.

Some demos require a specific number of MPI processes, as indicated at the beginning of each file. These demos should be run using:

```bash
mpiexec -n X_CORES python3 demoname.py
```

where X_CORES is the required number of MPI processes and demo_name.py is the name of the demo script.

## Literate demos

`elastic_fwi_automated_adjoint/elastic_fwi_automated_adjoint.py.rst` is a
literate demo in the format of the [Firedrake demos](https://www.firedrakeproject.org/demos/full_waveform_inversion.py.html):
a reStructuredText document whose `.. code-block:: python` blocks, read in
order, are the program. It runs an FWI of an isotropic elastic medium with the
automated adjoint, one MPI process per shot. Extract the script with
[pylit](https://pypi.org/project/pylit/) and run it with three processes:

```bash
cd demos/elastic_fwi_automated_adjoint
pylit --code-block-marker ".. code-block:: python" elastic_fwi_automated_adjoint.py.rst
mpiexec -n 3 python3 elastic_fwi_automated_adjoint.py
```

`tests/demo_tests/test_elastic_fwi_automated_adjoint_demo.py` runs the same
code with a coarser mesh, a larger time step and two iterations.
