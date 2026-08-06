These files demonstrate how to perform a synthetic full waveform inversion (FWI) using the forward and adjoint wave propagators developed in spyro.

Some demos require a specific number of MPI processes, as indicated at the beginning of each file. These demos should be run using:

```bash
mpiexec -n X_CORES python3 demoname.py
```

where X_CORES is the required number of MPI processes and demo_name.py is the name of the demo script.