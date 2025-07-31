import os
import pytest

import numpy as np

from firedrake import *

import spyro

from .inputfiles.Model1_parallel_2d import model as options


# forward = spyro.solvers.forward
# gradient = spyro.solvers.gradient
functional = spyro.utils.compute_functional


@pytest.mark.skip(reason="no way of currently testing this")
def test_parallel_source():
    comm = spyro.utils.mpi_init(options)

    mesh, V = spyro.io.read_mesh(options, comm)

    vp = Function(V).assign(1.0)

    sources = spyro.Sources(options, mesh, V, comm)

    receivers = spyro.Receivers(options, mesh, V, comm)

    wavelet = spyro.full_ricker_wavelet(
        options["timeaxis"]["dt"],
        options["timeaxis"]["tf"],
        options["acquisition"]["frequency"],
    )

    f, r = forward(
        options,
        mesh,
        comm,
        vp,
        sources,
        wavelet,
        receivers,
    )

    # print(np.amax(np.abs(r)))
    # spyro.io.save_shots('serial_shot.dat', r)
    r_s = spyro.io.load_shots(os.getcwd() + "/tests/on_one_core/serial_shot.dat")
    assert np.amax(np.abs(r - r_s)) < 1e-16


if __name__ == "__main__":
    test_parallel_source()
