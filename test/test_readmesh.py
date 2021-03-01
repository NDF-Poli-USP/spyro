import numpy as np

import spyro

"""Read in an external mesh and interpolate velocity to it"""


def test_readmesh2():
    from .inputfiles.Model1_2d_CG import model

    comm = spyro.utils.mpi_init(model)

    mesh, V = spyro.io.read_mesh(model, comm)

    vp = spyro.io.interpolate(model, mesh, V)

    assert not np.isnan(np.min(vp.dat.data[:]))


def test_readmesh3():
    from .inputfiles.Model1_3d_CG import model

    comm = spyro.utils.mpi_init(model)

    mesh, V = spyro.io.read_mesh(model, comm)

    vp = spyro.io.interpolate(model, mesh, V)

    assert not np.isnan(np.min(vp.dat.data[:]))


if __name__ == "__main__":
    test_readmesh2()
    test_readmesh3()
