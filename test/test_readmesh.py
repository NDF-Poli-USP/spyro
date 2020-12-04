import numpy as np

import Spyro

"""Read in an external mesh and interpolate velocity to it"""

from .inputfiles.Model1_2d_CG import model


def test_readmesh2():

    comm = Spyro.utils.mpi_init(model)

    mesh, V = Spyro.io.read_mesh(model, comm)

    vp = Spyro.io.interpolate(model, mesh, V)

    assert not np.isnan(np.min(vp.dat.data[:]))


if __name__ == "__main__":
    test_readmesh2()
