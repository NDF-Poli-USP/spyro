import numpy as np
import pytest

import spyro

"""Read in an external mesh and interpolate velocity to it"""

def test_readmesh2():
    from .inputfiles.Model1_2d_CG import model as oldmodel
    model = spyro.Wave(dictionary=oldmodel)

    vp = spyro.io.interpolate(model, oldmodel["mesh"]["initmodel"], model.function_space)

    assert not np.isnan(np.min(vp.dat.data[:]))

def test_readmesh3():
    from .inputfiles.Model1_3d_CG import model as oldmodel
    receivers = spyro.create_transect((-0.05, 0.3, 0.5), (-0.05, 0.9, 0.5), 3)
    oldmodel["acquisition"]["receiver_locations"] = receivers
    model = spyro.Wave(dictionary=oldmodel)

    vp = spyro.io.interpolate(model, oldmodel["mesh"]["initmodel"], model.function_space)

    assert not np.isnan(np.min(vp.dat.data[:]))


if __name__ == "__main__":
    test_readmesh2()
    test_readmesh3()
