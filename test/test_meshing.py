from spyro.meshing.meshing_functions import cells_per_wavelength
import numpy as np


def test_cpw_for_acoustic():
    method = 'MLT'
    degree = 3
    dimension = 2
    mlt3tri = cells_per_wavelength(method, degree, dimension)
    dimension = 3
    mlt3tet = cells_per_wavelength(method, degree, dimension)
    assert np.isclose(mlt3tri, 3.70) and np.isclose(mlt3tet, 3.72)


if __name__ == "__main__":
    test_cpw_for_acoustic()
