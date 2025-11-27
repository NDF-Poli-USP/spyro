import math
import numpy as np
import pytest

from spyro.examples.elastic_analytical import (analytical_solution,
                                               numerical_solution)
from numpy.linalg import norm

def err(u_a, u_n):
    return norm(u_a - u_n)/norm(u_a)


def check_err(u_a, u_n, err_max, norm_max):
    assert (err_max is None) or (norm_max is None)  # Sanity check
    if err_max is None:
        assert math.isclose(norm(u_a), 0.0)
        assert norm(u_n) < norm_max
    else:
        assert err(u_a, u_n) < err_max


@pytest.mark.parametrize("j,err_x_max,err_y_max,err_z_max,norm_x_max,norm_y_max,norm_z_max",[
                         (0,     0.03,     None,     0.03,      None,     2e-12,      None),
                         (1,     None,     0.03,     None,     2e-12,      None,     2e-12),
                         (2,     0.03,     None,     0.03,      None,     2e-12,      None)])
def test_force(j, err_x_max, err_y_max, err_z_max,
               norm_x_max, norm_y_max, norm_z_max):
    U_x = analytical_solution(0, j)
    U_y = analytical_solution(1, j)
    U_z = analytical_solution(2, j)

    u_n = numerical_solution(j)
    u_x = u_n[:, 0]
    u_y = u_n[:, 1]
    u_z = u_n[:, 2]

    check_err(U_x, u_x, err_x_max, norm_x_max)
    check_err(U_y, u_y, err_y_max, norm_y_max)
    check_err(U_z, u_z, err_z_max, norm_z_max)
