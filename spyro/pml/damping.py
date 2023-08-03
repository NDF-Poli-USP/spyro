import math
import warnings

from firedrake import *  # noqa: F403


def functions(Wave_obj):
    """Damping functions for the perfect matched layer for 2D and 3D

    Parameters
    ----------
    Wave_obj : obj
        Wave object with the parameters of the problem

    Returns
    -------
    sigma_x : obj
        Firedrake function with the damping function in the x direction
    sigma_z : obj
        Firedrake function with the damping function in the z direction
    sigma_y : obj
        Firedrake function with the damping function in the y direction

    """
    
    ps = Wave_obj.abc_exponent
    cmax = Wave_obj.abc_cmax # maximum acoustic wave velocity
    R = Wave_obj.abc_R # theoretical reclection coefficient
    pad_length = Wave_obj.abc_pad_length # length of the padding
    V = Wave_obj.function_space
    dimension = Wave_obj.dimension
    z = Wave_obj.mesh_z
    x = Wave_obj.mesh_x
    x1 = 0.0
    x2 = Wave_obj.length_x
    z1 = 0.0
    z2 = -Wave_obj.length_z

    bar_sigma = ((3.0 * cmax) / (2.0 * pad_length)) * math.log10(1.0 / R)
    aux1 = Function(V)
    aux2 = Function(V)


    # Sigma X
    sigma_max_x = bar_sigma  # Max damping
    aux1.interpolate(
        conditional(
            And((x >= x1 - pad_length), x < x1),
            ((abs(x - x1) ** (ps)) / (pad_length ** (ps))) * sigma_max_x,
            0.0,
        )
    )
    aux2.interpolate(
        conditional(
            And(x > x2, (x <= x2 + pad_length)),
            ((abs(x - x2) ** (ps)) / (pad_length ** (ps))) * sigma_max_x,
            0.0,
        )
    )
    sigma_x = Function(V, name="sigma_x").interpolate(aux1 + aux2)

    # Sigma Z
    tol_z = 1.000001
    sigma_max_z = bar_sigma  # Max damping
    aux1.interpolate(
        conditional(
            And(z < z2, (z >= z2 - tol_z * pad_length)),
            ((abs(z - z2) ** (ps)) / (pad_length ** (ps))) * sigma_max_z,
            0.0,
        )
    )

    sigma_z = Function(V, name="sigma_z").interpolate(aux1)

    if dimension == 2:
        return (sigma_x, sigma_z)

    elif dimension == 3:
        # Sigma Y
        sigma_max_y = bar_sigma  # Max damping
        y = Wave_obj.mesh_y
        y1 = 0.0
        y2 = Wave_obj.length_y
        aux1.interpolate(
            conditional(
                And((y >= y1 - pad_length), y < y1),
                ((abs(y - y1) ** (ps)) / (pad_length ** (ps))) * sigma_max_y,
                0.0,
            )
        )
        aux2.interpolate(
            conditional(
                And(y > y2, (y <= y2 + pad_length)),
                ((abs(y - y2) ** (ps)) / (pad_length ** (ps))) * sigma_max_y,
                0.0,
            )
        )
        sigma_y = Function(V, name="sigma_y").interpolate(aux1 + aux2)
        # sgm_y = File("pmlField/sigma_y.pvd")
        # sgm_y.write(sigma_y)

        return (sigma_x, sigma_y, sigma_z)


def matrices_2D(sigma_x, sigma_y):
    """Damping matrices for a two-dimensional problem"""
    Gamma_1 = as_tensor([[sigma_x, 0.0], [0.0, sigma_y]])
    Gamma_2 = as_tensor([[sigma_x - sigma_y, 0.0], [0.0, sigma_y - sigma_x]])

    return (Gamma_1, Gamma_2)


def matrices_3D(sigma_x, sigma_y, sigma_z):
    """Damping matrices for a three-dimensional problem"""
    Gamma_1 = as_tensor(
        [[sigma_x, 0.0, 0.0], [0.0, sigma_y, 0.0], [0.0, 0.0, sigma_z]]
    )
    Gamma_2 = as_tensor(
        [
            [sigma_x - sigma_y - sigma_z, 0.0, 0.0],
            [0.0, sigma_y - sigma_x - sigma_z, 0.0],
            [0.0, 0.0, sigma_z - sigma_x - sigma_y],
        ]
    )
    Gamma_3 = as_tensor(
        [
            [sigma_y * sigma_z, 0.0, 0.0],
            [0.0, sigma_x * sigma_z, 0.0],
            [0.0, 0.0, sigma_x * sigma_y],
        ]
    )

    return (Gamma_1, Gamma_2, Gamma_3)
