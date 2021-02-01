import math

from firedrake import *


def functions(
    model,
    V,
    dimension,
    x,
    x1,
    x2,
    a_pml,
    z,
    z1,
    z2,
    c_pml,
    y=None,
    y1=None,
    y2=None,
    b_pml=None,
):
    """ Damping functions for the perfect matched layer for 2D and 3D"""

    damping_type = model["PML"]["damping_type"]
    if damping_type == "polynomial":
        ps = model["PML"]["exponent"]  # polynomial scaling
    cmax = model["PML"]["cmax"]  # maximum acoustic wave velocity
    R = model["PML"]["R"]  # theoretical reclection coefficient

    bar_sigma = ((3.0 * cmax) / (2.0 * a_pml)) * math.log10(1.0 / R)
    aux1 = Function(V)
    aux2 = Function(V)

    # Sigma X
    if damping_type == "polynomial":
        sigma_max_x = bar_sigma  # Max damping
        aux1.interpolate(
            conditional(
                And((x >= x1 - a_pml), x < x1),
                ((abs(x - x1) ** (ps)) / (a_pml ** (ps))) * sigma_max_x,
                0.0,
            )
        )
        aux2.interpolate(
            conditional(
                And(x > x2, (x <= x2 + a_pml)),
                ((abs(x - x2) ** (ps)) / (a_pml ** (ps))) * sigma_max_x,
                0.0,
            )
        )
    elif damping_type == "hyperbolic":
        tol = 1e-2
        aux1.interpolate(
            conditional(
                And((x >= x1 - a_pml + tol), x < x1),
                (cmax / (a_pml - abs(x - x1))),
                0.0,
            )
        )
        aux2.interpolate(
            conditional(
                And(x > x2, (x <= x2 + a_pml - tol)),
                (cmax / (a_pml - abs(x - x2))),
                0.0,
            )
        )
    elif damping_type == "shifted_hyperbolic":
        tol = 1e-2
        aux1.interpolate(
            conditional(
                And((x >= x1 - a_pml + tol), x < x1),
                (cmax / (a_pml - abs(x - x1))) - (cmax / a_pml),
                0.0,
            )
        )
        aux2.interpolate(
            conditional(
                And(x > x2, (x <= x2 + a_pml - tol)),
                (cmax / (a_pml - abs(x - x2))) - (cmax / a_pml),
                0.0,
            )
        )
    sigma_x = Function(V, name="sigma_x").interpolate(aux1 + aux2)

    # Sigma Z
    if damping_type == "polynomial":
        tol_z = 1.000001
        sigma_max_z = bar_sigma  # Max damping
        aux1.interpolate(
            conditional(
                And(z < z2, (z >= z2 - tol_z * c_pml)),
                ((abs(z - z2) ** (ps)) / (c_pml ** (ps))) * sigma_max_z,
                0.0,
            )
        )
    elif damping_type == "hyperbolic":
        aux1.interpolate(
            conditional(
                And(z < z2, (z >= z2 - c_pml + tol)),
                (cmax / (c_pml - abs(z - z2))),
                0.0,
            )
        )
    elif damping_type == "shifted_hyperbolic":
        aux1.interpolate(
            conditional(
                And(z < z2, (z >= z2 - c_pml + tol)),
                (cmax / (a_pml - abs(z - z2))) - (cmax / a_pml),
                0.0,
            )
        )
    sigma_z = Function(V, name="sigma_z").interpolate(aux1)

    # sgm_x = File("pmlField/sigma_x.pvd")  # , target_degree=1, target_continuity=H1
    # sgm_x.write(sigma_x)
    # sgm_z = File("pmlField/sigma_z.pvd")
    # sgm_z.write(sigma_z)

    if dimension == 2:

        return (sigma_x, sigma_z)

    elif dimension == 3:
        # Sigma Y
        if damping_type == "polynomial":
            sigma_max_y = bar_sigma  # Max damping
            aux1.interpolate(
                conditional(
                    And((y >= y1 - b_pml), y < y1),
                    ((abs(y - y1) ** (ps)) / (b_pml ** (ps))) * sigma_max_y,
                    0.0,
                )
            )
            aux2.interpolate(
                conditional(
                    And(y > y2, (y <= y2 + b_pml)),
                    ((abs(y - y2) ** (ps)) / (b_pml ** (ps))) * sigma_max_y,
                    0.0,
                )
            )
        elif damping_type == "hyperbolic":
            aux1.interpolate(
                conditional(
                    And((y >= y1 - b_pml + tol), y < y1),
                    (cmax / (b_pml - abs(y - y1))),
                    0.0,
                )
            )
            aux2.interpolate(
                conditional(
                    And(y > y2, (y <= y2 + b_pml - tol)),
                    (cmax / (b_pml - abs(y - y2))),
                    0.0,
                )
            )
        elif damping_type == "shifted_hyperbolic":
            aux1.interpolate(
                conditional(
                    And((y >= y1 - b_pml + tol), y < y1),
                    (cmax / (b_pml - abs(y - y1))) - (cmax / b_pml),
                    0.0,
                )
            )
            aux2.interpolate(
                conditional(
                    And(y > y2, (y <= y2 + b_pml - tol)),
                    (cmax / (b_pml - abs(y - y2))) - (cmax / b_pml),
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
    Gamma_1 = as_tensor([[sigma_x, 0.0, 0.0], [0.0, sigma_y, 0.0], [0.0, 0.0, sigma_z]])
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
