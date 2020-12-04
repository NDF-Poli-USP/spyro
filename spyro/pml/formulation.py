from firedrake import *

from . import damping


def acoustic(
    model,
    mesh,
    V,
    W,
    nf,
    RHS,
    LHS,
    u,
    u_prevs,
    v,
    c,
    qr_x,
    qr_s,
    x,
    x1,
    x2,
    a_pml,
    z,
    z1,
    z2,
    c_pml,
    Z=None,
    y=None,
    y1=None,
    y2=None,
    b_pml=None,
):
    """Perfectly Matched Layer

    Reference:
    Kaltenbacher et al. (2013) - A modified and stable version of a perfectly matched
    layer technique for the 3-d second order wave equation in time domain with an
    application to aeroacoustics"""

    method = model["opts"]["method"]
    dimension = model["opts"]["dimension"]
    dt = model["timeaxis"]["dt"]
    outer_bc = model["PML"]["outer_bc"]

    p = TrialFunction(W)  # Trial Finction
    p_prevs = Function(W)
    p_prevs.assign(as_vector((0.0, 0.0)))  # Initial condition
    q = TestFunction(W)  # Test Function

    # 2D formulation
    if dimension == 2:

        (sigma_x, sigma_z) = damping.functions(
            model, V, dimension, x, x1, x2, a_pml, z, z1, z2, c_pml
        )  # damping functions
        (Gamma_1, Gamma_2) = damping.matrices_2D(sigma_z, sigma_x)  # damping matrices

        # Vectorial equation - Integrand
        g = dot(p, q) * dx(rule=qr_x)  # Right-hand side
        l = (
            dot(p_prevs, q) * dx(rule=qr_x)
            + dt * c * c * inner(grad(u_prevs[1]), dot(Gamma_2, q)) * dx(rule=qr_x)
            - dt * inner(dot(Gamma_1, p_prevs), q) * dx(rule=qr_x)
        )  # Left-hand side

        # Scalar equation - Integrand
        add_a = -(sigma_x + sigma_z) * ((u - u_prevs[1]) / dt) * v * dx(
            rule=qr_x
        ) - sigma_x * sigma_z * u_prevs[1] * v * dx(rule=qr_x)
        add_f = +inner(p_prevs, grad(v)) * dx(rule=qr_x)

        if method == "CG":
            # Non-liear form
            if outer_bc == "non-reflective":
                F = -LHS + RHS + dt * dt * (add_a + add_f + nf)
            elif outer_bc == "neumann" or outer_bc == "dirichlet":
                F = -LHS + RHS + dt * dt * (add_a + add_f)
            else:
                raise ValueError("Boundary condition not supported")

        elif method == "DG":
            # Vectorial equation
            n = FacetNormal(mesh)
            add_dg_l = -inner(jump(u_prevs[1], n), avg(c * c * dot(Gamma_2, q))) * dS(
                rule=qr_s
            )
            l = l + dt * add_dg_l  # update with dg term
            # Scalar equation
            add_dg_f = -inner(avg(p_prevs), jump(v, n)) * dS(rule=qr_s)
            # Non-liear form
            # Non-liear form
            if outer_bc == "non-reflective":
                F = -LHS + RHS + dt * dt * (add_a + add_f + nf + add_dg_f)
            elif outer_bc == "neumann" or outer_bc == "dirichlet":
                F = -LHS + RHS + dt * dt * (add_a + add_f + add_dg_f)
            else:
                raise ValueError("Boundary condition not supported")

        return (g, l, p_prevs, F)

    # 3D formulation
    elif dimension == 3:

        omega = TrialFunction(Z)  # Trial Finction
        omega_prevs = Function(Z)
        omega_prevs.assign(0.0)  # Initial condition
        theta = TestFunction(Z)  # Test Function

        (sigma_x, sigma_y, sigma_z) = damping.functions(
            model, V, dimension, x, x1, x2, a_pml, z, z1, z2, c_pml, y, y1, y2, b_pml
        )  # damping functions
        (Gamma_1, Gamma_2, Gamma_3) = damping.matrices_3D(
            sigma_x, sigma_y, sigma_z
        )  # damping matrices

        # Vectorial equation - Integrand - (II)
        g = dot(p, q) * dx(rule=qr_x)  # Right-hand side
        l = (
            dot(p_prevs, q) * dx(rule=qr_x)
            - dt * inner(dot(Gamma_1, p_prevs), q) * dx(rule=qr_x)
            + dt * c * c * inner(grad(u_prevs[1]), dot(Gamma_2, q)) * dx(rule=qr_x)
            - dt * c * c * inner(grad(omega_prevs), dot(Gamma_3, q)) * dx(rule=qr_x)
        )  # Left-hand side

        # Scalar equation - Integrand - (III)
        o = omega * theta * dx
        d = omega_prevs * theta * dx + dt * u_prevs[1] * theta * dx

        # Scalar equation - Integrand - main equation - (I)
        add_a = (
            -(sigma_x + sigma_y + sigma_z) * ((u - u_prevs[1]) / dt) * v * dx(rule=qr_x)
            - (sigma_x * sigma_y + sigma_x * sigma_z + sigma_y * sigma_z)
            * u_prevs[1]
            * v
            * dx(rule=qr_x)
            - (sigma_x * sigma_y * sigma_z) * omega_prevs * v * dx(rule=qr_x)
        )
        add_f = +inner(p_prevs, grad(v)) * dx(rule=qr_x)

        if method == "CG":
            # Non-liear form - main equation - (I)
            if outer_bc == "non-reflective":
                F = -LHS + RHS + dt * dt * (add_a + add_f + nf)
            elif outer_bc == "neumann" or outer_bc == "dirichlet":
                F = -LHS + RHS + dt * dt * (add_a + add_f)
            else:
                raise ValueError("Boundary condition not supported")

        elif method == "DG":
            # Vectorial equation
            n = FacetNormal(mesh)
            add_dg_l = -inner(jump(u_prevs[1], n), avg(c * c * dot(Gamma_2, q))) * dS(
                rule=qr_s
            ) + inner(jump(omega_prevs, n), avg(c * c * dot(Gamma_3, q))) * dS(
                rule=qr_s
            )
            l = l + dt * add_dg_l  # update with the dg term - (II)
            # Scalar equation
            add_dg_f = -inner(avg(p_prevs), jump(v, n)) * dS(rule=qr_s)

            # Non-liear form
            if outer_bc == "non-reflective":
                F = -LHS + RHS + dt * dt * (add_a + add_f + nf + add_dg_f)
            elif outer_bc == "neumann" or outer_bc == "dirichlet":
                F = -LHS + RHS + dt * dt * (add_a + add_f + add_dg_f)
            else:
                raise ValueError("Boundary condition not supported")

        return (g, l, o, d, p_prevs, omega_prevs, F)
