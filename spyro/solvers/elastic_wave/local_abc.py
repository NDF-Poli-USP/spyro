from firedrake import (Constant, ds, TestFunction, TrialFunction)


def local_abc_velocity(wave):
    """Return the finite-difference velocity the local ABCs use with central differences.

    The absorbing boundary conditions of :func:`local_abc_form` act on the
    velocity. With the central-difference time integrator it is
    approximated from the stored displacement levels according to
    ``absorving_boundary_conditions["nrbc"]["dt_scheme"]``: ``"backward"``
    (first order, the default), ``"backward_2nd"`` (second order, needs the
    level ``n-2``) or ``"central"`` (second order, implicit in the unknown
    level ``n+1``, which is the trial function).

    Parameters
    ----------
    wave : `isotropic_wave.IsotropicWave`
        Elastic wave solver holding the displacement levels.

    Returns
    -------
    ufl.core.expr.Expr
        The velocity expression at the current time level.

    Raises
    ------
    NotImplementedError
        If ``dt_scheme`` is not one of the three schemes above.
    """
    abc_dict = wave.input_dictionary.get("absorving_boundary_conditions", {})
    dt_scheme = abc_dict.get("nrbc", {}).get("dt_scheme", "backward")
    u_nm1 = wave.u_nm1
    u_n = wave.u_n
    dt = Constant(wave.dt)
    if dt_scheme == "backward":
        return (u_n - u_nm1)/dt
    elif dt_scheme == "backward_2nd":
        return (3*u_n - 4*u_nm1 + wave.u_nm2)/(2*dt)
    elif dt_scheme == "central":
        u = TrialFunction(wave.function_space)
        return (u - u_nm1)/(2*dt)
    raise NotImplementedError(
        f"Unsupported time discretization: {dt_scheme}")


def local_abc_form(wave, u, u_t):
    '''
    Returns the linear form associated with the traction loads
    when combined with local absorbing boundary conditions.

    Parameters
    ----------
    wave : `isotropic_wave.IsotropicWave`
        Elastic wave solver.
    u : ufl.core.expr.Expr
        Expression standing for the displacement in the spatial derivatives.
    u_t : ufl.core.expr.Expr
        Expression standing for the velocity, see :func:`local_abc_velocity`
        for the central-difference choice.

    Returns
    -------
    ufl.Form or int
        The boundary form, or ``0`` when no local ABC is active.
    '''
    abc_dict = wave.input_dictionary.get("absorving_boundary_conditions", None)
    if abc_dict is None:
        return 0
    else:
        abc_active = abc_dict.get("status", False)
        if abc_active:
            abc_type = abc_dict.get("nrbc", {}).get("type", "Stacey")
        else:
            return 0

    V = wave.function_space
    v = TestFunction(V)

    rho = wave.rho
    c_p = wave.c
    c_s = wave.c_s

    qr_s = wave.surface_quadrature_rule

    # Index of each coordinate
    iz = 0
    ix = 1
    iy = 2

    # Partial derivatives
    uz_dt = u_t[iz]
    ux_dt = u_t[ix]
    uz_dz = u[iz].dx(iz)
    uz_dx = u[iz].dx(ix)
    ux_dz = u[ix].dx(iz)
    ux_dx = u[ix].dx(ix)
    if wave.dimension == 3:
        uy_dt = u_t[iy]
        uz_dy = u[iz].dx(iy)
        ux_dy = u[ix].dx(iy)
        uy_dz = u[iy].dx(iz)
        uy_dx = u[iy].dx(ix)
        uy_dy = u[iy].dx(iy)
    else:
        uy_dt = None
        uz_dy = None
        ux_dy = None
        uy_dz = None
        uy_dx = None
        uy_dy = None

    if abc_type == "Stacey":
        callback = stacey_terms
    elif abc_type == "CE_A1":
        callback = clayton_engquist_A1_terms
    else:
        raise NotImplementedError(f"Unsupported local ABC: {abc_type}")

    return callback(wave.dimension, rho, c_p, c_s,
                    v, iz, ix, iy, qr_s,
                    uz_dt, ux_dt, uy_dt,
                    uz_dz, ux_dz, uy_dz,
                    uz_dx, ux_dx, uy_dx,
                    uz_dy, ux_dy, uy_dy)


def clayton_engquist_A1_terms(ndim, rho, c_p, c_s,
                              v, iz, ix, iy, qr_s,
                              uz_dt, ux_dt, uy_dt,
                              uz_dz, ux_dz, uy_dz,
                              uz_dx, ux_dx, uy_dx,
                              uz_dy, ux_dy, uy_dy):

    F_t = 0

    # Plane z = -(Lz + pad)
    sig_zz = rho*c_p*uz_dt + rho*(c_p**2 - 2*c_s**2)*ux_dx
    if ndim == 3:
        sig_zz += rho*(c_p**2 - 2*c_s**2)*uy_dy
    sig_xz = rho*c_s*ux_dt + rho*(c_s**2)*uz_dx
    F_t += -(sig_zz*v[iz] + sig_xz*v[ix])*ds(1, **qr_s)
    if ndim == 3:
        sig_yz = rho*c_s*uy_dt + rho*(c_s**2)*uz_dy
        F_t += -sig_yz*v[iy]*ds(1, **qr_s)

    # Plane z = 0
    sig_zz = -rho*c_p*uz_dt + rho*(c_p**2 - 2*c_s**2)*ux_dx
    if ndim == 3:
        sig_zz += rho*(c_p**2 - 2*c_s**2)*uy_dy
    sig_xz = -rho*c_s*ux_dt + rho*(c_s**2)*uz_dx
    F_t += (sig_zz*v[iz] + sig_xz*v[ix])*ds(2, **qr_s)
    if ndim == 3:
        sig_yz = -rho*c_s*uy_dt + rho*(c_s**2)*uz_dy
        F_t += sig_yz*v[iy]*ds(2, **qr_s)

    # Plane x = -pad
    sig_zx = rho*c_s*uz_dt + rho*(c_s**2)*ux_dz
    sig_xx = rho*c_p*ux_dt + rho*(c_p**2 - 2*c_s**2)*uz_dz
    if ndim == 3:
        sig_xx += rho*(c_p**2 - 2*c_s**2)*uy_dy
    F_t += -(sig_zx*v[iz] + sig_xx*v[ix])*ds(3, **qr_s)
    if ndim == 3:
        sig_yx = rho*c_s*uy_dt + rho*(c_s**2)*ux_dy
        F_t += -sig_yx*v[iy]*ds(3, **qr_s)

    # Plane x = Lx + pad
    sig_zx = -rho*c_s*uz_dt + rho*(c_s**2)*ux_dz
    sig_xx = -rho*c_p*ux_dt + rho*(c_p**2 - 2*c_s**2)*uz_dz
    if ndim == 3:
        sig_xx += rho*(c_p**2 - 2*c_s**2)*uy_dy
    F_t += (sig_zx*v[iz] + sig_xx*v[ix])*ds(4, **qr_s)
    if ndim == 3:
        sig_yx = -rho*c_s*uy_dt + rho*(c_s**2)*ux_dy
        F_t += sig_yx*v[iy]*ds(4, **qr_s)

    if ndim == 3:
        # Plane y = 0
        sig_zy = rho*c_s*uz_dt + rho*(c_s**2)*uy_dz
        sig_xy = rho*c_s*ux_dt + rho*(c_s**2)*uy_dx
        sig_yy = rho*c_p*uy_dt + rho*(c_p**2 - 2*c_s**2)*(uz_dz + ux_dx)
        F_t += -(sig_zy*v[iz] + sig_xy*v[ix] + sig_yy*v[iy])*ds(5, **qr_s)

        # Plane y = L_y + 2*pad
        sig_zy = -rho*c_s*uz_dt + rho*(c_s**2)*uy_dz
        sig_xy = -rho*c_s*ux_dt + rho*(c_s**2)*uy_dx
        sig_yy = -rho*c_p*uy_dt + rho*(c_p**2 - 2*c_s**2)*(uz_dz + ux_dx)
        F_t += (sig_zy*v[iz] + sig_xy*v[ix] + sig_yy*v[iy])*ds(6, **qr_s)

    return F_t


def stacey_terms(ndim, rho, c_p, c_s,
                 v, iz, ix, iy, qr_s,
                 uz_dt, ux_dt, uy_dt,
                 uz_dz, ux_dz, uy_dz,
                 uz_dx, ux_dx, uy_dx,
                 uz_dy, ux_dy, uy_dy):

    F_t = 0

    # Plane z = -(Lz + pad)
    sig_zz = rho*c_p*uz_dt + rho*c_s*(c_p - 2*c_s)*ux_dx
    if ndim == 3:
        sig_zz += rho*c_s*(c_p - 2*c_s)*uy_dy
    sig_xz = rho*c_s*ux_dt - rho*c_s*(c_p - 2*c_s)*uz_dx
    F_t += -(sig_zz*v[iz] + sig_xz*v[ix])*ds(1, **qr_s)
    if ndim == 3:
        sig_yz = rho*c_s*uy_dt - rho*c_s*(c_p - 2*c_s)*uz_dy
        F_t += -sig_yz*v[iy]*ds(1, **qr_s)

    # Plane z = 0
    sig_zz = -rho*c_p*uz_dt + rho*c_s*(c_p - 2*c_s)*ux_dx
    if ndim == 3:
        sig_zz += rho*c_s*(c_p - 2*c_s)*uy_dy
    sig_xz = -rho*c_s*ux_dt - rho*c_s*(c_p - 2*c_s)*uz_dx
    F_t += (sig_zz*v[iz] + sig_xz*v[ix])*ds(2, **qr_s)
    if ndim == 3:
        sig_yz = -rho*c_s*uy_dt - rho*c_s*(c_p - 2*c_s)*uz_dy
        F_t += sig_yz*v[iy]*ds(2, **qr_s)

    # Plane x = -pad
    sig_zx = rho*c_s*uz_dt - rho*c_s*(c_p - 2*c_s)*ux_dz
    sig_xx = rho*c_p*ux_dt + rho*c_s*(c_p - 2*c_s)*uz_dz
    if ndim == 3:
        sig_xx += rho*c_s*(c_p - 2*c_s)*uy_dy
    F_t += -(sig_zx*v[iz] + sig_xx*v[ix])*ds(3, **qr_s)
    if ndim == 3:
        sig_yx = rho*c_s*uy_dt - rho*c_s*(c_p - 2*c_s)*ux_dy
        F_t += -sig_yx*v[iy]*ds(3, **qr_s)

    # Plane x = Lx + pad
    sig_zx = -rho*c_s*uz_dt - rho*c_s*(c_p - 2*c_s)*ux_dz
    sig_xx = -rho*c_p*ux_dt + rho*c_s*(c_p - 2*c_s)*uz_dz
    if ndim == 3:
        sig_xx += rho*c_s*(c_p - 2*c_s)*uy_dy
    F_t += (sig_zx*v[iz] + sig_xx*v[ix])*ds(4, **qr_s)
    if ndim == 3:
        sig_yx = -rho*c_s*uy_dt - rho*c_s*(c_p - 2*c_s)*ux_dy
        F_t += sig_yx*v[iy]*ds(4, **qr_s)

    if ndim == 3:
        # Plane y = 0
        sig_zy = rho*c_s*uz_dt - rho*c_s*(c_p - 2*c_s)*uy_dz
        sig_xy = rho*c_s*ux_dt - rho*c_s*(c_p - 2*c_s)*uy_dx
        sig_yy = rho*c_p*uy_dt + rho*c_s*(c_p - 2*c_s)*(uz_dz + ux_dx)
        F_t += -(sig_zy*v[iz] + sig_xy*v[ix] + sig_yy*v[iy])*ds(5, **qr_s)

        # Plane y = L_y + 2*pad
        sig_zy = -rho*c_s*uz_dt - rho*c_s*(c_p - 2*c_s)*uy_dz
        sig_xy = -rho*c_s*ux_dt - rho*c_s*(c_p - 2*c_s)*uy_dx
        sig_yy = -rho*c_p*uy_dt + rho*c_s*(c_p - 2*c_s)*(uz_dz + ux_dx)
        F_t += (sig_zy*v[iz] + sig_xy*v[ix] + sig_yy*v[iy])*ds(6, **qr_s)

    return F_t
