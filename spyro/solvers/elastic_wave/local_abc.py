from firedrake import (Constant, ds, TestFunction, TrialFunction)


def local_abc_form(Wave):
    '''
    Returns the linear form associated with the traction loads
    when combined with local absorbing boundary conditions.
    '''
    abc_dict = Wave.input_dictionary.get("absorving_boundary_conditions", None)
    if abc_dict is None:
        return 0
    else:
        abc_active = abc_dict.get("status", False)
        if abc_active:
            abc_type = abc_dict.get("local", {}).get("type", "Stacey")
            dt_scheme = abc_dict.get("local", {}).get("dt_scheme", "backward")
        else:
            return 0

    V = Wave.function_space
    v = TestFunction(V)
    u_nm1 = Wave.u_nm1
    u_n = Wave.u_n

    dt = Constant(Wave.dt)
    rho = Wave.rho
    c_p = Wave.c
    c_s = Wave.c_s

    qr_s = Wave.surface_quadrature_rule

    # Index of each coordinate
    iz = 0
    ix = 1
    iy = 2

    # Partial derivatives
    if dt_scheme == "backward":
        uz_dt = (u_n[iz] - u_nm1[iz])/dt
        ux_dt = (u_n[ix] - u_nm1[ix])/dt
    elif dt_scheme == "backward_2nd":
        u_nm2 = Wave.u_nm2
        uz_dt = (3*u_n[iz] - 4*u_nm1[iz] + u_nm2[iz])/(2*dt)
        ux_dt = (3*u_n[ix] - 4*u_nm1[ix] + u_nm2[ix])/(2*dt)
    elif dt_scheme == "central":
        u = TrialFunction(V)
        uz_dt = (u[iz] - u_nm1[iz])/(2*dt)
        ux_dt = (u[ix] - u_nm1[ix])/(2*dt)
    else:
        raise NotImplementedError(
            f"Unsupported time discretization: {dt_scheme}")
    uz_dz = u_n[iz].dx(iz)
    uz_dx = u_n[iz].dx(ix)
    ux_dz = u_n[ix].dx(iz)
    ux_dx = u_n[ix].dx(ix)
    if Wave.dimension == 3:
        if dt_scheme == "backward":
            uy_dt = (u_n[iy] - u_nm1[iy])/dt
        elif dt_scheme == "backward_2nd":
            uy_dt = (3*u_n[iy] - 4*u_nm1[iy] + u_nm2[iy])/(2*dt)
        elif dt_scheme == "central":
            uy_dt = (u[iy] - u_nm1[iy])/(2*dt)
        uz_dy = u_n[iz].dx(iy)
        ux_dy = u_n[ix].dx(iy)
        uy_dz = u_n[iy].dx(iz)
        uy_dx = u_n[iy].dx(ix)
        uy_dy = u_n[iy].dx(iy)
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

    return callback(Wave.dimension, rho, c_p, c_s,
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
    F_t += -(sig_zz*v[iz] + sig_xz*v[ix])*ds(1, scheme=qr_s)
    if ndim == 3:
        sig_yz = rho*c_s*uy_dt + rho*(c_s**2)*uz_dy
        F_t += -sig_yz*v[iy]*ds(1, scheme=qr_s)

    # Plane z = 0
    sig_zz = -rho*c_p*uz_dt + rho*(c_p**2 - 2*c_s**2)*ux_dx
    if ndim == 3:
        sig_zz += rho*(c_p**2 - 2*c_s**2)*uy_dy
    sig_xz = -rho*c_s*ux_dt + rho*(c_s**2)*uz_dx
    F_t += (sig_zz*v[iz] + sig_xz*v[ix])*ds(2, scheme=qr_s)
    if ndim == 3:
        sig_yz = -rho*c_s*uy_dt + rho*(c_s**2)*uz_dy
        F_t += sig_yz*v[iy]*ds(2, scheme=qr_s)

    # Plane x = -pad
    sig_zx = rho*c_s*uz_dt + rho*(c_s**2)*ux_dz
    sig_xx = rho*c_p*ux_dt + rho*(c_p**2 - 2*c_s**2)*uz_dz
    if ndim == 3:
        sig_xx += rho*(c_p**2 - 2*c_s**2)*uy_dy
    F_t += -(sig_zx*v[iz] + sig_xx*v[ix])*ds(3, scheme=qr_s)
    if ndim == 3:
        sig_yx = rho*c_s*uy_dt + rho*(c_s**2)*ux_dy
        F_t += -sig_yx*v[iy]*ds(3, scheme=qr_s)

    # Plane x = Lx + pad
    sig_zx = -rho*c_s*uz_dt + rho*(c_s**2)*ux_dz
    sig_xx = -rho*c_p*ux_dt + rho*(c_p**2 - 2*c_s**2)*uz_dz
    if ndim == 3:
        sig_xx += rho*(c_p**2 - 2*c_s**2)*uy_dy
    F_t += (sig_zx*v[iz] + sig_xx*v[ix])*ds(4, scheme=qr_s)
    if ndim == 3:
        sig_yx = -rho*c_s*uy_dt + rho*(c_s**2)*ux_dy
        F_t += sig_yx*v[iy]*ds(4, scheme=qr_s)

    if ndim == 3:
        # Plane y = 0
        sig_zy = rho*c_s*uz_dt + rho*(c_s**2)*uy_dz
        sig_xy = rho*c_s*ux_dt + rho*(c_s**2)*uy_dx
        sig_yy = rho*c_p*uy_dt + rho*(c_p**2 - 2*c_s**2)*(uz_dz + ux_dx)
        F_t += -(sig_zy*v[iz] + sig_xy*v[ix] + sig_yy*v[iy])*ds(5, scheme=qr_s)

        # Plane y = L_y + 2*pad
        sig_zy = -rho*c_s*uz_dt + rho*(c_s**2)*uy_dz
        sig_xy = -rho*c_s*ux_dt + rho*(c_s**2)*uy_dx
        sig_yy = -rho*c_p*uy_dt + rho*(c_p**2 - 2*c_s**2)*(uz_dz + ux_dx)
        F_t += (sig_zy*v[iz] + sig_xy*v[ix] + sig_yy*v[iy])*ds(6, scheme=qr_s)

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
    F_t += -(sig_zz*v[iz] + sig_xz*v[ix])*ds(1, scheme=qr_s)
    if ndim == 3:
        sig_yz = rho*c_s*uy_dt - rho*c_s*(c_p - 2*c_s)*uz_dy
        F_t += -sig_yz*v[iy]*ds(1, scheme=qr_s)

    # Plane z = 0
    sig_zz = -rho*c_p*uz_dt + rho*c_s*(c_p - 2*c_s)*ux_dx
    if ndim == 3:
        sig_zz += rho*c_s*(c_p - 2*c_s)*uy_dy
    sig_xz = -rho*c_s*ux_dt - rho*c_s*(c_p - 2*c_s)*uz_dx
    F_t += (sig_zz*v[iz] + sig_xz*v[ix])*ds(2, scheme=qr_s)
    if ndim == 3:
        sig_yz = -rho*c_s*uy_dt - rho*c_s*(c_p - 2*c_s)*uz_dy
        F_t += sig_yz*v[iy]*ds(2, scheme=qr_s)

    # Plane x = -pad
    sig_zx = rho*c_s*uz_dt - rho*c_s*(c_p - 2*c_s)*ux_dz
    sig_xx = rho*c_p*ux_dt + rho*c_s*(c_p - 2*c_s)*uz_dz
    if ndim == 3:
        sig_xx += rho*c_s*(c_p - 2*c_s)*uy_dy
    F_t += -(sig_zx*v[iz] + sig_xx*v[ix])*ds(3, scheme=qr_s)
    if ndim == 3:
        sig_yx = rho*c_s*uy_dt - rho*c_s*(c_p - 2*c_s)*ux_dy
        F_t += -sig_yx*v[iy]*ds(3, scheme=qr_s)

    # Plane x = Lx + pad
    sig_zx = -rho*c_s*uz_dt - rho*c_s*(c_p - 2*c_s)*ux_dz
    sig_xx = -rho*c_p*ux_dt + rho*c_s*(c_p - 2*c_s)*uz_dz
    if ndim == 3:
        sig_xx += rho*c_s*(c_p - 2*c_s)*uy_dy
    F_t += (sig_zx*v[iz] + sig_xx*v[ix])*ds(4, scheme=qr_s)
    if ndim == 3:
        sig_yx = -rho*c_s*uy_dt - rho*c_s*(c_p - 2*c_s)*ux_dy
        F_t += sig_yx*v[iy]*ds(4, scheme=qr_s)

    if ndim == 3:
        # Plane y = 0
        sig_zy = rho*c_s*uz_dt - rho*c_s*(c_p - 2*c_s)*uy_dz
        sig_xy = rho*c_s*ux_dt - rho*c_s*(c_p - 2*c_s)*uy_dx
        sig_yy = rho*c_p*uy_dt + rho*c_s*(c_p - 2*c_s)*(uz_dz + ux_dx)
        F_t += -(sig_zy*v[iz] + sig_xy*v[ix] + sig_yy*v[iy])*ds(5, scheme=qr_s)

        # Plane y = L_y + 2*pad
        sig_zy = -rho*c_s*uz_dt - rho*c_s*(c_p - 2*c_s)*uy_dz
        sig_xy = -rho*c_s*ux_dt - rho*c_s*(c_p - 2*c_s)*uy_dx
        sig_yy = -rho*c_p*uy_dt + rho*c_s*(c_p - 2*c_s)*(uz_dz + ux_dx)
        F_t += (sig_zy*v[iz] + sig_xy*v[ix] + sig_yy*v[iy])*ds(6, scheme=qr_s)

    return F_t
