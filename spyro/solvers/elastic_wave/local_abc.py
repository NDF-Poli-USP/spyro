from firedrake import (Constant, ds, ds_b, ds_t, ds_v, TestFunction,
                       TrialFunction)


def _boundary_measures(wave, qr_s):
    """Return coordinate-ordered exterior measures for box meshes."""
    if not getattr(wave.mesh, "extruded", False):
        return tuple(ds(marker, **qr_s) for marker in range(1, 7))

    # The base rectangle is (z, x), with its z coordinate negated by Spyro.
    # Firedrake markers 1/2 are therefore z=max/z=min, while extrusion is y.
    marker_measures = {
        1: ds_v(2, **qr_s),
        2: ds_v(1, **qr_s),
        3: ds_v(3, **qr_s),
        4: ds_v(4, **qr_s),
        # Top/bottom extrusion facets have a different tensor factorization;
        # Firedrake builds the appropriate quadrature for these measures.
        5: ds_b,
        6: ds_t,
    }
    status_keys = {1: 2, 2: 1, 3: 3, 4: 4, 5: "bottom", 6: "top"}
    status = wave.mesh_parameters.boundary_ids_map
    return tuple(
        marker_measures[marker]
        if status.get(status_keys[marker], True) else 0
        for marker in range(1, 7)
    )


def local_abc_form(wave):
    '''
    Returns the linear form associated with the traction loads
    when combined with local absorbing boundary conditions.
    '''
    abc_dict = wave.input_dictionary.get("absorving_boundary_conditions", None)
    if abc_dict is None:
        return 0
    else:
        abc_active = abc_dict.get("status", False)
        if abc_active:
            abc_type = abc_dict.get("nrbc", {}).get("type", "Stacey")
            dt_scheme = abc_dict.get("nrbc", {}).get("dt_scheme", "backward")
        else:
            return 0

    V = wave.function_space
    v = TestFunction(V)
    u_nm1 = wave.u_nm1
    u_n = wave.u_n

    dt = Constant(wave.dt)
    rho = wave.rho
    c_p = wave.c
    c_s = wave.c_s

    qr_s = wave.surface_quadrature_rule
    boundary_measures = _boundary_measures(wave, qr_s)

    # Index of each coordinate
    iz = 0
    ix = 1
    iy = 2

    # Partial derivatives
    if dt_scheme == "backward":
        uz_dt = (u_n[iz] - u_nm1[iz])/dt
        ux_dt = (u_n[ix] - u_nm1[ix])/dt
    elif dt_scheme == "backward_2nd":
        u_nm2 = wave.u_nm2
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
    if wave.dimension == 3:
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

    return callback(wave.dimension, rho, c_p, c_s,
                    v, iz, ix, iy, boundary_measures,
                    uz_dt, ux_dt, uy_dt,
                    uz_dz, ux_dz, uy_dz,
                    uz_dx, ux_dx, uy_dx,
                    uz_dy, ux_dy, uy_dy)


def clayton_engquist_A1_terms(ndim, rho, c_p, c_s,
                              v, iz, ix, iy, boundary_measures,
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
    F_t += -(sig_zz*v[iz] + sig_xz*v[ix])*boundary_measures[0]
    if ndim == 3:
        sig_yz = rho*c_s*uy_dt + rho*(c_s**2)*uz_dy
        F_t += -sig_yz*v[iy]*boundary_measures[0]

    # Plane z = 0
    sig_zz = -rho*c_p*uz_dt + rho*(c_p**2 - 2*c_s**2)*ux_dx
    if ndim == 3:
        sig_zz += rho*(c_p**2 - 2*c_s**2)*uy_dy
    sig_xz = -rho*c_s*ux_dt + rho*(c_s**2)*uz_dx
    F_t += (sig_zz*v[iz] + sig_xz*v[ix])*boundary_measures[1]
    if ndim == 3:
        sig_yz = -rho*c_s*uy_dt + rho*(c_s**2)*uz_dy
        F_t += sig_yz*v[iy]*boundary_measures[1]

    # Plane x = -pad
    sig_zx = rho*c_s*uz_dt + rho*(c_s**2)*ux_dz
    sig_xx = rho*c_p*ux_dt + rho*(c_p**2 - 2*c_s**2)*uz_dz
    if ndim == 3:
        sig_xx += rho*(c_p**2 - 2*c_s**2)*uy_dy
    F_t += -(sig_zx*v[iz] + sig_xx*v[ix])*boundary_measures[2]
    if ndim == 3:
        sig_yx = rho*c_s*uy_dt + rho*(c_s**2)*ux_dy
        F_t += -sig_yx*v[iy]*boundary_measures[2]

    # Plane x = Lx + pad
    sig_zx = -rho*c_s*uz_dt + rho*(c_s**2)*ux_dz
    sig_xx = -rho*c_p*ux_dt + rho*(c_p**2 - 2*c_s**2)*uz_dz
    if ndim == 3:
        sig_xx += rho*(c_p**2 - 2*c_s**2)*uy_dy
    F_t += (sig_zx*v[iz] + sig_xx*v[ix])*boundary_measures[3]
    if ndim == 3:
        sig_yx = -rho*c_s*uy_dt + rho*(c_s**2)*ux_dy
        F_t += sig_yx*v[iy]*boundary_measures[3]

    if ndim == 3:
        # Plane y = 0
        sig_zy = rho*c_s*uz_dt + rho*(c_s**2)*uy_dz
        sig_xy = rho*c_s*ux_dt + rho*(c_s**2)*uy_dx
        sig_yy = rho*c_p*uy_dt + rho*(c_p**2 - 2*c_s**2)*(uz_dz + ux_dx)
        F_t += -(sig_zy*v[iz] + sig_xy*v[ix] + sig_yy*v[iy])*boundary_measures[4]

        # Plane y = L_y + 2*pad
        sig_zy = -rho*c_s*uz_dt + rho*(c_s**2)*uy_dz
        sig_xy = -rho*c_s*ux_dt + rho*(c_s**2)*uy_dx
        sig_yy = -rho*c_p*uy_dt + rho*(c_p**2 - 2*c_s**2)*(uz_dz + ux_dx)
        F_t += (sig_zy*v[iz] + sig_xy*v[ix] + sig_yy*v[iy])*boundary_measures[5]

    return F_t


def stacey_terms(ndim, rho, c_p, c_s,
                 v, iz, ix, iy, boundary_measures,
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
    F_t += -(sig_zz*v[iz] + sig_xz*v[ix])*boundary_measures[0]
    if ndim == 3:
        sig_yz = rho*c_s*uy_dt - rho*c_s*(c_p - 2*c_s)*uz_dy
        F_t += -sig_yz*v[iy]*boundary_measures[0]

    # Plane z = 0
    sig_zz = -rho*c_p*uz_dt + rho*c_s*(c_p - 2*c_s)*ux_dx
    if ndim == 3:
        sig_zz += rho*c_s*(c_p - 2*c_s)*uy_dy
    sig_xz = -rho*c_s*ux_dt - rho*c_s*(c_p - 2*c_s)*uz_dx
    F_t += (sig_zz*v[iz] + sig_xz*v[ix])*boundary_measures[1]
    if ndim == 3:
        sig_yz = -rho*c_s*uy_dt - rho*c_s*(c_p - 2*c_s)*uz_dy
        F_t += sig_yz*v[iy]*boundary_measures[1]

    # Plane x = -pad
    sig_zx = rho*c_s*uz_dt - rho*c_s*(c_p - 2*c_s)*ux_dz
    sig_xx = rho*c_p*ux_dt + rho*c_s*(c_p - 2*c_s)*uz_dz
    if ndim == 3:
        sig_xx += rho*c_s*(c_p - 2*c_s)*uy_dy
    F_t += -(sig_zx*v[iz] + sig_xx*v[ix])*boundary_measures[2]
    if ndim == 3:
        sig_yx = rho*c_s*uy_dt - rho*c_s*(c_p - 2*c_s)*ux_dy
        F_t += -sig_yx*v[iy]*boundary_measures[2]

    # Plane x = Lx + pad
    sig_zx = -rho*c_s*uz_dt - rho*c_s*(c_p - 2*c_s)*ux_dz
    sig_xx = -rho*c_p*ux_dt + rho*c_s*(c_p - 2*c_s)*uz_dz
    if ndim == 3:
        sig_xx += rho*c_s*(c_p - 2*c_s)*uy_dy
    F_t += (sig_zx*v[iz] + sig_xx*v[ix])*boundary_measures[3]
    if ndim == 3:
        sig_yx = -rho*c_s*uy_dt - rho*c_s*(c_p - 2*c_s)*ux_dy
        F_t += sig_yx*v[iy]*boundary_measures[3]

    if ndim == 3:
        # Plane y = 0
        sig_zy = rho*c_s*uz_dt - rho*c_s*(c_p - 2*c_s)*uy_dz
        sig_xy = rho*c_s*ux_dt - rho*c_s*(c_p - 2*c_s)*uy_dx
        sig_yy = rho*c_p*uy_dt + rho*c_s*(c_p - 2*c_s)*(uz_dz + ux_dx)
        F_t += -(sig_zy*v[iz] + sig_xy*v[ix] + sig_yy*v[iy])*boundary_measures[4]

        # Plane y = L_y + 2*pad
        sig_zy = -rho*c_s*uz_dt - rho*c_s*(c_p - 2*c_s)*uy_dz
        sig_xy = -rho*c_s*ux_dt - rho*c_s*(c_p - 2*c_s)*uy_dx
        sig_yy = -rho*c_p*uy_dt + rho*c_s*(c_p - 2*c_s)*(uz_dz + ux_dx)
        F_t += (sig_zy*v[iz] + sig_xy*v[ix] + sig_yy*v[iy])*boundary_measures[5]

    return F_t
