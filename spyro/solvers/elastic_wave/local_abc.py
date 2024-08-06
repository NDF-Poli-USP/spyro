from firedrake import (Constant, ds, TestFunction)

def clayton_engquist_A1(wave):
    '''
    Returns the linear form associated with the traction loads 
    when combined with the Clayton-Engquist A1 relations.
    '''
    F_t = 0 # linear form

    V = wave.function_space
    v = TestFunction(V)
    u_nm1 = wave.u_nm1
    u_n = wave.u_n

    dt = Constant(wave.dt)
    rho = wave.rho
    c_p = wave.c
    c_s = wave.c_s

    qr_s = wave.surface_quadrature_rule

    # Index of each coordinate
    iz = 0; ix = 1; iy = 2

    # Partial derivatives
    uz_dt = (u_n[iz] - u_nm1[iz])/dt
    ux_dt = (u_n[ix] - u_nm1[ix])/dt
    uz_dz = u_n[iz].dx(iz)
    uz_dx = u_n[iz].dx(ix)
    ux_dz = u_n[ix].dx(iz)
    ux_dx = u_n[ix].dx(ix)
    if wave.dimension == 3:
        uy_dt = (u_n[iy] - u_nm1[iy])/dt
        uz_dy = u_n[iz].dx(iy)
        ux_dy = u_n[ix].dx(iy)
        uy_dz = u_n[iy].dx(iz)
        uy_dx = u_n[iy].dx(ix)
        uy_dy = u_n[iy].dx(iy)

    # Plane z = -(Lz + pad)
    sig_zz = rho*c_p*uz_dt + rho*(c_p**2 - 2*c_s**2)*ux_dx
    sig_xz = rho*c_s*ux_dt + rho*(c_s**2)*uz_dx
    F_t += -(sig_zz*v[iz] + sig_xz*v[ix])*ds(1, scheme=qr_s)
    
    # Plane z = 0
    sig_zz = -rho*c_p*uz_dt + rho*(c_p**2 - 2*c_s**2)*ux_dx
    sig_xz = -rho*c_s*ux_dt + rho*(c_s**2)*uz_dx
    F_t += (sig_zz*v[iz] + sig_xz*v[ix])*ds(2, scheme=qr_s)

    # Plane x = -pad
    sig_zx = rho*c_s*uz_dt + rho*(c_s**2)*ux_dz
    sig_xx = rho*c_p*ux_dt + rho*(c_p**2 - 2*c_s**2)*uz_dz
    F_t += -(sig_zx*v[iz] + sig_xx*v[ix])*ds(3, scheme=qr_s)

    # Plane x = Lx + pad
    sig_zx = -rho*c_s*uz_dt + rho*(c_s**2)*ux_dz
    sig_xx = -rho*c_p*ux_dt + rho*(c_p**2 - 2*c_s**2)*uz_dz
    F_t += (sig_zx*v[iz] + sig_xx*v[ix])*ds(4, scheme=qr_s)

    return F_t