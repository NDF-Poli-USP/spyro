import finat
import firedrake as fire
import numpy as np
import scipy.sparse as ss
import scipy.linalg as la
import ipdb
import spyro.habc.habc as habc

# For theoretical background, see:
# Absorbing Boundary Conditions for Difference Approximations
# to the Multi-Dimensional Wave Equation (Higdon, 1986)
# Absorbing Boundary Conditions for acoustic and elastic
# wave equations (Clayton and Engquist, 1977)
# Experiments with Higdon’s absorbing boundary conditions
# for a number of wave equations (Mulder, 1996): Growing
# of low-frequency modes
# Absorbing Boundary Conditions for Acoustic and
# Elastic Waves in Stratified Media (Higdon, 1990)
# Local absorbing boundary conditions for the elastic wave
# equation (Turkel et al., 2023)

# Dimensions [m]
b = 140
h = 60
t = 40

# Element size [m]
elsize = 10

# Discretization
nelx = int(b / elsize)
nely = int(h / elsize)
nelz = max(int(t / elsize), 2)

# Mesh
mesh = fire.BoxMesh(nelx, nely, nelz, b, h, t)


# Space functions
V = fire.VectorFunctionSpace(mesh, "KMV", 3)
G = fire.TensorFunctionSpace(mesh, "KMV", 3, shape=(3, 3))
H = fire.TensorFunctionSpace(mesh, "KMV", 3, shape=(6, 6))

# Dirichlet BC
# bc = fire.DirichletBC(V, ((0., 0., 0.)), 1)

# Integration domain
# q_degree = 3
# dx = fire.dx(metadata={'quadrature_degree': q_degree})
quad_rule = finat.quadrature.make_quadrature(
    V.finat_element.cell, V.ufl_element().degree(), "KMV")
dx = fire.dx(scheme=quad_rule)
ds = fire.ds()

# Surface tension
# F1 = fire.Constant((0., -0.01, 0.))

# Boundary nodes indices
bnd_nod = fire.DirichletBC(V, ((0., 0., 0.)), "on_boundary").nodes

# Boundary coordinates
coord = fire.Function(V).interpolate(fire.SpatialCoordinate(mesh))

# Define source position
source_position = np.array([b / 2., h / 2., t / 2.])
source_vertex = int(np.linalg.norm(
    coord.dat.data - source_position, axis=1).argmin())


def c_vti_tensor(vP, vS, rho, epsilon, gamma, delta, anysotropy):
    ''''
    Constructs the elastic tensor for a material with VTI anisotropy.
    Reference: Thomsen (1986). Geophysics 51, 10, 1954-1966

    Parameters
    ----------
    vP: `float`
        P-wave velocity [km/s]
    vS: `float`
        S-wave velocity [km/s]
    rho: `float`
        Density [kg/m³]
    epsilon: `float`
        Thomsen parameter epsilon
    gamma: `float`
        Thomsen parameter gamma
    delta: `float`
        Thomsen parameter delta
    anysotropy: `str`
        Type of anisotropy: 'weak' or 'exact'

    Returns
    -------
    C_vti: `ufl.tensors.ListTensor`
        Elastic tensor
    '''

    # Computing the elastic tensor components
    C33 = rho * vP**2
    C11 = C33 * (1. + 2. * epsilon)
    C44 = rho * vS**2
    C66 = C44 * (1. + 2. * gamma)
    C12 = C11 - 2 * C66

    # C13 is calculated based on the type of anisotropy
    dC = C33 - C44
    if anysotropy == 'weak':
        C13 = (delta * C33**2 + 0.5 * dC * (C11 + C33 - 2 * C44))**0.5
    elif anysotropy == 'exact':
        C13 = (dC * (C33 * (1. + 2 * delta) - C44))**0.5
    C13 -= C44

    # Assembling the elastic tensor
    C_vti = fire.as_tensor(((C11, C12, C13, 0, 0, 0),
                            (C12, C11, C13, 0, 0, 0),
                            (C13, C13, C33, 0, 0, 0),
                            (0, 0, 0, C44, 0, 0),
                            (0, 0, 0, 0, C44, 0),
                            (0, 0, 0, 0, 0, C66)))

    return C_vti


def c_tti_tensor(C_vti, theta, phi=0., heterogeneous_field=True):
    '''
    Constructs the elastic tensor for a material with TTI anisotropy.
    References: Yang et al (2020). Survey in Geophysics 41, 805-833

    Parameters
    ----------
    C_vti: `ufl.tensors.ListTensor`
        Elastic tensor for VTI anisotropy
    theta: `float` or `firedrake.Function`
        Tilt angle in degrees
    phi: `float` or `firedrake.Function`, optional
        Azimuth angle in degrees (default is 0: 2D case)
    heterogeneous_field: `bool`, optional
        If True, the input angles are fields. Default is True

    Returns
    -------
    C_tti: `ufl.tensors.ListTensor`
        Elastic tensor for TTI anisotropy
    '''

    # Especial angle values
    c_zero = [np.pi / 2., 3 * np.pi / 2.]
    s_zero = [0., np.pi, 2 * np.pi]

    # Tilt angle
    t = theta * np.pi / 180.

    if isinstance(theta, float):
        ct = 0. if t in c_zero else np.cos(theta)
        st = 0. if t in s_zero else np.sin(theta)
    else:
        ct = fire.cos(t)
        st = fire.sin(t)

    # Azimuth angle
    p = phi * np.pi / 180.

    if isinstance(phi, float):
        cp = 0. if p in c_zero else np.cos(phi)
        sp = 0. if p in s_zero else np.sin(phi)
    else:
        cp = fire.cos(p)
        sp = fire.sin(p)

    # Rotation matrix components
    R11 = ct * cp
    R22 = cp
    R33 = ct
    R12 = -sp
    R13 = st * cp
    R21 = ct * sp
    R23 = st * sp
    R31 = -st
    R32 = 0.

    if heterogeneous_field:

        # Transformation matrix for Voigt notation using UFL
        T = fire.as_tensor([
            [R11**2, R12**2, R13**2, 2 * R12 * R13, 2 * R13 * R11, 2 * R11 * R12],
            [R21**2, R22**2, R23**2, 2 * R22 * R23, 2 * R23 * R21, 2 * R21 * R22],
            [R31**2, R32**2, R33**2, 2 * R32 * R33, 2 * R33 * R31, 2 * R31 * R32],
            [R21*R31, R22*R32, R23*R33, R22*R33 + R23*R32, R21*R33 + R23*R31,
             R22*R31 + R21*R32],
            [R31*R11, R32*R12, R33*R13, R12*R33 + R13*R32, R13*R31 + R11*R33,
             R11*R32 + R12*R31],
            [R11*R21, R12*R22, R13*R23, R12*R23 + R13*R22, R13*R21 + R11*R23,
             R11*R22 + R12*R21]
        ])

        # Apply transformation: C_tti = T * C_vti * T^T
        C_tti = fire.dot(fire.dot(T, C_vti), fire.transpose(T))

    else:

        # Convert to array
        C_vti_arr = np.zeros(C_vti.ufl_shape)
        for i in range(C_vti.ufl_shape[0]):
            for j in range(C_vti.ufl_shape[1]):
                C_vti_arr[i, j] = C_vti[i, j]

        # Transformation matrix
        T_arr = np.array((
            (R11**2, R12**2, R13**2,
                2 * R12 * R13, 2 * R13 * R11, 2 * R11 * R12),
            (R21**2, R22**2, R23**2,
                2 * R22 * R23, 2 * R23 * R21, 2 * R21 * R22),
            (R31**2, R32**2, R33**2,
                2 * R32 * R33, 2 * R33 * R31, 2 * R31 * R32),
            (R21*R31, R22*R32, R23*R33,
                R22*R33 + R23*R32, R21*R33 + R23*R31, R22*R31 + R21*R32),
            (R31*R11, R32*R12, R33*R13,
                R12*R33 + R13*R32, R13*R31 + R11*R33, R11*R32 + R12*R31),
            (R11*R21, R12*R22, R13*R23,
                R12*R23 + R13*R22, R13*R21 + R11*R23, R11*R22 + R12*R21)))

        # Apply transformation to the VTI tensor
        C_tti_arr = T_arr * C_vti_arr * T_arr.T
        C_tti = fire.as_tensor(C_tti_arr)

    return C_tti


def strain_tensor(u):
    '''
    Compute the strain tensor in Voight notation

    Parameters
    ----------
    u: `firedrake.Function`
        Displacement field

    Returns
    -------
    epsilon: `ufl.tensors.ListTensor`
        Strain tensor in Voight notation
    '''
    # Components
    eps_x = u[0].dx(0)
    eps_y = u[1].dx(1)
    eps_z = u[2].dx(2)
    gamma_xy = u[0].dx(1) + u[1].dx(0)
    gamma_yz = u[1].dx(2) + u[2].dx(1)
    gamma_xz = u[0].dx(2) + u[2].dx(0)

    # Assembling the strain
    epsilon = fire.as_vector((eps_x, eps_y, eps_z,
                              gamma_yz, gamma_xz, gamma_xy))

    return epsilon


def nrbc_tensor(rho, Gamma, bnd_nod, G, V):
    ''''
    Constructs the NRBC tensor for the elastic ABC.

    Parameters
    ----------
    rho: `float`
        Density [kg/m³]
    Gamma: `ufl.tensors.ListTensor`
        Crhistoffel tensor
    bnd_nod : `array`
        Boundary nodes indices.
    G : `firedrake.TensorFunctionSpace`
        Function space for the Crhistoffel tensor
    V: `firedrake.VectorFunctionSpace`
        Function space for the displacement field

    Returns
    -------
    nrbc_tensor: `ufl.tensors.ListTensor`
        NRBC elastic tensor
    '''

    Gamma_func = fire.interpolate(Gamma, G)
    V_app = fire.Function(V, name='Apparent_Velocity')

    for idx in bnd_nod:
        Chris_mat = Gamma_func.dat.data[idx]
        Chris_mat = (Chris_mat + Chris_mat.T) / 2.
        # rhot_mat = rho.dat.data[idx] * np.eye(3)
        eigenvalues = abs(la.eig(Chris_mat)[0])
        vel_app = np.sqrt(eigenvalues / rho.dat.data[idx])
        V_app.dat.data[idx] = vel_app

    # # Computing the NRBC tensor components
    # cos_theta = fire.sqrt(1 - (vS / vP)**2)

    # Assembling the elastic NRBC tensor
    Z_nrbc = fire.as_tensor(((V_app.sub(0), 0., 0.),
                             (0., V_app.sub(1), 0.),
                             (0., 0., V_app.sub(2))))

    return Z_nrbc


def Crhistoffel_VTI(vP, vS, rho, epsilon, gamma, delta,
                    anysotropy, propag_vector):
    ''''
    Constructs the Crhistoffel tensor for a material with VTI anisotropy.

    Parameters
    ----------
    vP: `float`
        P-wave velocity [km/s]
    vS: `float`
        S-wave velocity [km/s]
    rho: `float`
        Density [kg/m³]
    epsilon: `float`
        Thomsen parameter epsilon
    gamma: `float`
        Thomsen parameter gamma
    delta: `float`
        Thomsen parameter delta
    anysotropy: `str`
        Type of anisotropy: 'weak' or 'exact'
    propag_vector: `list` or `np.array`
        Propagation vector

    Returns
    -------
    Gamma_vti: `ufl.tensors.ListTensor`
        Crhistoffel tensor for VTI anisotropy
    '''

    # Computing the elastic tensor components
    C33 = rho * vP**2
    C11 = C33 * (1. + 2. * epsilon)
    C44 = rho * vS**2
    C66 = C44 * (1. + 2. * gamma)
    C12 = C11 - 2 * C66

    # C13 is calculated based on the type of anisotropy
    dC = C33 - C44
    if anysotropy == 'weak':
        C13 = (delta * C33**2 + 0.5 * dC * (C11 + C33 - 2 * C44))**0.5
    elif anysotropy == 'exact':
        C13 = (dC * (C33 * (1. + 2 * delta) - C44))**0.5
    C13 -= C44

    n1, n2, n3 = propag_vector

    G11 = C11 * n1**2 + C66 * n2**2 + C44 * n3**2
    G22 = C66 * n1**2 + C11 * n2**2 + C44 * n3**2
    G33 = C44 * (n1**2 + n2**2) + C33 * n3**2
    G12 = (C12 + C66) * n1 * n2
    G13 = (C13 + C44) * n1 * n3
    G23 = (C13 + C44) * n2 * n3

    # Assembling the elastic tensor
    Gamma_vti = fire.as_tensor(((G11, G12, G13),
                                (G12, G22, G13),
                                (G13, G13, G33)))

    return Gamma_vti


def Crhistoffel_TTI(C_tti, propag_vector, H):
    ''''
    Constructs the Crhistoffel tensor for a material with TTI anisotropy.
    Carcione (2022). Wave Fields in Real Media: Anisotropic elastic media

    Parameters
    ----------
    propag_vector: `list` or `np.array`
        Propagation vector

    Returns
    -------
    Gamma_tti: `ufl.tensors.ListTensor`
        Crhistoffel tensor for TTI anisotropy
    '''

    # Computing the elastic tensor components
    C_tti_func = fire.interpolate(C_tti, H)

    C11, C12, C13, C14, C15, C16 = C_tti_func[0, :]
    _, C22, C23, C24, C25, C26 = C_tti_func[1, :]
    _, _, C33, C34, C35, C36 = C_tti_func[2, :]
    _, _, _, C44, C45, C46 = C_tti_func[3, :]
    C55, C56 = C_tti_func[4, 4], C_tti_func[4, 5]
    C66 = C_tti_func[5, 5]

    n1, n2, n3 = propag_vector

    G11 = C11 * n1**2 + C66 * n2**2 + C55 * n3**2 + \
        2 * C56 * n2 * n3 + 2 * C15 * n3 * n1 + 2 * C16 * n1 * n2
    G22 = C66 * n1**2 + C22 * n2**2 + C44 * n3**2 + \
        2 * C24 * n2 * n3 + 2 * C46 * n3 * n1 + 2 * C26 * n1 * n2
    G33 = C55 * n1**2 + C44 * n2**2 + C33 * n3**2 + \
        2 * C34 * n2 * n3 + 2 * C35 * n3 * n1 + 2 * C45 * n1 * n2
    G12 = C16 * n1**2 + C26 * n2**2 + C45 * n3**2 + \
        (C46 + C25) * n2 * n3 + (C14 + C56) * n3 * n1 + (C12 + C66) * n1 * n2
    G13 = C15 * n1**2 + C46 * n2**2 + C35 * n3**2 + \
        (C45 + C36) * n2 * n3 + (C13 + C55) * n3 * n1 + (C14 + C56) * n1 * n2
    G23 = C56 * n1**2 + C24 * n2**2 + C34 * n3**2 + \
        (C44 + C23) * n2 * n3 + (C36 + C45) * n3 * n1 + (C25 + C46) * n1 * n2
    # Assembling the elastic tensor
    Gamma_tti = fire.as_tensor(((G11, G12, G13),
                                (G12, G22, G13),
                                (G13, G13, G33)))

    return Gamma_tti


def propag_vector(coord, source_coord, bnd_nod, V):
    '''
    Compute a unitary reference vector from the source to a boundary point

    Parameters
    ----------
    coord : `Firedrake function`
        Mesh coordinates
    source_coord : `Firedrake function`
        Source coordinates
    bnd_nod : `array`
        Boundary nodes indices.
    V : `Firedrake function space`

    Returns
    -------
    unit_propag_vct : `array`
        Unit reference vector from the source to a boundary point
    '''

    # Source coordinates
    psoux, psouy, psouz = source_coord

    # Unitary vector pointing to the boundary point
    unit_propag_vct = fire.Function(V, name='Propagation_Vector')
    prop = coord.dat.data[bnd_nod] - source_position
    unit_propag_vct.dat.data[bnd_nod] = prop / np.linalg.norm(
        prop, axis=1)[:, np.newaxis]

    return unit_propag_vct


# Data
vP_o = 1500  # P-wave velocity [m/s]
vS_o = 750  # S-wave velocity [m/s]
rho_o = 1e3  # Density [kg/m³]
eps1_o = 0.2  # Thomsen parameter epsilon
gamma_o = 0.3  # Thomsen parameter gamma
delta_o = 0.1  # Thomsen parameter delta
theta_o = 30.  # Tilt angle in degrees
phi_o = 0.  # azimuth angle in degrees (phi = 0: 2D case)

# Create fields
# W = fire.FunctionSpace(mesh, "CG", 1)
W = fire.FunctionSpace(mesh, "KMV", 3)
vP = fire.Function(W)
vP.dat.data[:] = np.random.uniform(1.5e3, 2e3, vP.dat.data.shape)
# vP.assign(vP_o)
vS = fire.Function(W)
vS.dat.data[:] = np.random.uniform(750, 1e3, vS.dat.data.shape)
# vS.assign(vS_o)
rho = fire.Function(W)
rho.dat.data[:] = np.random.uniform(1e3, 2e3, rho.dat.data.shape)
# rho.assign(rho_o)
eps1 = fire.Function(W)
eps1.dat.data[:] = np.random.uniform(0.1, 0.3, eps1.dat.data.shape)
# eps1.assign(eps1_o)
gamma = fire.Function(W)
gamma.dat.data[:] = np.random.uniform(0.2, 0.4, gamma.dat.data.shape)
# gamma.assign(gamma_o)
delta = fire.Function(W)
delta.dat.data[:] = np.random.uniform(-0.1, 0.2, delta.dat.data.shape)
# delta.assign(delta_o)
theta = fire.Function(W)
theta.dat.data[:] = np.random.uniform(-60, 60, theta.dat.data.shape)
# theta.assign(theta_o)
phi = fire.Function(W)
# phi.dat.data[:] = np.random.uniform(-180, 180, phi.dat.data.shape)
phi.assign(phi_o)

# print("VTI Static Problem")
# static_VTI(V)
# print("TTI Static Problem")
# static_TTI(V)


def explosive_source(mesh, source_coord, W, sigma=15.0):
    '''
    Create properly normalized explosive source

    Parameters
    ----------
    mesh: `firedrake.Mesh`
        3D mesh
    source_coord: `list` or `np.array`
        Source coordinates
    W: `firedrake.FunctionSpace`
        Function space for the source
    sigma: `float`, optional
        Standard deviation of the Gaussian (default is 20.0)
    '''
    F1 = fire.Function(W, name='F')
    x, y, z = fire.SpatialCoordinate(mesh)
    xs, ys, zs = source_coord

    # Create Gaussian
    gaussian = fire.exp(-((x - xs)**2 + (y - ys)**2
                          + (z - zs)**2) / (2 * sigma**2))
    F1.interpolate(gaussian)

    # Normalization
    source_integral = fire.assemble(F1 * dx)
    F1.assign(F1 / source_integral)

    # Verify
    # final_integral = fire.assemble(F1 * dx)
    # assert abs(final_integral - 1.0) < 1e-10,\
    #     f"Normalization failed: {final_integral}"

    return F1


def apply_source(t, f0, v, F1, F_sou=1.):
    '''
    Ricker source for time t

    Parameters
    ----------
    t: `float`
        Current time
    f0: `float`
        Source central frequency
    v: `firedrake.TestFunction`
        Test function
    F1: `firedrake.Function`
        Source spatial distribution
    F_sou: `float`, optional
        Maximum source amplitude. Default is 1
    '''

    # Amplitude excitation
    r = (np.pi * f0 * (t - 1./f0))**2
    amp = (1. - 2. * r) * np.exp(-r) * F_sou
    # print('Amp: {:3.2e}'.format(amp))

    # Traction force
    # F1 = fire.Constant((0., amp, 0.))
    # L = fire.inner(F1, v) * fire.ds(2)

    # Point load
    # F1.dat.data[source_vertex] = [0, amp, 0.]  # Point source
    # L = fire.inner(F1, v) * dx  # Source term

    # Explosion source
    L = -fire.Constant(amp) * F1 * fire.div(v) * dx  # Source term
    # print(fire.assemble(L))

    return L


def propagation_VTI(V, G, W):
    '''
    Algorithm to solve the forward problem

    Parameters
    ----------
    V: `firedrake.VectorFunctionSpace`
        Function space for the displacement field
    G : `firedrake.TensorFunctionSpace`
        Function space for the Crhistoffel tensor
    W : `firedrake.FunctionSpace`
        Function space for the scalar fields
    '''
    # Final Time
    T = 2.  # s

    # Number of timesteps
    steps = 200

    # Timestep size
    dt = round(T / steps, 6)

    # Trial and test functions
    du = fire.TrialFunction(V)
    v = fire.TestFunction(V)

    # State variable functions
    u = fire.Function(V, name='u')
    u_ant1 = fire.Function(V, name='u_ant1')
    u_ant2 = fire.Function(V, name='u_ant2')

    # Load function
    # F1 = fire.Function(V, name='F')
    F1 = explosive_source(mesh, source_position, W)

    # Strain tensor
    epsilon = strain_tensor(du)

    # Virtual strain tensor
    epsilon_v = strain_tensor(v)

    # VTI elastic tensor
    C_vti = c_vti_tensor(vP, vS, rho, eps1,
                         gamma, delta, 'weak')

    # Cauchy's stress tensor
    sigma = C_vti * epsilon

    # Bilinear form
    a = fire.inner(sigma, epsilon_v) * dx

    # Propagation vector
    unit_propag_vct = propag_vector(coord, source_position, bnd_nod, V)

    # Crhistoffel tensor
    Gamma_vti = Crhistoffel_VTI(vP, vS, rho, eps1, gamma,
                                delta, 'weak', unit_propag_vct)

    # NRBC tensor
    Z_nrbc = nrbc_tensor(rho, Gamma_vti, bnd_nod, G, V)

    # Current time we are solving for
    t = 0.

    # Number of timesteps solved
    ntimestep = 0

    # Save results
    outfile = fire.VTKFile("nrbc_vti/disp_VTI.pvd")
    # Time integration loop
    print('Solving Forward Problem')
    print(67*'*')
    while True:
        # Update the time it is solving for
        print('Step: {:1d} of {:1d} - Time: {:1.4f} ms'.format(
            ntimestep, steps, t))

        # Ricker source for time t
        L = apply_source(t, 5., v, F1, F_sou=1e8)

        # Variational problem
        m = fire.Constant(1. / dt**2) * rho * fire.inner(
            du - 2 * u_ant1 + u_ant2, v) * dx
        R = fire.inner(Z_nrbc * (du - u_ant1), v) * ds

        # Complete weak form
        form = m + a - L + R
        # fire.solve(fire.lhs(form) == fire.rhs(form), u, bcs=bc)
        fire.solve(fire.lhs(form) == fire.rhs(form), u)
        outfile.write(u, time=t)

        t = round(t + dt, 6)
        ntimestep += 1

        # Cycling the variables
        print(u.dat.data.max())
        u_ant2.assign(u_ant1)
        u_ant1.assign(u)

        if t > T:
            break

    print(67*'*')


def propagation_TTI(V, G, H, W):
    '''
    Algorithm to solve the forward problem

    Parameters
    ----------
    V: `firedrake.VectorFunctionSpace`
        Function space for the displacement field
    G : `firedrake.TensorFunctionSpace`
        Function space for the Crhistoffel tensor
    H : `firedrake.TensorFunctionSpace`
        Function space for the TTI elastic tensor
    W : `firedrake.FunctionSpace`
        Function space for the scalar fields
    '''
    # Final Time
    T = 2.  # s

    # Number of timesteps
    steps = 200

    # Timestep size
    dt = round(T / steps, 6)

    # Trial and test functions
    du = fire.TrialFunction(V)
    v = fire.TestFunction(V)

    # State variable functions
    u = fire.Function(V, name='u')
    u_ant1 = fire.Function(V, name='u_ant1')
    u_ant2 = fire.Function(V, name='u_ant2')

    # Load function
    # F1 = fire.Function(V, name='F')
    F1 = explosive_source(mesh, source_position, W)

    # Strain tensor
    epsilon = strain_tensor(du)

    # Virtual strain tensor
    epsilon_v = strain_tensor(v)

    # VTI elastic tensor
    C_vti = c_vti_tensor(vP, vS, rho, eps1,
                         gamma, delta, 'weak')

    # TTI elastic tensor
    C_tti = c_tti_tensor(C_vti, theta, phi)

    # Cauchy's stress tensor
    sigma = C_tti * epsilon

    # Bilinear form
    a = fire.inner(sigma, epsilon_v) * dx

    # Propagation vector
    unit_propag_vct = propag_vector(coord, source_position, bnd_nod, V)

    # Crhistoffel tensor
    Gamma_tti = Crhistoffel_TTI(C_tti, unit_propag_vct, H)

    # NRBC tensor
    Z_nrbc = nrbc_tensor(rho, Gamma_tti, bnd_nod, G, V)

    # Current time we are solving for
    t = 0.

    # Number of timesteps solved
    ntimestep = 0

    # Save results
    outfile = fire.VTKFile("nrbc_tti/disp_TTI.pvd")
    # Time integration loop
    print('Solving Forward Problem')
    print(67*'*')
    while True:
        # Update the time it is solving for
        print('Step: {:1d} of {:1d} - Time: {:1.4f} ms'.format(
            ntimestep, steps, t))

        # Ricker source for time t
        L = apply_source(t, 5., v, F1, F_sou=1e6)

        # Variational problem
        m = fire.Constant(1. / dt**2) * rho * fire.inner(
            du - 2 * u_ant1 + u_ant2, v) * dx
        R = fire.inner(Z_nrbc * (du - u_ant1), v) * ds

        # Complete weak form
        form = m + a - L + R
        # fire.solve(fire.lhs(form) == fire.rhs(form), u, bcs=bc)
        fire.solve(fire.lhs(form) == fire.rhs(form), u)
        outfile.write(u, time=t)

        t = round(t + dt, 6)
        ntimestep += 1

        # Cycling the variables
        print(u.dat.data.max())
        u_ant2.assign(u_ant1)
        u_ant1.assign(u)

        if t > T:
            break

    print(67*'*')


# print("VTI Propagation Problem")
# propagation_VTI(V, G, W)
print("TTI Propagation Problem")
propagation_TTI(V, G, H, W)


# #VertexOnly functions
# def delta(time_expr, v, mesh, source_locations):
#     """Creates a point source using VertexOnlyMesh approach."""
#     vom = fire.VertexOnlyMesh(mesh, source_locations)
#     if v.ufl_shape == ():
#         # Scalar function space
#         P0 = fire.FunctionSpace(vom, "DG", 0)
#         Fvom = fire.Cofunction(P0.dual()).assign(1)
#     else:
#         # Vector function space
#         P0_vec = fire.VectorFunctionSpace(vom, "DG", 0)
#         Fvom = fire.Cofunction(P0_vec.dual())
#         Fvom_x = Fvom.sub(1)
#         Fvom_x.assign(1)
#     return fire.interpolate(time_expr * v, Fvom)
# ricker = RickerWavelet(t, freq, amp=1.0)
# F_s = delta(ricker, v, mesh, [source_location])
# receiver_evaluator = fire.PointEvaluator(mesh, [receptor_coords])
