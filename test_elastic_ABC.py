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

# Largura do domínio [m]
b = 140

# Altura do domínio [m]
h = 60

# Espessura do domínio [m]
t = 40

# Tamanho do elemento [m]
elsize = 10

# Número de elementos em x
nelx = int(b / elsize)

# Número de elementos em y
nely = int(h / elsize)

# Número de elementos em y
nelz = max(int(t / elsize), 2)

# Definir malha do MEF
mesh = fire.BoxMesh(nelx, nely, nelz, b, h, t)

# Espaço para a variável de estado (deslocamento)
# V = fire.VectorFunctionSpace(mesh, "CG", 1)
V = fire.VectorFunctionSpace(mesh, "KMV", 3)

# Criando a condição de contorno de Dirichlet
# bc = fire.DirichletBC(V, ((0., 0., 0.)), 1)

# Criando dx
# q_degree = 3
# dx = fire.dx(metadata={'quadrature_degree': q_degree})
quad_rule = finat.quadrature.make_quadrature(
    V.finat_element.cell, V.ufl_element().degree(), "KMV")
dx = fire.dx(scheme=quad_rule)
ds = fire.ds()

# Força de superfície
# F1 = fire.Constant((0., -0.01, 0.))

# Boundary nodes indices
bnd_nod = fire.DirichletBC(V, ((0., 0., 0.)), "on_boundary").nodes

# Boundary coordinates
coord = fire.Function(V).interpolate(fire.SpatialCoordinate(mesh))

# Define source position
source_position = np.array([b / 2., h / 2., t / 2.])
source_vertex = int(np.linalg.norm(
    coord.dat.data - source_position, axis=1).argmin())

# Space function
G = fire.TensorFunctionSpace(mesh, "KMV", 3, shape=(3, 3))
H = fire.TensorFunctionSpace(mesh, "KMV", 3, shape=(6, 6))


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

    Returns
    -------
    C_tti: `ufl.tensors.ListTensor`
        Elastic tensor for TTI anisotropy
    comp_C_tti: `dict`
        Non-zero components of the elastic tensor
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


def deformacoes(u):
    '''
    Retorna o tensor de deformações em notação de Voight
    '''

    # Componentes do tensor de deformações
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


def problema_direto(V):
    '''
    Resolve o problema direto e retorna u
    '''

    # Criar um trial function (u = N du)
    du = fire.TrialFunction(V)

    # Criar uma função de teste
    v = fire.TestFunction(V)

    # Criar uma função para representar a variável de estado
    u = fire.Function(V, name='displacement')

    # Derivada de du em relação a dx (deformações)
    epsilon = deformacoes(du)

    # Derivada de v em relação a dx
    epsilon_v = deformacoes(v)

    # Data
    vP = 1.5  # P-wave velocity [km/s]
    vS = 0.75  # S-wave velocity [km/s]
    rho = 1e3  # Density [kg/m³]
    eps1 = 0.2  # Thomsen parameter epsilon
    gamma = 0.3  # Thomsen parameter gamma
    delta = 0.1  # Thomsen parameter delta
    theta = 30.  # Tilt angle in degrees
    phi = 0.  # azimuth angle in degrees (phi = 0: 2D case)

    C1_vti = c_vti_tensor(vP, vS, rho, eps1,
                          gamma, delta, 'weak')
    C2_vti = c_vti_tensor(1.5 * vP, vP / 4., 2.5 * rho,
                          0.3, 0.4, 0.2, 'weak')

    C1_tti = c_tti_tensor(C1_vti, theta, phi)
    C2_tti = c_tti_tensor(C2_vti, theta, phi)

    # Generate a simple model
    x = fire.SpatialCoordinate(mesh)
    left_region = fire.conditional(x[0] <= 0.5, 1, 0)

    # Combined expression
    # C = C1_vti * left_region + C2_vti * (1 - left_region)
    C = C1_tti * left_region + C2_tti * (1 - left_region)

    # Tensor de tensões de Cauchy
    sigma = C * epsilon

    # Problema bilinear (Princípio dos trabalhos virtuais)
    a = fire.inner(sigma, epsilon_v) * dx
    L = fire.inner(F1, v) * fire.ds(2)
    form = a - L
    # fire.solve(fire.lhs(form) == fire.rhs(form), u, bcs=bc)
    fire.solve(fire.lhs(form) == fire.rhs(form), u)
    # fire.solve(a == L, u)

    fire.VTKFile("displacement.pvd").write(u)
    # ipdb.set_trace()

    # # Fazer o "assemble" do sistema linear
    # A, b = assemble_system(a, L, bc)


def static_VTI(V):
    '''
    VTI static problem
    '''

    # Criar um trial function (u = N du)
    du = fire.TrialFunction(V)

    # Criar uma função de teste
    v = fire.TestFunction(V)

    # Criar uma função para representar a variável de estado
    u = fire.Function(V, name='u_VTI')

    # Derivada de du em relação a dx (deformações)
    epsilon = deformacoes(du)

    # Derivada de v em relação a dx
    epsilon_v = deformacoes(v)

    # Tensor elástico
    C_vti = c_vti_tensor(vP, vS, rho, eps1,
                         gamma, delta, 'weak')

    # Tensor de tensões de Cauchy
    sigma = C_vti * epsilon

    # # Problema bilinear (Princípio dos trabalhos virtuais)
    a = fire.inner(sigma, epsilon_v) * dx
    L = fire.inner(F1, v) * fire.ds(2)
    form = a - L
    # fire.solve(fire.lhs(form) == fire.rhs(form), u, bcs=bc)
    fire.solve(fire.lhs(form) == fire.rhs(form), u)
    fire.VTKFile("disp_VTI.pvd").write(u)


def static_TTI(V):
    '''
    static TTI problem
    '''

    # Criar um trial function (u = N du)
    du = fire.TrialFunction(V)

    # Criar uma função de teste
    v = fire.TestFunction(V)

    # Criar uma função para representar a variável de estado
    u = fire.Function(V, name='u_TTI')

    # Derivada de du em relação a dx (deformações)
    epsilon = deformacoes(du)

    # Derivada de v em relação a dx
    epsilon_v = deformacoes(v)

    # Tensor elástico
    C_vti = c_vti_tensor(vP, vS, rho, eps1,
                         gamma, delta, 'weak')

    C_tti = c_tti_tensor(C_vti, theta, phi)

    # Tensor de tensões de Cauchy
    sigma = C_tti * epsilon

    # # Problema bilinear (Princípio dos trabalhos virtuais)
    a = fire.inner(sigma, epsilon_v) * dx
    L = fire.inner(F1, v) * fire.ds(2)
    form = a - L
    # fire.solve(fire.lhs(form) == fire.rhs(form), u, bcs=bc)
    fire.solve(fire.lhs(form) == fire.rhs(form), u)
    fire.VTKFile("disp_TTI.pvd").write(u)


def nrbc_tensor(rho, Gamma, bnd_nod, G, V):
    ''''
    Constructs the NRBC tensor for the elastic ABC.

    Parameters
    ----------
    vP: `float`
        P-wave velocity [km/s]
    vS: `float`
        S-wave velocity [km/s]
    rho: `float`
        Density [kg/m³]
    Gamma: `ufl.tensors.ListTensor`
        Crhistoffel tensor
    bnd_nod : `array`
        Boundary nodes indices.
    G : `firedrake.TensorFunctionSpace`
        Function space for the crhistoffel tensor
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
    Constructs the crhistoffel tensor for a material with VTI anisotropy.

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
    propag_vector: `list` or `np.array`
        Propagation vector

    Returns
    -------
    Gamma_vti: `ufl.tensors.ListTensor`
        Crhistoffel tensor
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
    Constructs the crhistoffel tensor for a material with TTI anisotropy.
    Carcione (2022). Wave Fields in Real Media: Anisotropic elastic media

    Parameters
    ----------
    propag_vector: `list` or `np.array`
        Propagation vector

    Returns
    -------
    Gamma_tti: `ufl.tensors.ListTensor`
        Crhistoffel tensor
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


def apply_source(t, f0, v, F1, F_sou=1.):
    '''
    Ricker source for time t
    t: Current time
    f0: Source central frequency
    V: Test function space
    F1: Load function
    F_sou: Maximum source amplitude
    '''

    # Amplitude excitation
    r = (np.pi * f0 * (t - 1./f0))**2
    amp = (1. - 2. * r) * np.exp(-r) * F_sou

    # print('Amp: {:3.2e}'.format(amp))

    # Força de superfície
    # F1 = fire.Constant((0., amp, 0.))
    # L = fire.inner(F1, v) * fire.ds(2)

    F1.dat.data[source_vertex] = [0, amp, 0.]  # Point source
    L = fire.inner(F1, v) * dx  # Source term

    return L


def propagation_VTI(V, G):
    '''
    Algorithm to solve the forward problem
    '''
    # Final Time
    T = 2.  # s

    # Number of timesteps
    steps = 200

    # Timestep size
    dt = round(T / steps, 6)

    # Criar um trial function
    du = fire.TrialFunction(V)

    # Criar uma função de teste
    v = fire.TestFunction(V)

    # Criar uma função para representar a variável de estado
    u = fire.Function(V, name='u')
    u_ant1 = fire.Function(V, name='u_ant1')
    u_ant2 = fire.Function(V, name='u_ant2')

    # Load function
    F1 = fire.Function(V, name='F')

    # Derivada de du em relação a dx (deformações)
    epsilon = deformacoes(du)

    # Derivada de v em relação a dx
    epsilon_v = deformacoes(v)

    # Tensor elástico
    C_vti = c_vti_tensor(vP, vS, rho, eps1,
                         gamma, delta, 'weak')

    # Tensor de tensões de Cauchy
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


def propagation_TTI(V, G, H):
    '''
    Algorithm to solve the forward problem
    '''
    # Final Time
    T = 2.  # s

    # Number of timesteps
    steps = 200

    # Timestep size
    dt = round(T / steps, 6)

    # Criar um trial function
    du = fire.TrialFunction(V)

    # Criar uma função de teste
    v = fire.TestFunction(V)

    # Criar uma função para representar a variável de estado
    u = fire.Function(V, name='u')
    u_ant1 = fire.Function(V, name='u_ant1')
    u_ant2 = fire.Function(V, name='u_ant2')

    # Load function
    F1 = fire.Function(V, name='F')

    # Derivada de du em relação a dx (deformações)
    epsilon = deformacoes(du)

    # Derivada de v em relação a dx
    epsilon_v = deformacoes(v)

    # Tensor elástico
    C_vti = c_vti_tensor(vP, vS, rho, eps1,
                         gamma, delta, 'weak')

    C_tti = c_tti_tensor(C_vti, theta, phi)

    # Tensor de tensões de Cauchy
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
# propagation_VTI(V, G)
print("TTI Propagation Problem")
propagation_TTI(V, G, H)


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
