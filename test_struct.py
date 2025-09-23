import firedrake as fire
import numpy as np


# -*- coding: utf-8 -*-

# # Importar fenics e dolfin adjoint
# from fenics import*
# from fenics_adjoint import*

# # Importar ipdb para debugar
import ipdb

# Altura do domínio
h = 100

# Largura do domínio
b = 300

# Espessura do domínio
t = 10

# Módulo de Young
E = 10

# Coefieciente de Poisson
nu = .3

# # Volume do domínio
# delta = b / h

# # Volume máximo (fração volumétrica para restrição)
# v_frac = Constant(0.6) * delta

# Força de superfície
F1 = fire.Constant((0., -0.01, 0.))

# Matriz constitutiva (matriz de elasticidade) isotrópica
# C_iso = E / (1 * nu**2) * fire.as_tensor(((1, nu, 0),
#                                           (nu, 1, 0),
#                                           (0, 0, (1 - nu) / 2)))


# # Penalizador do SIMP
# p = 3

# # Raio do filtro
# r = 6

# Tamanho do elemento
elsize = 2

# Número de elementos em x
nelx = int(b / elsize)

# Número de elementos em y
nely = int(h / elsize)

# Número de elementos em y
nelz = int(t / elsize)

# Definir malha do MEF
# mesh = fire.RectangleMesh(nelx, nely, b, h)
mesh = fire.BoxMesh(nelx, nely, nelz, b, h, t)

# Espaço para a variável de estado (deslocamento)
V = fire.VectorFunctionSpace(mesh, "CG", 1)


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


def c_tti_tensor(C_vti, theta, phi=0.):
    '''
    Constructs the elastic tensor for a material with TTI anisotropy.
    References: Yang et al (2020). Survey in Geophysics 41, 805-833

    Parameters
    ----------
    C_vti: `ufl.tensors.ListTensor`
        Elastic tensor for VTI anisotropy
    theta: `float`
        Tilt angle in degrees
    phi: `float`, optional
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
    ct = 0. if t in c_zero else np.cos(theta)
    st = 0. if t in s_zero else np.sin(theta)

    # Azimuth angle
    p = phi * np.pi / 180.
    cp = 0. if p in c_zero else np.cos(phi)
    sp = 0. if p in s_zero else np.sin(phi)

    # Rotation matrix components
    R11 = ct * cp
    R22 = cp
    R33 = ct
    R12 = -sp
    R13 = st * cp
    R21 = ct * sp
    R23 = st * sp
    R31 = -st
    R32 = 0

    # Convert to array
    C_vti_arr = np.zeros(C_vti.ufl_shape)
    for i in range(C_vti.ufl_shape[0]):
        for j in range(C_vti.ufl_shape[1]):
            C_vti_arr[i, j] = C_vti[i, j]

    # Transformation matrix
    T_arr = np.array((
        (R11**2, R12**2, R13**2, 2 * R12 * R13, 2 * R13 * R11, 2 * R11 * R12),
        (R21**2, R22**2, R23**2, 2 * R22 * R23, 2 * R23 * R21, 2 * R21 * R22),
        (R31**2, R32**2, R33**2, 2 * R32 * R33, 2 * R33 * R31, 2 * R31 * R32),
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

# # Espaço para a variável de projeto (pseudo-densidade)
# P = fire.FunctionSpace(mesh, "CG", 1)


# class Clamped(SubDomain):
#     '''
#     Definir parte fixa do contorno (posição da condição de contorno
#     de Dirichlet)
#     '''

#     def inside(self, x, on_boundary):
#         '''
#         uᵢ = (0,0) ∀ x = 0
#         '''
#         return near(x[0], 0) and on_boundary


# class Load(SubDomain):
#     '''
#     Definir onde a força de superfície é aplicada (posição da
#     condição de contorno de Neumann)
#     '''

#     def inside(self, x, on_boundary):
#         return near(x[0], l) and x[1] >= 45. and x[1] <= 55. and on_boundary


# # Criando uma MeshFunction
# boundaries = MeshFunction("size_t", mesh, 1)

# # "Pintando" toda a mesh function com a cor "0"
# boundaries.set_all(0)

# # Pintando a parte fixa com a cor "1"
# Clamped().mark(boundaries, 1)

# # Pintando a posição da aplicação da força com a cor "2"
# Load().mark(boundaries, 2)

# Criando a condição de contorno de Dirichlet
bc = fire.DirichletBC(V, ((0., 0., 0.)), 1)

# # Criando ds (onde a força de superfície será integrada porteriormente)
# ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Criando dx
q_degree = 3
dx = fire.dx(metadata={'quadrature_degree': q_degree})


def SIMP(rho, p, rho_min=1e-3):
    '''
    Função de interpolação do modelo de material SIMP
    '''
    return rho_min + (1 - rho_min) * rho**p


def helmholtz_filter(a):
    '''
    Função para aplicar o filtro de Helmholtz
    '''
    a_f = TrialFunction(P)
    vH = TestFunction(P)
    F = (
        inner(r * r * grad(a_f), grad(vH)) * dx
        + inner(a_f, vH) * dx
        - inner(a, vH) * dx
    )
    aH = lhs(F)
    LH = rhs(F)
    a_f = Function(P, name="Filtered Control")
    solve(aH == LH, a_f)
    return a_f


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

    # Tensor de deformações em notação de Voigth
    # epsilon = fire.as_vector((eps_x, eps_y, gamma_xy))

    return epsilon


def problema_direto(V):
    '''
    Resolve o problema direto e retorna u
    '''

    # # Filtrar ρ
    # rho_f = helmholtz_filter(rho)

    # Criar um trial function (u = N du)
    du = fire.TrialFunction(V)

    # Criar uma função de teste
    v = fire.TestFunction(V)

    # Criar uma função para representar a variável de estado
    u = fire.Function(V)

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

    # # Problema bilinear (Princípio dos trabalhos virtuais)
    # # a = t * inner(SIMP(rho_f, p) * sigma, epsilon_v) * dx
    # # L = inner(F1, v) * ds(2)
    a = fire.inner(sigma, epsilon_v) * dx
    L = fire.inner(F1, v) * fire.ds(2)

    fire.solve(a == L, u, bcs=bc)
    # fire.solve(a == L, u)

    fire.VTKFile("displacement.pvd").write(u)
    # ipdb.set_trace()

    # # Fazer o "assemble" do sistema linear
    # A, b = assemble_system(a, L, bc)


problema_direto(V)
