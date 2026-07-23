from firedrake import (assemble, Constant, curl, DirichletBC, div, Function,
                       FunctionSpace, project, VectorFunctionSpace, TensorFunctionSpace, sym, grad,
                       as_matrix, as_vector)

from ufl import conditional, lt
import ufl

from .anisotropy import *

def C_computation(self):
    dim = self.function_space.mesh().topological_dimension()

    if self.wave_type == 'isotropic':
        C_elas = c_iso_tensor(self)

    elif self.wave_type == 'anisotropic_VTI':
        C_elas = c_vti_tensor(self.PropISO, self.PropVTI, dim)

    else:
        c_vti = c_vti_tensor(self.PropISO, self.PropVTI, dim)
        C_elas = c_tti_tensor(c_vti, self.PropTTI, dim)
    
    return C_elas

def c_iso_tensor(self):
    dim = self.function_space.mesh().topological_dimension()

    if dim == 2:
        c = as_matrix([
                [self.lmbda + 2*self.mu, self.lmbda,       0],
                [self.lmbda,       self.lmbda + 2*self.mu, 0],
                [0,           0,            self.mu]
                ])
    else:
        c = as_matrix([
            [self.lmbda + 2*self.mu, self.lmbda,       self.lmbda,       0,    0,    0],
            [self.lmbda,       self.lmbda + 2*self.mu, self.lmbda,       0,    0,    0],
            [self.lmbda,       self.lmbda,       self.lmbda + 2*self.mu, 0,    0,    0],
            [0,           0,           0,           self.mu,   0,    0],
            [0,           0,           0,           0,    self.mu,   0],
            [0,           0,           0,           0,    0,    self.mu]
            ])
    return c

def c_vti_tensor(PropISO, PropVTI, dim):
    """Constructs the elastic tensor for a material with VTI anisotropy.

    TODO References: Thomsen (1986). Geophysics 51, 10, 1954-1966

    Parameters
        ----------
    PropISO: `object`
        An instance of the isotropic properties class. Attributes:
        - vP: `Firedrake.Function`
            P-wave velocity [m/s]
        - vS: `Firedrake.Function`
            S-wave velocity [m/s]
        - rho: `Firedrake.Function`
            Density [kg/m³]
    PropVTI: `object`
        An instance of the VTI anisotropy properties class. Attributes:
        - epsilon: `Firedrake.Function`
            Thomsen parameter epsilon
        - gamma: `Firedrake.Function`
            Thomsen parameter gamma
        - delta: `Firedrake.Function`
            Thomsen parameter delta
        - anisotropy: `str`
            Type of anisotropy: 'weak' or 'exact'

    Returns
    -------
    C_vti: `ufl.tensors.ListTensor`
        Elastic tensor
    """

    print('VTI')

    # Assigning isotropic properties
    rho = PropISO.rho
    vP = PropISO.vP
    vS = PropISO.vS

    # Assigning anisotropic properties
    epsilon = PropVTI.epsilon
    gamma = PropVTI.gamma
    delta = PropVTI.delta
    anisotropy = PropVTI.anisotropy

    # Computing the elastic tensor components
    C33 = rho * vP ** 2
    C11 = C33 * (1. + 2. * epsilon)
    C44 = rho * vS ** 2
    C66 = C44 * (1. + 2. * gamma)
    C12 = C11 - 2 * C66

    # C13 is calculated based on the type of anisotropy
    dC = C33 - C44
    if anisotropy == 'weak':
        C13 = (delta * C33**2 + 0.5 * dC * (C11 + C33 - 2 * C44))**0.5
    elif anisotropy == 'exact':
        C13 = (dC * (C33 * (1. + 2 * delta) - C44))**0.5
    C13 -= C44

    if dim == 2:
        C_vti = fire.as_tensor(((C11, C13, 0),
                                (C13, C33, 0),
                                (0, 0, C44)))

    else:
        # Assembling the elastic tensor
        C_vti = fire.as_tensor(((C11, C12, C13, 0, 0, 0),
                                (C12, C11, C13, 0, 0, 0),
                                (C13, C13, C33, 0, 0, 0),
                                (0, 0, 0, C44, 0, 0),
                                (0, 0, 0, 0, C44, 0),
                                (0, 0, 0, 0, 0, C66)))

    return C_vti

def c_tti_tensor(C_vti, PropTTI, dim):
    """Constructs the elastic tensor for a material with TTI anisotropy.

    TODO References: Yang et al (2020). Survey in Geophysics 41, 805-833

    Parameters
    ----------
    C_vti: `ufl.tensors.ListTensor`
        Elastic tensor for VTI anisotropy
    PropTTI: `object`
        An instance of the TTI anisotropy properties class. Attributes:
        - theta: `Firedrake.Function`
            Tilt angle in degrees
        - phi: `Firedrake.Function`
            Azimuth angle in degrees (default is 0: 2D case)

    Returns
    -------
    C_tti: `ufl.tensors.ListTensor`
        Elastic tensor for TTI anisotropy
    """

    # Assigning anisotropic properties
    theta = PropTTI.theta
    phi = PropTTI.phi
    
    if dim == 2:
        T = bond_rotation_2d_elastic(theta)
        C_tti = fire.dot(fire.dot(T, C_vti), fire.transpose(T))

    elif dim == 3:
        # Tilt angle
        t = theta * np.pi / 180.
        ct = fire.cos(t)
        st = fire.sin(t)

        # Azimuth angle
        p = phi * np.pi / 180.
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

        # Transformation matrix for Voigt notation using UFL
        T = fire.as_tensor([
                [R11 ** 2, R12 ** 2, R13 ** 2, 2 * R12 * R13, 2 * R13 * R11, 2 * R11 * R12],
                [R21 ** 2, R22 ** 2, R23 ** 2, 2 * R22 * R23, 2 * R23 * R21, 2 * R21 * R22],
                [R31 ** 2, R32 ** 2, R33 ** 2, 2 * R32 * R33, 2 * R33 * R31, 2 * R31 * R32],
                [R21 * R31, R22 * R32, R23 * R33, R22 * R33 + R23 * R32,
                R21 * R33 + R23 * R31, R22 * R31 + R21*R32],
                [R31 * R11, R32 * R12, R33 * R13, R12 * R33 + R13 * R32,
                R13 * R31 + R11 * R33, R11 * R32 + R12*R31],
                [R11 * R21, R12 * R22, R13 * R23, R12 * R23 + R13 * R22,
                R13 * R21 + R11 * R23, R11 * R22 + R12*R21]
            ])

        # Apply transformation: C_tti = T * C_vti * T^T
        C_tti = fire.dot(fire.dot(T, C_vti), fire.transpose(T))

    return C_tti

def bond_rotation_2d_elastic(theta):
    """
    Constrói a matriz de transformação de Bond para o caso 2D elástico (P-SV).
    A rotação é aplicada no plano x-z (apenas ângulo de inclinação theta).
    Retorna uma matriz 3x3 que atua no vetor de Voigt [xx, zz, xz]^T.
    """
    t = theta * np.pi / 180.0
    ct = fire.cos(t)
    st = fire.sin(t)

    # Matriz de rotação 3x3 de Bond para 2D (P-SV)
    T = fire.as_tensor([
        [ct**2,      st**2,      2*ct*st],
        [st**2,      ct**2,     -2*ct*st],
        [-ct*st,     ct*st,      ct**2 - st**2]
    ])
    return T

def build_Gamma(self):

    dim = self.function_space.mesh().topological_dimension()

    C = self.C_elas

    PropISO = self.PropISO
    PropVTI = self.PropVTI
    PropTTI = self.PropTTI

    if self.viscoelastic == True:
        if self.wave_type == 'isotropic':
            Gamma = Gamma_iso(self)

        elif self.wave_type == 'anisotropic_VTI':
            Gamma = Gamma_VTI(self, PropISO, PropVTI, C, dim)
        
        elif self.wave_type == 'anisotropic_TTI':
            Gamma = Gamma_TTI(self, PropISO, PropVTI, PropTTI, dim, C)
    else: 
        Gamma = np.zeros((6,6))
    return Gamma

def Gamma_iso(self):
    kappa = self.lmbda + (2/3) * self.mu
    alpha_sq = (self.lmbda + 2*self.mu) / self.rho
    beta_sq = self.mu / self.rho
    denom = alpha_sq - (4/3) * beta_sq

    Qkappa_inv = conditional(
        lt(abs(denom), 1e-12), self.Qp_inv, (alpha_sq * self.Qp_inv - (4/3) * beta_sq * self.Qs_inv) / denom)

    lmbda_Q = kappa * Qkappa_inv - (2/3) * self.mu * self.Qs_inv

    ratio = conditional(lt(abs(self.lmbda), 1e-12), 0.0, lmbda_Q / self.lmbda)

    Gamma = as_matrix([
                    [self.Qp_inv,   ratio,    ratio,    0, 0, 0],
                    [ratio,    self.Qp_inv,   ratio,    0, 0, 0],
                    [ratio,    ratio,    self.Qp_inv,   0, 0, 0],
                    [0,        0,        0,        self.Qs_inv, 0, 0],
                    [0,        0,        0,        0, self.Qs_inv, 0],
                    [0,        0,        0,        0, 0, self.Qs_inv]
                    ])
    
    return Gamma

def Gamma_VTI(self, PropISO, PropVTI, C, dim):

    rho = PropISO.rho
    vP = PropISO.vP
    vS = PropISO.vS

    epsilon = PropVTI.epsilon
    gamma = PropVTI.gamma
    delta = PropVTI.delta
    anisotropy = PropVTI.anisotropy
    
    if dim == 2:
        C11 = C[0, 0]
        C13 = C[0, 1]
        C33 = C[1, 1]
        C44 = C[2, 2]

    elif dim == 3:
        C11 = C[0, 0]
        C13 = C[0, 2]
        C33 = C[2, 2]
        C44 = C[3, 3]

    Q33 = self.Qp_inv
    Q11 = 2 * epsilon/(1 + 2 * epsilon) * self.Qepsilon_inv + Q33 
    Q44 = self.Qs_inv
    Q66 = 2 * gamma/(1 + 2 * gamma) * self.Qgamma_inv + Q44 
            
    num1 = vP**2 * (1 + 2 * epsilon)
    num2 = 2 * vS**2 * (1 + 2 * gamma)
    denom = vP**2 * (1 + 2 * epsilon) - 2 * vS**2 * (1 + 2 * gamma)
    Q12 = (num1 * Q11 - num2 * Q66)/denom

    if anisotropy == 'weak':
        c1 = (2 * delta * C33 + 0.5 * (C11 + 2 * C33 - 3 * C44)) / (2 * (C13 + C44))
        c2 = (- C11 - 3 * C33 + 4 * C44) / (4 * (C13 + C44)) - 1
        c3 = (C33 - C44) / (4 * (C13 + C44))
        c4 = C33**2 / (2 * (C13 + C44))
        Q13 = (c1 * C33 * Q33 + c2 * C44 * Q44 + c3 * C11 * Q11 + c4 * delta * self.Qdelta_inv) / C13
       
    elif anisotropy == 'exact':
        c1 = (C33 * (1 + 2 * gamma) - C44 + (C33 - C44) * (1 + 2 * delta)) / (2 * (C13 + C44))
        c2 = (- C33 * (1 + 2 * delta) + 2 * C44 - C33) / (2 * (C13 + C44)) - 1
        c3 = C33 * (C33 - C44)/(C13 + C44)
        Q13 = (c1 * C33 * Q33 + c2 * C44 * Q44 + c3 * delta * self.Qdelta_inv)/C13

    if dim == 2:
        Gamma = fire.as_matrix([[Q11, Q13, 0],
                                [Q13, Q33, 0],
                                [0, 0, Q44]])
    else:
        Gamma = fire.as_matrix([[Q11, Q12, Q13, 0, 0, 0],
                                [Q12, Q11, Q13, 0, 0, 0],
                                [Q13, Q13, Q33, 0, 0, 0],
                                [0, 0, 0, Q44, 0, 0],
                                [0, 0, 0, 0, Q44, 0],
                                [0, 0, 0, 0, 0, Q66]])
    return Gamma

def Gamma_TTI(self, PropISO, PropVTI, PropTTI, dim, C):
    """
    Build Gamma (Q^{-1}) on TTI system.
    """
    # 1. Obter a parte real da matriz VTI
    C_vti_real = c_vti_tensor(PropISO, PropVTI, dim)

    Q_vti = Gamma_VTI(self, PropISO, PropVTI, C, dim)

    eps = 1e-12

    shape = C_vti_real.ufl_shape
    n = shape[0]  # 2, 3 ou 6

    if n == 6:  # 3D

        Q11 = Q_vti[0,0]
        Q33 = Q_vti[2,2]
        Q44 = Q_vti[3,3]
        Q66 = Q_vti[5,5]
        Q13 = Q_vti[0,2]
        Q12 = Q_vti[0,1]

        C11 = C_vti_real[0,0]
        C12 = C_vti_real[0,1]
        C13 = C_vti_real[0,2]
        C33 = C_vti_real[2,2]
        C44 = C_vti_real[3,3]
        C66 = C_vti_real[5,5]

        # Q12 derivado
        denom = C11 - 2*C66
        Q12 = conditional(abs(denom) > 1e-12,
                              (C11/Q11 - 2*C66/Q66) / denom,
                              0.0)

        C_imag = fire.as_tensor([
            [C11*Q11, C12*Q12, C13*Q13, 0.0, 0.0, 0.0],
            [C12*Q12, C11/Q11, C13*Q13, 0.0, 0.0, 0.0],
            [C13*Q13, C13*Q13, C33*Q33, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, C44*Q44, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, C44*Q44, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, C66*Q66]
        ])

    elif n == 3:  # 2D elástico P-SV

        Q11 = Q_vti[0,0]
        Q33 = Q_vti[1,1]
        Q13 = Q_vti[0,1]
        Q44 = Q_vti[2,2]

        C11 = C_vti_real[0,0]
        C13 = C_vti_real[0,1]
        C33 = C_vti_real[1,1]
        C44 = C_vti_real[2,2]

        C_imag = fire.as_tensor([
            [C11*Q11, C13*Q13, 0.0],
            [C13*Q13, C33*Q33, 0.0],
            [0.0, 0.0, C44*Q44]
        ])

    elif n == 2:  # 2D acústico
        C11 = C_vti_real[0,0]
        C13 = C_vti_real[0,1]
        C33 = C_vti_real[1,1]

        C_imag = fire.as_tensor([
            [C11*Q11, C13*Q13],
            [C13*Q13, C33*Q33]
        ])

    else:
        raise ValueError(f"Unexpected matrix size: {n}")

    # 3. Rotacionar separadamente as partes real e imaginária
    #    (a função c_tti_tensor aplica a rotação de Bond)
    C_tti_real = c_tti_tensor(C_vti_real, PropTTI, dim)
    C_tti_imag = c_tti_tensor(C_imag, PropTTI, dim)

    # 4. Calcular Gamma = imag / real (com proteção)
    eps = 1e-12
    Gamma = fire.as_tensor([
        [conditional(abs(C_tti_real[i, j]) > eps,
                     C_tti_imag[i, j] / C_tti_real[i, j],
                     0.0)
         for j in range(n)]
        for i in range(n)
    ])
    return Gamma