from firedrake import (assemble, Constant, curl, DirichletBC, div, Function,
                       FunctionSpace, project, VectorFunctionSpace, TensorFunctionSpace, sym, grad,
                       as_matrix, as_vector)

from ufl import conditional, lt
import ufl
from ...utils.typing import  WaveType

from .anisotropy import *

def C_computation(self):
    dim = self.function_space.mesh().topological_dimension()

    if self.wave_type == WaveType.ISOTROPIC_ELASTIC:
        Elastic_C = C_isotropic_tensor(self)
    elif self.wave_type == WaveType.ANISOTROPIC_VTI_ELASTIC:
        Elastic_C = C_vti_tensor(self, dim)
    elif self.wave_type == WaveType.ANISOTROPIC_TTI_ELASTIC:
        c_vti = C_vti_tensor(self, dim)
        Elastic_C = C_tti_tensor(self, c_vti, dim)
    
    return Elastic_C

def C_isotropic_tensor(self):
    """
    Elastic tensor in terms of vp, vs and rho.
    """

    dim = self.function_space.mesh().topological_dimension()

    rho = self.rho # self.IsotropicProperties.rho
    vp  = self.c # self.IsotropicProperties.vP
    vs  = self.c_s # self.IsotropicProperties.vS

    C11 = rho * vp**2
    C44 = rho * vs**2
    C12 = C11 - 2*C44

    if dim == 2:
        C_isotropic = fire.as_tensor((
            (C11, C12, 0),
            (C12, C11, 0),
            (0,   0,   C44)
        ))
    else:
        C_isotropic = fire.as_tensor((
            (C11, C12, C12, 0,   0,   0),
            (C12, C11, C12, 0,   0,   0),
            (C12, C12, C11, 0,   0,   0),
            (0,   0,   0,   C44, 0,   0),
            (0,   0,   0,   0,   C44, 0),
            (0,   0,   0,   0,   0,   C44)
        ))

    return C_isotropic

def C_vti_tensor(self, dim):
    """Constructs the elastic tensor for a material with VTI anisotropy.

    TODO References: Thomsen (1986). Geophysics 51, 10, 1954-1966

    Parameters
        ----------
    IsotropicProperties: `object`
        An instance of the isotropic properties class. Attributes:
        - vP: `Firedrake.Function`
            P-wave velocity [m/s]
        - vS: `Firedrake.Function`
            S-wave velocity [m/s]
        - rho: `Firedrake.Function`
            Density [kg/m³]
    AnisotropicPropertiesVTI: `object`
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

    # Assigning isotropic properties
    rho = self.rho
    vP = self.c
    vS = self.c_s

    # Assigning anisotropic properties
    epsilon = self.epsilon
    gamma = self.gamma
    delta = self.delta
    anisotropy = self.anisotropy_type

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

def C_tti_tensor(self, C_vti, dim):
    """Constructs the elastic tensor for a material with TTI anisotropy.

    TODO References: Yang et al (2020). Survey in Geophysics 41, 805-833

    Parameters
    ----------
    C_vti: `ufl.tensors.ListTensor`
        Elastic tensor for VTI anisotropy
    AnisotropicPropertiesTTI: `object`
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
    theta = self.theta
    phi = self.phi
    
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