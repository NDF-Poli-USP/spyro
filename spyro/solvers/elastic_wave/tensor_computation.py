from firedrake import as_matrix
import firedrake as fire
import numpy as np

from ufl import conditional, lt
import ufl
from ...utils.typing import  WaveType

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

    rho = self.rho 
    vp  = self.c 
    vs  = self.c_s 
    
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

    t = theta * np.pi / 180.0
    ct = fire.cos(t)
    st = fire.sin(t)

    T = fire.as_tensor([
        [ct**2,      st**2,      2*ct*st],
        [st**2,      ct**2,     -2*ct*st],
        [-ct*st,     ct*st,      ct**2 - st**2]
    ])
    return T

def build_Gamma(self):

    dim = self.function_space.mesh().topological_dimension()

    C = self.Elastic_C

    if self.viscoelastic:
        if self.wave_type == WaveType.ISOTROPIC_ELASTIC:
            Gamma = Gamma_isotropic(self)

        elif self.wave_type == WaveType.ANISOTROPIC_VTI_ELASTIC:
            Gamma = Gamma_VTI(self, C, dim)

        elif self.wave_type == WaveType.ANISOTROPIC_TTI_ELASTIC:
            Gamma = Gamma_TTI(self, dim, C)
    else: 
        Gamma = np.zeros((6,6))
    return Gamma

def Gamma_isotropic(self):
    dim = self.function_space.mesh().topological_dimension()

    vp_sq = self.c**2
    vs_sq = self.c_s**2

    denom = vp_sq - 2.0 * vs_sq

    ratio = (vp_sq * self.Q_vp - 2 * vs_sq * self.Q_vs) / denom

    if dim == 2:
        Gamma = as_matrix([
        [self.Q_vp, ratio,     0],
        [ratio,     self.Q_vp, 0],
        [0,     0,     self.Q_vs]
    ])
    else:
        Gamma = as_matrix([
            [self.Q_vp, ratio,     ratio,       0,            0,            0],
            [ratio,     self.Q_vp, ratio,       0,            0,            0],
            [ratio,     ratio,     self.Q_vp,   0,            0,            0],
            [0,           0,           0,        self.Q_vs,  0,            0],
            [0,           0,           0,           0,            self.Q_vs,  0],
            [0,           0,           0,           0,            0,            self.Q_vs],
        ])
    
    return Gamma

def Gamma_VTI(self, C, dim):

    rho = self.rho
    vP = self.vP
    vS = self.vS

    epsilon = self.epsilon
    gamma = self.gamma
    delta = self.delta
    anisotropy = self.anisotropy
    
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

    Q33 = self.Q_vp
    Q11 = 2 * epsilon/(1 + 2 * epsilon) * self.Qepsilon_inv + Q33 
    Q44 = self.Q_vs
    Q66 = 2 * gamma/(1 + 2 * gamma) * self.Qgamma_inv + Q44 
            
    num1 = vP**2 * (1 + 2 * epsilon)
    num2 = 2 * vS**2 * (1 + 2 * gamma)
    denom = vP**2 * (1 + 2 * epsilon) - 2 * vS**2 * (1 + 2 * gamma)
    Q12 = (num1 * Q11 - num2 * Q66)/denom

    if anisotropy == 'weak':
        c1 = (delta * C33 + 0.5 * (C11 + 2 * C33 - 3 * C44)) / (2 * (C13 + C44))
        c2 = (- C11 - 3 * C33 + 4 * C44) / (4 * (C13 + C44)) - 1
        c3 = (C33 - C44) / (4 * (C13 + C44))
        c4 = C33 / (2 * (C13 + C44))
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

def Gamma_TTI(self, dim, C):

    C_vti_real = C_vti_tensor(dim)

    Q_vti = Gamma_VTI(self, C, dim)

    eps = 1e-12

    shape = C_vti_real.ufl_shape
    n = shape[0]  

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

    elif n == 3:  

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

    C_tti_real = C_tti_tensor(C_vti_real, dim)
    C_tti_imag = C_tti_tensor(C_imag, dim)

    eps = 1e-12
    Gamma = fire.as_tensor([
        [conditional(abs(C_tti_real[i, j]) > eps,
                     C_tti_imag[i, j] / C_tti_real[i, j],
                     0.0)
         for j in range(n)]
        for i in range(n)
    ])
    return Gamma
