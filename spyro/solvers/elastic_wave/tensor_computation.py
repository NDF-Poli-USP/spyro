from firedrake import (assemble, Constant, curl, DirichletBC, div, Function,
                       FunctionSpace, project, VectorFunctionSpace, TensorFunctionSpace, sym, grad,
                       as_matrix, as_vector)

from ufl import conditional, lt

from .anisotropy import *

def C_computation(self):
    if self.wave_type == 'isotropic':
        dim = self.function_space.mesh().topological_dimension()
        print('dim',dim)

        if dim == 2:
            C_elas = as_matrix([
                    [self.lmbda + 2*self.mu, self.lmbda,       0],
                    [self.lmbda,       self.lmbda + 2*self.mu, 0],
                    [0,           0,            self.mu]
                ])
        else:
            C_elas = as_matrix([
                [self.lmbda + 2*self.mu, self.lmbda,       self.lmbda,       0,    0,    0],
                [self.lmbda,       self.lmbda + 2*self.mu, self.lmbda,       0,    0,    0],
                [self.lmbda,       self.lmbda,       self.lmbda + 2*self.mu, 0,    0,    0],
                [0,           0,           0,           self.mu,   0,    0],
                [0,           0,           0,           0,    self.mu,   0],
                [0,           0,           0,           0,    0,    self.mu]
                ])
    elif self.wave_type == 'anisotropic_VTI':
        C_elas = c_vti_tensor(self.PropISO, self.PropVTI, self.function_space.mesh().topological_dimension())

    else:
        C_elas = c_tti_tensor(self.PropISO, self.PropVTI, self.function_space.mesh().topological_dimension())
    return C_elas


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
        print(C_vti)

    else:
        # Assembling the elastic tensor
        C_vti = fire.as_tensor(((C11, C12, C13, 0, 0, 0),
                                (C12, C11, C13, 0, 0, 0),
                                (C13, C13, C33, 0, 0, 0),
                                (0, 0, 0, C44, 0, 0),
                                (0, 0, 0, 0, C44, 0),
                                (0, 0, 0, 0, 0, C66)))

    return C_vti

def c_tti_tensor(C_vti, PropTTI):
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

def build_Gamma(self):
    if self.wave_type == 'isotropic':
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
    else: 
        Gamma = np.zeros((6,6))
    return Gamma