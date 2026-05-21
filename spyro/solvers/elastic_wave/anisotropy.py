import firedrake as fire
import numpy as np


class AnisotropyTensor():
    """Class for the Elastic tensor VTI and TTI anisotropy materials."""

    def c_vti_tensor(PropISO, PropVTI):
        """Constructs the elastic tensor for a material with VTI anisotropy.

        Reference: Thomsen (1986). Geophysics 51, 10, 1954-1966

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
