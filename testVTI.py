import firedrake as fire
import numpy as np
import ipdb


def c_vti_tensor(vP, vS, rho, epsilon, gamma, delta, anysotropy):
    ''''
    Constructs the elastic tensor for a material with VTI anisotropy

    Parameters
    ----------
    vP: float
        P-wave velocity [km/s]
    vS: float
        S-wave velocity [km/s]
    rho: float
        Density [kg/m³]
    epsilon: float
        Thomsen parameter epsilon
    gamma: float
        Thomsen parameter gamma
    delta: float
        Thomsen parameter delta

    Returns
    -------
    C: ufl.tensors.ListTensor
        Elastic tensor
    '''

    C33 = rho * vP**2
    C11 = C33 * (1. + 2. * epsilon)
    C44 = rho * vS**2
    C66 = C44 * (1. + 2. * gamma)
    C12 = C11 - 2 * C66

    dC = C33 - C44
    if anysotropy == 'weak':
        C13 = (delta * C33**2 + 0.5 * dC * (C11 + C33 - 2 * C44))**0.5
    elif anysotropy == 'exact':
        C13 = (dC * (C33 * (1. + 2 * delta) - C44))**0.5
    C13 -= C44

    c_vti = fire.as_tensor(((C11, C12, C13, 0, 0, 0),
                            (C12, C11, C13, 0, 0, 0),
                            (C13, C13, C33, 0, 0, 0),
                            (0, 0, 0, C44, 0, 0),
                            (0, 0, 0, 0, C44, 0),
                            (0, 0, 0, 0, 0, C66)))

    comp_c_vti = np.array([[C11, C12, C13],
                           [C33, C44, C66]])

    return c_vti, comp_c_vti


def gen_tensor_pvd(iso_par, thomsen_par, anysotropy, name):
    ''''
    Generate a pvd file for the elastic tensor with VTI anisotropy

    Parameters
    ----------
    iso_par: list
        List of isotropic parameters [vP, vS, rho]
    thomsen_par: list
        List of Thomsen parameters [epsilon, gamma, delta]
    anysotropy: str
        Type of anisotropy ('weak' or 'exact')
    name: str
        Name of the file

    Returns
    -------
    None
    '''

    vP, vS, rho = iso_par
    epsilon, gamma, delta = thomsen_par

    # Space function
    W = fire.TensorFunctionSpace(mesh, 'CG', 1, shape=(2, 3))

    # Constitutive tensor for VTI
    c_ani = fire.Function(W, name='c_vti')

    # c_vti, comp_c_vti = c_vti_tensor(vP, vS, rho, epsilon, gamma, delta)
    # c_ani.dat.data[:] = comp_c_vti

    c1_vti, comp1_c_vti = c_vti_tensor(vP, vS, rho, epsilon,
                                       gamma, delta, anysotropy)
    c2_vti, comp2_c_vti = c_vti_tensor(1.5 * vP, vP / 4., 2.5 * rho,
                                       0.3, 0.4, 0.2, anysotropy)

    print(comp1_c_vti)
    print(comp2_c_vti)

    # More elegant Firedrake way
    x = fire.SpatialCoordinate(mesh)
    left_region = fire.conditional(x[0] <= 0.5, 1, 0)

    # Create full tensor expressions
    C1 = fire.as_tensor(comp1_c_vti)
    C2 = fire.as_tensor(comp2_c_vti)

    # Combined expression
    C_combined = C1 * left_region + C2 * (1 - left_region)

    # Inteporlate to function space
    c_ani.interpolate(C_combined)

    # Save file
    outfile = fire.VTKFile("output/" + name + ".pvd")
    outfile.write(c_ani)


Lx = 1.           # Length of the rectangle
Ly = 1.           # Width of the rectangle
lmax = 0.05       # Maximum edge length
lmin = lmax       # Minimum edge length

# Create rectangular mesh
nx, ny = int(Lx / lmax), int(Ly / lmax)
mesh = fire.RectangleMesh(nx, ny, Lx, Ly)

# P = fire.FunctionSpace(mesh, 'CG', 1)
# V = fire.VectorFunctionSpace(mesh, 'CG', 1)

# Data
vP = 1.5  # P-wave velocity [km/s]
vS = 0.75  # S-wave velocity [km/s]
rho = 1e3  # Density [kg/m³]
epsilon = 0.2  # Thomsen parameter epsilon
gamma = 0.3  # Thomsen parameter gamma
delta = 0.1  # Thomsen parameter delta

gen_tensor_pvd([vP, vS, rho], [epsilon, gamma, delta], 'exact', 'c_vti_exact')
gen_tensor_pvd([vP, vS, rho], [epsilon, gamma, delta], 'weak', 'c_vti_weak')


# class FunTrig:
#     """docstring for ClassName"""

#     def __init__(self, labFib):
#         self.labFib = labFib

#     def cosf(self, theta, pf=20):
#         '''
#         Cosine function depending on material model
#         '''
#         if self.labFib == 'spimfo':
#             # Cosine function represented by a Taylor series
#             if abs(pi * theta) == pi/2:
#                 c = 0
#             else:
#                 i = 0
#                 c = 0
#                 while i <= int(float(pf)):
#                     c += (-1.)**i / factorial(2 * i) * \
#                         (pi * theta)**(2 * i)
#                     i += 1

#         elif self.labFib == 'ndfo-m':
#             # System function
#             if abs(theta) == 90.:
#                 c = 0
#             else:
#                 c = cos(theta*pi/180)

#         return c

#     def sinf(self, theta, pf=20):
#         '''
#         Sine function depending on material model
#         '''
#         if self.labFib == 'spimfo':
#             # Sine function represented by a Taylor series
#             if abs(pi * theta) == pi:
#                 s = 0
#             else:
#                 i = 0
#                 s = 0
#                 while i <= int(float(pf)):
#                     s += (-1.)**i / factorial(2 * i + 1) * \
#                         (pi * theta)**(2 * i + 1)
#                     i += 1

#         elif self.labFib == 'ndfo-m':
#             # System function
#             if abs(theta) == 180.:
#                 s = 0
#             else:
#                 s = sin(theta*pi/180)
#         return s


# C = self.transformation(funTrig, C, theta, pf)
# def transformation(self, funTrig, C, theta, pf=20):
#     c = funTrig.cosf(theta, pf)
#     s = funTrig.sinf(theta, pf)
#     T = as_tensor(((c * c, s * s, 0, 0, 0, 2 * s * c),  # Transformation matrix
#                    (s * s, c * c, 0, 0, 0, -2 * s * c),
#                    (0, 0, 1, 0, 0, 0),
#                    (0, 0, 0, c, -s, 0),
#                    (0, 0, 0, s, c, 0),
#                    (-s * c, s * c, 0, 0, 0, c * c - s * s)))  # (2.96)
#     DT_1 = c * c + s * s
#     DT_3 = c**4 + 2 * c * c * s * s + s**4
#     DT_2 = (2 * c * s) / DT_3
#     T_inv = as_tensor(((c * c / DT_3, s * s / DT_3, 0, 0, 0, -DT_2),
#                        (s * s / DT_3, c * c / DT_3, 0, 0, 0, DT_2),
#                        (0, 0, 1, 0, 0, 0),
#                        (0, 0, 0, c / DT_1, s / DT_1, 0),
#                        (0, 0, 0, -s / DT_1, c / DT_1, 0),
#                        (c * s / DT_3, -c * s / DT_3, 0, 0, 0, (c * c - s * s) / DT_3)))
#     R = as_tensor(((1, 0, 0, 0, 0, 0),
#                    (0, 1, 0, 0, 0, 0),
#                    (0, 0, 1, 0, 0, 0),
#                    (0, 0, 0, 2, 0, 0),
#                    (0, 0, 0, 0, 2, 0),
#                    (0, 0, 0, 0, 0, 2)))  # (2.101) Reuter matrix

#     RInv = as_tensor(((1, 0, 0, 0, 0, 0),
#                       (0, 1, 0, 0, 0, 0),
#                       (0, 0, 1, 0, 0, 0),
#                       (0, 0, 0, 0.5, 0, 0),
#                       (0, 0, 0, 0, 0.5, 0),
#                       (0, 0, 0, 0, 0, 0.5)))    # (2.101) Inv. Reuter matrix
#     return T_inv * C * R * T * RInv
