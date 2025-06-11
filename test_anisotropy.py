import firedrake as fire
import numpy as np
import ipdb


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
    comp_C_vti: `dict`
        Non-zero components of the elastic tensor.
        Keys: [C11, C12, C13, C33, C44, C66]
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

    # Non-zero components
    comp_C_vti = {}
    comp_C_vti["C11"] = C11
    comp_C_vti["C12"] = C12
    comp_C_vti["C13"] = C13
    comp_C_vti["C33"] = C33
    comp_C_vti["C44"] = C44
    comp_C_vti["C66"] = C66

    return C_vti, comp_C_vti


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

    # Non-zero components
    comp_C_tti = {}
    for i in range(C_vti.ufl_shape[0]):
        for j in range(i, C_vti.ufl_shape[1]):
            if C_tti[i, j] != 0.:
                comp_C_tti["C" + str(i+1) + str(j+1)] = C_tti[i, j]

    return C_tti, comp_C_tti


def gen_tensor_pvd(mesh, iso_par, thomsen_par, anysotropy, name, tilt_par=None):
    ''''
    Generate a pvd file for the elastic tensor with VTI or TTI anisotropy

    Parameters
    ----------
    mesh: `firedrake.Mesh`
        Firedrake mesh object
    iso_par: `list`
        List of isotropic parameters [vP, vS, rho]
    thomsen_par: `list`
        List of Thomsen parameters [epsilon, gamma, delta]
    anysotropy: `str`
        Type of anisotropy ('weak' or 'exact')
    name: `str`
        Name of the file
    tilt_par: `list`, optional
        List of TTI parameters [theta, phi]

    Returns
    -------
    None
    '''

    vP, vS, rho = iso_par
    epsilon, gamma, delta = thomsen_par

    C1_vti, comp_C1 = c_vti_tensor(vP, vS, rho, epsilon,
                                   gamma, delta, anysotropy)
    C2_vti, comp_C2 = c_vti_tensor(1.5 * vP, vP / 4., 2.5 * rho,
                                   0.3, 0.4, 0.2, anysotropy)

    if tilt_par is not None:
        theta, phi = tilt_par
        C1_tti, comp_C1 = c_tti_tensor(C1_vti, theta, phi)
        C2_tti, comp_C2 = c_tti_tensor(C2_vti, theta, phi)

    # Show tensor components
    print("C1 components:", comp_C1)
    print("C2 components:", comp_C2)

    # Determine the number of components
    c1_arr = np.array(list(comp_C1.values()))
    c2_arr = np.array(list(comp_C2.values()))
    if tilt_par is None:
        shape = (3, 2)
        name = 'C_vti'
    else:
        n_comp = len(comp_C1.keys())
        if not (n_comp % 3 == 0):
            c1_arr = np.append(c1_arr, np.zeros(3 - n_comp % 3))
            c2_arr = np.append(c2_arr, np.zeros(3 - n_comp % 3))
            n_comp += 3 - n_comp % 3
        shape = (3, n_comp // 3)
        name = 'C_tti'

    # Reshape the arrays to match the shape
    c1_arr = c1_arr.reshape(shape)
    c2_arr = c2_arr.reshape(shape)

    # Space function
    W = fire.TensorFunctionSpace(mesh, 'CG', 1, shape=shape)

    # Constitutive tensor for TTI
    C_ani = fire.Function(W, name='C_tti')

    # Generate a simple model
    x = fire.SpatialCoordinate(mesh)
    left_region = fire.conditional(x[0] <= 0.5, 1, 0)

    # Create full tensor expressions
    C1 = fire.as_tensor(c1_arr)
    C2 = fire.as_tensor(c2_arr)

    # Combined expression
    C_combined = C1 * left_region + C2 * (1 - left_region)

    # Inteporlate to function space
    C_ani.interpolate(C_combined)

    # Save file
    outfile = fire.VTKFile("output/" + name + ".pvd")
    outfile.write(C_ani)


Lx = 1.           # Length of the rectangle
Ly = 1.           # Width of the rectangle
lmax = 0.05       # Maximum edge length
lmin = lmax       # Minimum edge length

# Create rectangular mesh
nx, ny = int(Lx / lmax), int(Ly / lmax)
mesh = fire.RectangleMesh(nx, ny, Lx, Ly)

# Data
vP = 1.5  # P-wave velocity [km/s]
vS = 0.75  # S-wave velocity [km/s]
rho = 1e3  # Density [kg/m³]
epsilon = 0.2  # Thomsen parameter epsilon
gamma = 0.3  # Thomsen parameter gamma
delta = 0.1  # Thomsen parameter delta
theta = 30.  # Tilt angle in degrees
phi = 0.  # azimuth angle in degrees (phi = 0: 2D case)

gen_tensor_pvd(mesh, [vP, vS, rho], [epsilon, gamma, delta],
               'exact', 'c_vti_exact')
gen_tensor_pvd(mesh, [vP, vS, rho], [epsilon, gamma, delta],
               'weak', 'c_vti_weak')

# gen_tensor_pvd(mesh, [vP, vS, rho], [epsilon, gamma, delta],
#                'exact', 'c_tti_exact', tilt_par=[theta, phi])
# gen_tensor_pvd(mesh, [vP, vS, rho], [epsilon, gamma, delta],
#                'weak', 'c_tti_weak', tilt_par=[theta, phi])
