import numpy as np
import ipdb

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014


def F(x, a, m=1, s=0.999, typ='FL'):
    '''
    Function whose zeros are solution for the layer size

    Example: c = 1.5 km/s,  l = 1.2km, f = 5Hz
    Zeros for s = 0.999, a = 0.25 = 1.5/(1.2*5) without rounding
    Expected: F_L1=0.1917, F_L2=0.2682, F_L3=0.2981, F_L4=0.4130, F_L5=0.4244
    Tol 1e-2: F_L1=0.1917, F_L2=0.2682, F_L3=0.2981, F_L4=0.4131, F_L5=0.4244
    Tol 1e-3: F_L1=0.1917, F_L2=0.2682, F_L3=0.2981, F_L4=0.4131, F_L5=0.4244
    Tol 1e-4: F_L1=0.1917, F_L2=0.2682, F_L3=0.2981, F_L4=0.4130, F_L5=0.4244
    Tol 1e-5: F_L1=0.1917, F_L2=0.2682, F_L3=0.2981, F_L4=0.4130, F_L5=0.4244

    Example: c = 1.5 km/s,  l = 1.2km, f = 5Hz
    Zeros for s = 0.999, a = 0.555 = 1.5/(1.2*2.25) without rounding
    Expected: F_L1=0.4267, F_L2=0.5971, F_L3=0.6637, F_L4=0.9197, F_L5=0.9450
    Tol 1e-2: F_L1=0.4258, F_L2=0.5956, F_L3=0.6621, F_L4=0.9179, F_L5=0.9431
    Tol 1e-3: F_L1=0.4259, F_L2=0.5959, F_L3=0.6624, F_L4=0.9179, F_L5=0.9431
    Tol 1e-4: F_L1=0.4259, F_L2=0.5959, F_L3=0.6624, F_L4=0.9179, F_L5=0.9431
    Tol 1e-5: F_L1=0.4259, F_L2=0.5959, F_L3=0.6624, F_L4=0.9179, F_L5=0.9431
    '''
    # Reflection coefficient
    CR = abs(s**2 / (s**2 + (4 * x / (m * a))**2))
    ax0 = m * np.pi * (1 + (1 / 8) * (s * m * a / x)**2)
    ax1 = (1 - s**2)**0.5
    ax2 = s / ax1
    ax3 = (2 * np.pi * x / a) * (1 + (1 / 8) * (s * m * a / x)**2)
    # Attenuation coefficient
    RF = abs(np.exp(-s * ax0) * (np.cos(ax1 * ax0)
                                 + ax2 * np.sin(ax1 * ax0)) * np.cos(ax3))

    if typ == 'FL':
        return CR - RF
    elif typ == 'CR':
        return CR


def calcZero(xini, a, tol, nz=1):
    '''
    Loop for calculating several layer sizes
    '''

    if nz == 1:
        x = tol * round(xini / tol)
    else:
        x = xini

    f_tol = tol
    tol_ref = tol / 100

    while abs(F(x, a)) > f_tol or f_tol > tol_ref:

        if (abs(F(x, a)) <= f_tol or F(x, a) * F(x - f_tol, a) < 0) \
                and x > xini + f_tol:
            x = f_tol * np.floor((x - f_tol) / f_tol)
            f_tol *= 0.1
        x += f_tol

        if f_tol < 1e-16:
            break  # add to code

    return x


def calc_size_lay(Wave, Eikonal, nz=5, crtCR=0, tol_rel=1e-3, monitor=False):
    """
    Calculate the lenght of the absorbing layer

    fref: Reference frequency
    z_par: Inverse of minimum Eikonal (Equivalent to c_bound / lref)
    lmin: Minimal dimension of finite element
    lref: Reference length for the size of the absorbing layer
    nz: Number of layer sizes calculated
    crtCR: Position in CRpos. Default: 0
    """

    fref = Wave.frequency
    z_par = Eikonal.eik_bnd[0][3]
    print(f"Parameter z = {round(z_par, 4)}, , Reference Frequency: {fref}")

    lmin = Eikonal.lmin
    lref = Eikonal.eik_bnd[0][4]
    print(f"Minimum Mesh Length: {lmin}, Reference Length: {lref}")

    a = z_par / fref  # Adimensional parameter
    FLmin = 0.5 * lmin / lref  # Initial guess
    print(f"Parameter a: {round(a, 4)}, Initial Guess: {FLmin}")

    x = FLmin
    FLpos = []  # Size factor
    crtCR = min(crtCR, nz - 1)  # Position in CRpos. Default: 0
    dig_x = 13

    tol = 10**int(np.log10(tol_rel * FLmin))
    for i in range(1, nz + 1):
        x = calcZero(x, a, tol, i)

        # Checking for duplicate values
        if i > 1 and x == FLpos[i - 2]:
            x = calcZero(x, a, tol, i)

        x_rnd = round(x, dig_x)
        FLpos.append(x_rnd)

        if monitor:
            print("********")
            print("Possible FL")
            print(x_rnd, F(x_rnd, a))

    # Reflection coefficients
    CRpos = np.array([round(abs(F(x, a, typ="CR")), 4) for x in FLpos])

    # Selecting a size
    F_L = FLpos[crtCR]

    # Size of damping layer
    pml = F_L * lref

    # Visualizing options for layer size
    print('Selected F_L:', round(F_L, 4))
    print('Options for F_L:', [round(float(x), 4) for x in FLpos])
    print('Elements for F_L:', [int(x * lref / lmin) for x in FLpos])
    print('Options for CR:', CRpos)
    print('Layer Size:', round(pml, 4), '(km)')

    return F_L, pml
