import numpy as np
import ipdb

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


def f_layer(x, a, m=1, s=0.999, typ='FL'):
    '''
    Function whose zeros are solution for the parameter size of the layer

    Parameters
    ----------
    x : `float`
        Size  parameter of the absorbing layer (FL)
    a : `float`
        Adimensional parameter for the absorbing layer (a = z/f, z = c/l)
    m : `float`, optional
        Vibration mode. Default is 1 (Fundamental mode)
    s : `float`, optional
        Damping ratio. Default is 0.999
    typ : `str`, optional
        Type of function to be computed. Default is 'FL'
        Options: 'FL' (size layer criterion) or 'CR' (reflection coeficient).

    Returns
    -------
    CritFL = CR - RF: `float`
        Value of the function for the size criterion
    CR: `float`
        Value for the reflection coefficient4

    Examples
    -------
    Let c = 1.5 km/s, l = 1.2 km, m=1, s=0.999

    Zeros for  f = 5Hz, a = 0.25
    Expected: F_L1=0.1917, F_L2=0.2682, F_L3=0.2981, F_L4=0.4130, F_L5=0.4244
    Tol 1e-2: F_L1=0.1917, F_L2=0.2682, F_L3=0.2981, F_L4=0.4131, F_L5=0.4244
    Tol 1e-3: F_L1=0.1917, F_L2=0.2682, F_L3=0.2981, F_L4=0.4131, F_L5=0.4244
    Tol 1e-4: F_L1=0.1917, F_L2=0.2682, F_L3=0.2981, F_L4=0.4130, F_L5=0.4244
    Tol 1e-5: F_L1=0.1917, F_L2=0.2682, F_L3=0.2981, F_L4=0.4130, F_L5=0.4244

    Zeros for f = 2.25Hz, a = 0.555 (without rounding)
    Expected: F_L1=0.4259, F_L2=0.5959, F_L3=0.6624, F_L4=0.9179, F_L5=0.9431
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
    Compute several parameter sizes for the absorbing layer

    Parameters
    ----------
    xini : `float`
        Initial guess for size parameter of the absorbing layer (FL)
    a : `float`
        Adimensional parameter for the absorbing layer (a = z/f, z = c/l)
    tol : `float`
        Tolerance for the n-th root fo the function f_layer(x, a)
    nz : `int`, optional
        Number of layer sizes calculated. Default is 1

    Returns
    -------
    x : `float`
        Size  parameter of the absorbing layer (FL)
    '''

    if nz == 1:
        x = tol * round(xini / tol)
    else:
        x = xini

    f_tol = tol
    tol_ref = tol / 100

    while abs(f_layer(x, a)) > f_tol or f_tol > tol_ref:

        # Identifying neighborhood of the root
        if (abs(f_layer(x, a)) <= f_tol
            or f_layer(x, a) * f_layer(x - f_tol, a) < 0) \
                and x > xini + f_tol:

            # Adjusting initial guess and tolerance
            x = f_tol * np.floor((x - f_tol) / f_tol)
            f_tol *= 0.1

        x += f_tol  # Next step

        if f_tol < 1e-16:  # Avoiding infinite loop
            break

    return x


def calc_size_lay(Wave, Eikonal, nz=5, crtCR=1, tol_rel=1e-3, monitor=False):
    '''
    Calculate the lenght of the absorbing layer

    Parameters
    ----------
    Wave : `wave`
            Wave object
    Eikonal : `eikonal`
            Eikonal object
    nz : `int`, optional
        Number of layer sizes calculated. Default is 5
    crtCR : `int`, optional
        n-th Root selected for the size of the absorbing layer. Default is 1
    tol_rel : `float`, optional
        Relative convergence tolerance w.r.t the initial guess. Default is 1e-3
    monitor : `bool`, optional
        Print the parameter sizes of the absorbing layer. Default is False

    Returns
    -------
    F_L : `float`
        Size  parameter of the absorbing layer
    pad_len : `float`
        Size of damping layer
    '''

    # fref: Reference frequency
    fref = Wave.frequency
    # z_par: Inverse of minimum Eikonal (Equivalent to c_bound / lref)
    z_par = Eikonal.eik_bnd[0][3]
    print(f"Parameter z={round(z_par, 4)}, Reference Frequency: {round(fref, 4)}")

    # lmin: Minimal dimension of finite element in mesh
    lmin = Wave.lmin
    # lref: Reference length for the size of the absorbing layer
    lref = Eikonal.eik_bnd[0][4]
    print(f"Minimum Mesh Length: {lmin}, Reference Length: {round(lref, 4)}")

    a = z_par / fref  # Adimensional parameter
    FLmin = 0.5 * lmin / lref  # Initial guess
    print(f"Parameter a: {round(a, 4)}, Initial Guess: {round(FLmin, 4)}")

    x = FLmin
    FLpos = []  # Size factor
    crtCR = min(crtCR - 1, nz - 1)  # Size selected. Default: 0
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
            print(x_rnd, f_layer(x_rnd, a))

    # Reflection coefficients
    CRpos = np.array([round(abs(f_layer(x, a, typ="CR")), 4) for x in FLpos])

    # Selecting a size
    F_L = FLpos[crtCR]

    # Size of damping layer
    pad_len = F_L * lref

    # Visualizing options for layer size
    print('Selected F_L:', round(F_L, 4))
    print('Options for F_L:', [round(float(x), 4) for x in FLpos])
    print('Elements for F_L:', [int(x * lref / lmin) for x in FLpos])
    print('Options for CR:', CRpos)
    print('Layer Size:', round(pad_len, 4), '(km)')

    return F_L, pad_len
