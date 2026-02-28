# This file contains methods for sizing an absorbing layer

import numpy as np

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
        Size  parameter of the absorbing layer (F_L)
    a : `float`
        Adimensional propagation speed parameter (a = z / f, z = c / l)
        Also, "z" parameter is the inverse of the minimum Eikonal (1 / phi_min)
    m : `int`, optional
        Vibration mode. Default is 1 (Fundamental mode)
    s : `float`, optional
        Damping ratio. Default is 0.999
    typ : `str`, optional
        Type of function to be computed. Default is 'FL'
        Options: 'FL' (size layer criterion) or 'CR' (reflection coeficient)

    Returns
    -------
    CritFL = CR - RF : `float`
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

    if typ == 'CR':
        return CR

    # Attenuation amplitude factor
    AS = 1 + (1 / 8) * (s * m * a / x)**2
    ax0 = m * np.pi * AS
    ax1 = (1 - s**2)**0.5
    ax2 = s / ax1
    ax3 = (2 * np.pi * x / a) * AS
    RF = abs(np.exp(-s * ax0) * (np.cos(ax1 * ax0)
                                 + ax2 * np.sin(ax1 * ax0)) * np.cos(ax3))

    if typ == 'FL':
        return CR - RF


def calc_zero(xini, a, tol, nz=1):
    '''
    Compute several parameter sizes for the absorbing layer

    Parameters
    ----------
    xini : `float`
        Initial guess for size parameter of the absorbing layer (F_L)
    a : `float`
        Adimensional propagation speed parameter (a = z / f, z = c / l)
        Also, "z" parameter is the inverse of the minimum Eikonal (1 / phi_min)
    tol : `float`
        Tolerance for the n-th root fo the function f_layer(x, a)
    nz : `int`, optional
        Number of layer sizes calculated. Default is 1

    Returns
    -------
    x : `float`
        Size  parameter of the absorbing layer (F_L)
    '''

    if nz == 1:
        x = tol * round(xini / tol)
    else:
        x = xini

    # Initial tolerances
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


def loop_roots(a, lmin, lref, max_roots, tol_rel=1e-3,
               show_ig=True, monitor=False):
    '''
    Loop to calculate the size parameter for the absorbing layer

    Parameters
    ----------
    a : `float`
        Adimensional propagation speed parameter (a = z / f, z = c / l)
        Also, "z" parameter is the inverse of the minimum Eikonal (1 / phi_min)
    lmin : `float`
        Minimal dimension of finite element in mesh
    lref : `float`
        Reference length for the size of the absorbing layer
    nz : `int`, optional
        Number of layer sizes to be calculated. Default is 5
    tol_rel : `float`, optional
        Relative convergence tolerance w.r.t the initial guess. Default is 1e-3
    show_ig : `bool`, optional
        Print the initial guess for the size parameter. Default is True
    monitor : `bool`, optional
        Print the parameter sizes of the absorbing layer. Default is False

    Returns
    -------
    FLpos : `list`
        Possible size parameters for the absorbing layer without rounding
    '''

    # Initial guess
    FLmin = 0.5 * lmin / lref
    if show_ig:
        print("Initial Guess for Size Parameter: {:.4f}".format(
            FLmin), flush=True)

    # Number of digits to round the size parameter
    dig_x = 13

    # Root Tolerance
    tol = 10**int(np.log10(tol_rel * FLmin))

    x = FLmin
    FLpos = []  # Size parameter
    for i in range(max_roots):

        # Calculating the size parameter
        x = calc_zero(x, a, tol, nz=i + 1)

        # Checking for duplicate values
        if round(x, 4) in np.round(FLpos, 4):
            x = calc_zero(x + tol, a, tol, nz=i + 1)

        # Rounding the size parameter
        x_rnd = round(x, dig_x)
        FLpos.append(x_rnd)

        # Monitoring the size parameter
        if monitor:
            print("**************** Possible FL ****************", flush=True)
            print(f"Root {i + 1}: {x_rnd: >} - Res: {f_layer(x_rnd, a): >}",
                  flush=True)

    return FLpos


def calc_size_lay(fref, z_par, lmin, lref, nz=5, n_root=1, tol_rel=1e-3,
                  layer_based_on_mesh=True, monitor=False):
    '''
    Calculate the lenght of the absorbing layer

    Parameters
    ----------
    freq_ref : `float`
        Reference frequency of the wave
    z_par : `float`
        Inverse of the minimum Eikonal (Equivalent to c_bound/lref)
    lmin : `float`
        Minimal dimension of finite element in mesh
    lref : `float`
        Reference length for the size of the absorbing layer
    nz : `int`, optional
        Number of layer sizes to be calculated. Default is 5
    n_root : `int`, optional
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
        Size of the absorbing layer
    ele_pad : `int`
        Approximated number of elements in the layer of edge length 'lmin'
    d_norm : `float`
        Normalized element size (lmin / pad_len)
    a : `float`
        Adimensional propagation speed parameter (a = z / f, z = c / l)
        Also, "z" parameter is the inverse of the minimum Eikonal (1 / phi_min)
    FLpos : `list`
        Possible size parameters for the absorbing layer without rounding
    '''

    # Visualizing parameters for computing layer size
    print("\nComputing Size for Absorbing Layer", flush=True)

    # Parameters for the absorbing layer
    aux0 = "Parameter z (1/s): {:.4f},".format(z_par)
    a = z_par / fref  # Adimensional parameter
    aux1 = "Parameter a (adim): {:.4f}".format(a)
    print(aux0, aux1, flush=True)

    # Minimum and reference length of the layer
    aux2 = "Minimum Mesh Length (km): {:.4f},".format(lmin)
    aux3 = "Reference Length (km): {:.4f}".format(lref)
    print(aux2, aux3, flush=True)

    # Maximum number of sizes to be computed
    nz = max(1, n_root + 1, nz)

    # Calculating roots of size parameter function
    FLpos = loop_roots(a, lmin, lref, nz, tol_rel=tol_rel, monitor=monitor)

    # Reflection coefficients
    CRpos = np.round(np.array([f_layer(x, a, typ="CR") for x in FLpos]), 4)

    # Visualizing options for layer size
    format_FL = ', '.join(['{:.4f}'.format(x) for x in FLpos])
    print("Options for FL: [{}]".format(format_FL), flush=True)
    format_CR = ', '.join(['{:.4f}'.format(x) for x in CRpos])
    print("Options for CR: [{}]".format(format_CR), flush=True)
    format_lay = ', '.join(['{:.4f}'.format(x * lref) for x in FLpos])
    print("Options for Layer Size (km): [{}]".format(format_lay), flush=True)
    format_ele = [int(x * lref / lmin) for x in FLpos]
    print("Aprox. Number of Elements ({:.3f} km) in Layer: {}".format(
        lmin, format_ele), flush=True)

    # Selecting a size
    F_L = FLpos[n_root - 1]

    # Size of the absorving layer
    pad_len = F_L * lref

    print("Selected Parameter Size FL: {:.4f}".format(F_L), flush=True)
    print("Selected Layer Size (km): {:.4f}".format(pad_len), flush=True)

    # Approximated number of elements in the layer of edge length 'lmin'
    ele_pad = format_ele[n_root - 1]

    if layer_based_on_mesh:
        F_L, pad_len, ele_pad = roundFL(lmin, lref, F_L)

    # Normalized element size
    d_norm = lmin / pad_len
    print("Normalized Element Size (adim): {0:.5f}".format(d_norm), flush=True)

    return F_L, pad_len, ele_pad, d_norm, a, FLpos


def roundFL(lmin, lref, F_L):
    '''
    Adjust the layer parameter based on the element size to get
    an integer number of elements within the layer

    Parameters
    ----------
    F_L : `float`
        Size parameter of the absorbing layer
    lmin : `float`
        Minimum mesh size
    lref : `float`
        Reference length for the size of the absorbing layer

    Returns
    -------
    F_L : `float`
        Modified size parameter of the absorbing layer according to mesh size
    pad_len : `float`
        Modified size of the absorbing layer
    ele_pad : `int`
        Number of elements in the layer of edge length 'lmin'
    '''

    # Adjusting the parameter size of the layer
    F_L = (lmin / lref) * np.ceil(lref * F_L / lmin)

    # New size of the absorving layer
    pad_len = F_L * lref

    # Number of elements in the layer
    ele_pad = int(pad_len / lmin)

    print("\nModifying Layer Size Based on the Element Size", flush=True)
    print("Modified Parameter Size FL: {:.4f}".format(F_L), flush=True)
    print("Modified Layer Size (km): {:.4f}".format(pad_len), flush=True)
    print("Elements ({:.3f} km) in Layer: {}".format(
        lmin, ele_pad), flush=True)

    return F_L, pad_len, ele_pad
