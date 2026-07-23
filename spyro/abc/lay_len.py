# This file contains methods for sizing an absorbing layer
from ..io.basicio import parallel_print as pprint
from numpy import array, ceil, cos, exp, floor, log10, pi, round, sin

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender.


def f_layer(x, a_par, vibration_mode=1, damping_ratio=0.999, function_type='FL'):
    """
    Function whose zeros are solution for the parameter size of the layer.

    Parameters
    ----------
    x : `float`
        Size  parameter of the absorbing layer (F_L).
    a_par : `float`
        Adimensional propagation speed parameter (a = z / f, z = c / l).
        Also, 'z' parameter is the inverse of the minimum Eikonal (1 / phi_min).
    vibration_mode : `int`, optional
        Vibration mode (m). Default is 1 (Fundamental mode).
    damping_ratio : `float`, optional
        Damping ratio (s). Default is 0.999.
    function_type : `str`, optional
        Type of function to be computed. Default is 'FL'.
        Options: 'FL' (size layer criterion) or 'CR' (reflection coeficient).

    Returns
    -------
    CritFL: `float`
        Value of the function for the size criterion computed as CritFL = CR - RF.
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
    """

    # Reflection coefficient
    s = damping_ratio
    s2 = damping_ratio ** 2.
    m = vibration_mode
    ma = vibration_mode * a_par
    CR = abs(s2 / (s2 + (4. * x / ma) ** 2.))

    if function_type == "CR":
        return CR

    # Attenuation amplitude factor
    AS = 1. + (1. / 8.) * (s * ma / x)**2.
    ax0 = m * pi * AS
    ax1 = (1. - s2)**0.5
    ax2 = s / ax1
    ax3 = (2. * pi * x / a_par) * AS
    RF = abs(exp(-s * ax0) * (cos(ax1 * ax0) + ax2 * sin(ax1 * ax0)) * cos(ax3))

    if function_type == "FL":
        return CR - RF


def calc_zero(xini, a_par, tol, nz=1):
    """Compute several parameter sizes for the absorbing layer.

    Parameters
    ----------
    xini : `float`
        Initial guess for size parameter of the absorbing layer (F_L).
    a_par : `float`
        Adimensional propagation speed parameter (a = z / f, z = c / l).
        Also, "z" parameter is the inverse of the minimum Eikonal (1 / phi_min).
    tol : `float`
        Tolerance for the n-th root fo the function f_layer(x, a).
    nz : `int`, optional
        Number of layer sizes calculated. Default is 1.

    Returns
    -------
    x : `float`
        Size  parameter of the absorbing layer (F_L).
    """

    if nz == 1:
        x = tol * round(xini / tol)
    else:
        x = xini

    # Initial tolerances
    f_tol = tol
    tol_ref = tol / 100.

    while abs(f_layer(x, a_par)) > f_tol or f_tol > tol_ref:

        # Identifying neighborhood of the root
        if (abs(f_layer(x, a_par)) <= f_tol
                or f_layer(x, a_par) * f_layer(x - f_tol, a_par) < 0) and x > xini + f_tol:

            # Adjusting initial guess and tolerance
            x = f_tol * floor((x - f_tol) / f_tol)
            f_tol *= 0.1

        x += f_tol  # Next step

        if f_tol < 1e-16:  # Avoiding infinite loop
            break

    return x


def loop_roots(a_par, lmin, lref, max_roots, tol_rel=1e-3, show_ig=True, monitor=False):
    """Loop to calculate the size parameter for the absorbing layer.

    Parameters
    ----------
    a_par : `float`
        Adimensional propagation speed parameter (a = z / f, z = c / l).
        Also, 'z' parameter is the inverse of the minimum Eikonal (1 / phi_min).
    lmin : `float`
        Minimal dimension of finite element in mesh.
    lref : `float`
        Reference length for the size of the absorbing layer.
    nz : `int`, optional
        Number of layer sizes to be calculated. Default is 5.
    tol_rel : `float`, optional
        Relative convergence tolerance w.r.t the initial guess. Default is 1e-3.
    show_ig : `bool`, optional
        Print the initial guess for the size parameter. Default is `True`.
    monitor : `bool`, optional
        Print the parameter sizes of the absorbing layer. Default is `False`.

    Returns
    -------
    FLpos : `list`
        Possible size parameters for the absorbing layer without rounding.
    """

    # Initial guess
    FLmin = 0.5 * lmin / lref
    if show_ig:
        pprint(f"Initial Guess for Size Parameter: {FLmin:.4f}")

    # Number of digits to round the size parameter
    dig_x = 13

    # Root Tolerance
    tol = 10**int(log10(tol_rel * FLmin))

    x = FLmin
    FLpos = []  # Size parameter
    for i in range(max_roots):

        # Calculating the size parameter
        x = calc_zero(x, a_par, tol, nz=i + 1)

        # Checking for duplicate values
        if round(x, 4) in round(FLpos, 4):
            x = calc_zero(x + tol, a_par, tol, nz=i + 1)

        # Rounding the size parameter
        x_rnd = round(x, dig_x)
        FLpos.append(x_rnd)

        # Monitoring the size parameter
        if monitor:
            pprint("**************** Possible FL ****************")
            pprint(f"Root {i + 1}: {x_rnd: >} - Res: {f_layer(x_rnd, a_par): >}")

    return FLpos


def calc_size_lay(fref, z_par, lmin, lref, nz=5, n_root=1, tol_rel=1e-3,
                  layer_based_on_mesh=True, monitor=False):
    """Calculate the lenght of the absorbing layer.

    Parameters
    ----------
    freq_ref : `float`
        Reference frequency of the wave.
    z_par : `float`
        Inverse of the minimum Eikonal (Equivalent to c_bound/lref).
    lmin : `float`
        Minimal dimension of finite element in mesh.
    lref : `float`
        Reference length for the size of the absorbing layer.
    nz : `int`, optional
        Number of layer sizes to be calculated. Default is 5.
    n_root : `int`, optional
        n-th Root selected for the size of the absorbing layer. Default is 1.
    tol_rel : `float`, optional
        Relative convergence tolerance w.r.t the initial guess. Default is 1e-3.
    monitor : `bool`, optional
        Print the parameter sizes of the absorbing layer. Default is False.

    Returns
    -------
    factor_length_pad : `float`
        Size  parameter of the absorbing layer.
    pad_length : `float`
        Size of the absorbing layer.
    ele_pad : `int`
        Approximated number of elements in the layer of edge length equal to 'lmin'.
    d_norm : `float`
        Normalized element size (lmin / pad_length).
    a_par : `float`
        Adimensional propagation speed parameter (a = z / f, z = c / l).
        'z' parameter is the inverse of the minimum Eikonal (1 / phi_min).
    FLpos : `list`
        Possible size parameters for the absorbing layer without rounding.
    """

    # Visualizing parameters for computing layer size
    pprint("\nComputing Size for Absorbing Layer")

    # Parameters for the absorbing layer
    a_par = z_par / fref  # Adimensional parameter
    pprint(f"Parameter 'z' (1/s): {z_par:.4f} - Parameter 'a' (adim): {a_par:.4f}")

    # Minimum and reference length of the layer
    pprint(f"Minimum Mesh Length (km): {lmin:.4f} - Reference Length (km): {lref:.4f}")

    # Maximum number of sizes to be computed
    nz = max(1, n_root + 1, nz)

    # Calculating roots of size parameter function
    FLpos = loop_roots(a_par, lmin, lref, nz, tol_rel=tol_rel, monitor=monitor)

    # Reflection coefficients
    CRpos = round(array([f_layer(x, a_par, function_type="CR") for x in FLpos]), 4)

    # Visualizing options for layer size
    format_FL = ', '.join([f"{x:.4f}" for x in FLpos])
    pprint(f"Options for FL: [{format_FL}]")
    format_CR = ', '.join([f"{x:.4f}" for x in CRpos])
    pprint(f"Options for CR: [{format_CR}]")
    format_lay = ', '.join([f"{x * lref:.4f}" for x in FLpos])
    pprint(f"Options for Layer Size (km): [{format_lay}]")
    format_ele = [int(x * lref / lmin) for x in FLpos]
    pprint(f"Aprox. Number of Elements ({lmin:.3f} km) in Layer: {format_ele}")

    # Selecting a size
    factor_length_pad = FLpos[n_root - 1]

    # Size of the absorving layer
    pad_length = factor_length_pad * lref
    pprint(f"Selected Parameter Size FL: {factor_length_pad:.4f}")
    pprint(f"Selected Layer Size (km): {pad_length:.4f}")

    # Approximated number of elements in the layer of edge length 'lmin'
    ele_pad = format_ele[n_root - 1]

    if layer_based_on_mesh:
        factor_length_pad, pad_length, ele_pad = roundFL(lmin, lref, factor_length_pad)

    # Normalized element size
    d_norm = lmin / pad_length
    pprint(f"Normalized Element Size (adim): {d_norm:.5f}")

    return factor_length_pad, pad_length, ele_pad, d_norm, a_par, FLpos


def roundFL(lmin, lref, factor_length_pad):
    """Adjust layer parameter to enforce an integer number of elements within the layer.

    Parameters
    ----------
    factor_length_pad : `float`
        Size parameter of the absorbing layer.
    lmin : `float`
        Minimum mesh size.
    lref : `float`
        Reference length for the size of the absorbing layer.

    Returns
    -------
    factor_length_pad : `float`
        Modified size parameter of the absorbing layer according to mesh size.
    pad_length : `float`
        Modified size of the absorbing layer.
    ele_pad : `int`
        Number of elements in the layer of edge length 'lmin'.
    """

    # Adjusting the parameter size of the layer
    factor_length_pad = (lmin / lref) * ceil(lref * factor_length_pad / lmin)

    # New size of the absorving layer
    pad_length = factor_length_pad * lref

    # Number of elements in the layer
    ele_pad = int(pad_length / lmin)

    pprint("\nModifying Layer Size Based on the Element Size")
    pprint(f"Modified Parameter Size FL: {factor_length_pad:.4f}")
    pprint(f"Modified Layer Size (km): {pad_length:.4f}")
    pprint(f"Elements ({lmin:.3f} km) in Layer: {ele_pad}")

    return factor_length_pad, pad_length, ele_pad
