import numpy as np
from scipy.fft import fft
import sys

# Work from Ruben Andres Salas,
# Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva, Hybrid absorbing scheme based on hyperel-
# liptical layers with non-reflecting boundary conditions in scalar wave equations, Applied Mathematical
# Modelling (2022), doi: https://doi.org/10.1016/j.apm.2022.09.014

###############
# Execute this code after eikonal for iteration 0, for other iterations
# after finite element analysis
##############

def detLref(posCrit, possou):
    '''
    Determining the reference of the size of the absorbing layer
    '''
    # Defining Critical Source
    print('Defining Critical Source')
    # Distances between sources and the critical point
    # (with minimum eikonal at boundaries)
    souCrit = possou[np.argmin([np.linalg.norm(posCrit - p) for p in possou])]
    # Reference length for the size of the absorbing layer
    print('Defining Reference Length')
    lsrx = abs(posCrit[0] - souCrit[0])
    print(f'lsrx = {lsrx}')
    lsry = abs(posCrit[1] - souCrit[1])
    print(f'posCrit1 = {posCrit[1]}')
    print(f'souCrit1 = {souCrit[1]}')
    print(f'lsry = {lsry}')
    lref = np.linalg.norm(np.array([lsrx, lsry]))
    return lref


def F(x, a, m=1, s=0.999, typ='FL'):
    '''
    Function whose zeros are solution for layer size
    Zeros for s = 0.999, a = c/(lref*fref) = Z/fref = 0.25 = 1.5/(1.2*5) without rounding
    Expected: F_L1=0.1917, F_L2=0.2682, F_L3=0.2981, F_L4=0.4130, F_L5=0.4244
    Tol 1e-2: F_L1=0.1917, F_L2=0.2682, F_L3=0.2981, F_L4=0.4130, F_L5=0.4244
    Tol 1e-3: F_L1=0.1917, F_L2=0.2682, F_L3=0.2981, F_L4=0.4130, F_L5=0.4244
    Tol 1e-4: F_L1=0.1917, F_L2=0.2682, F_L3=0.2981, F_L4=0.4130, F_L5=0.4244

    Zeros for s = 0.999, a = c/(lref*fref) = 0.555 = 1.5/(1.2*2.25) without rounding
    Expected: F_L1=0.4267, F_L2=0.5971, F_L3=0.6637, F_L4=0.9197, F_L5=0.9450
    Tol 1e-4: F_L1=0.4268, F_L2=0.5971, F_L3=0.6638, F_L4=0.9197, F_L5=0.9450
    '''
    # Reflection coefficient
    CR = abs(s**2 / (s**2 + (4 * x / (m * a))**2))
    ax0 = m * np.pi * (1 + (1 / 8) * (s * m * a / x)**2)
    ax1 = (1 - s**2)**0.5
    ax2 = s / ax1
    ax3 = (2 * np.pi * x / a) * (1 + (1 / 8) * (s * m * a / x)**2)
    # Attenuation coefficient
    RF = abs(np.exp(-s * ax0) * (np.cos(ax1 * ax0) +
                                 ax2 * np.sin(ax1 * ax0)) * np.cos(ax3))

    if typ == 'FL':
        return CR - RF
    elif typ == 'CR':
        return CR


def calcZero(xini, a, nz=1):
    '''
    Loop for calculating layer sizes
    '''
    print(f'xini = {xini}')
    print('CalcZero')
    f_tol = 1e-5
    if xini >= 1.0:
        f_tol = 1e-3
    if nz == 1:
        x = f_tol * round(xini / f_tol)
    else:
        x = xini

    while abs(F(x, a)) > f_tol or f_tol > 1e-6:
        if (abs(F(x, a)) <= f_tol or F(x, a) * F(x - f_tol, a) < 0) \
                and x > xini + f_tol:
            x = f_tol * np.floor((x - f_tol) / f_tol)
            f_tol *= 0.1
        x += f_tol

        if f_tol < 1e-16:
            break

    print('CalcZero fim')
    return x


def redFL(F_L, lmin, lref):
    '''
    Adjust layer according to element size
    '''
    return (lmin / lref) * np.ceil(lref * F_L / lmin)


def calcp(x_rel, a, b, lim, pmax=20):
    '''
    p limit for hyperellipse
    r < 1 ensures that the point is inside the layer
    '''
    # Superness s= 0.5^(-1/n): Extreme points of Hyperellipse
    h = max(int(np.ceil(np.log(
        0.5) / np.log((1 / a + 1 / b) / (1 / x_rel[0] + 1 / x_rel[1])))), 2)
    rh = abs(x_rel[0] / a)**h + abs(x_rel[1] / b)**h
    print('Superness "Harm" - r:{:5.4f} - p:{}*'.format(rh, h))
    s = max(
        int(np.ceil(np.log(0.25) / np.log(x_rel[0] * x_rel[1] / (a * b)))), 2)
    rs = abs(x_rel[0] / a)**s + abs(x_rel[1] / b)**s
    print('Superness "Geom" - r:{:5.4f} - p:{}*'.format(rs, s))
    z = max(
        int(np.ceil(np.log(0.5) / np.log((x_rel[0] + x_rel[1]) / (a + b)))), 2)
    rz = abs(x_rel[0] / a)**z + abs(x_rel[1] / b)**z
    print('Superness "Arit" - r:{:5.4f} - p:{}*'.format(rz, z))
    r = p = 1
    while r >= 1 and p <= pmax:
        p += 1
        r = abs(x_rel[0] / a)**p + abs(x_rel[1] / b)**p
        # print('ParHypEll - r:{:5.4f} - p:{}'.format(r, p))
    print('ParHypEll - r:{:5.4f} - p:{}'.format(r, p))
    print('a(km):{:5.3f} - b(km):{:5.3f}'.format(a / 1e3, b / 1e3))
    print(
        'x0(km):{:5.3f} - y0(km):{:5.3f}'.format(x_rel[0] / 1e3,
                                                 x_rel[1] / 1e3))
    print('************************************')
    if lim == 'MIN':
        if rs < r:
            s = p
        if rz < r:
            z = p
        if rh < r:
            h = p
        p = max(p, s, z, h)
    elif lim == 'MAX':
        p = min(p, s, z, h)
    return p


def CalcFL(TipLay, Lx, Ly, fref, lmin, lref, Z, nexp, nz=5, crtCR=0):
    '''
    Calculate the lenght of absorption layer
    TipLay: Layer damping type ('rectangular' or 'hyperelliptical')
    fref: Reference frequency
    lmin: Minimal dimension of finite element
    lref: Reference length for the size of the absorbing layer
    Z: Inverse of minimum Eikonal
    nexp: Hyperellipse exponent for damping layer
    nz: Number of layer sizes calculated
    crtCR: Position in CRpos. Default: 0
    '''
    a = Z / fref  # print(a, Z,fref)
    print(f"a = {a}")
    print(f"fref = {fref}")
    FLpos = []
    crtCR = min(crtCR, nz-1)  # Position in CRpos. Default: 0
    print(f"lmin: {lmin}, lref: {lref}")
    FLmin = 2 * lmin / lref # passar lmin da camada de agua
    x = FLmin
    for i in range(1, nz + 1):
        x = calcZero(x, a, i)
        xred = redFL(x, lmin, lref)
        if i > 1 and xred == FLpos[i - 2]:
            x = calcZero(xred, a, i)
        FLpos += [redFL(x, lmin, lref)]
        print('********')
        print('Possible FL')
        print(x, F(x, a))
    CRpos = np.array([round(abs(F(x, a, typ='CR')), 4) for x in FLpos])
    indCR = np.array([crtCR])
    indFL = np.where(np.array(FLpos) < 1)[0]
    if indCR.size > 0 and indFL.size > 0:
        if FLpos[min(indCR)] < 1:
            F_L = FLpos[min(indCR)]
        else:
            F_L = FLpos[min(indFL)]
    else:
        F_L = FLpos[0]

    # Visualizing options for layer size
    print('F_L:', round(F_L, 4), '- Options for F_L:',
          [round(x, 4) for x in FLpos])
    print('Options for CR:', CRpos)

    if not TipLay == 'rectangular':
        pmlRect = F_L * lref
        bdom = Lx + 2 * pmlRect
    elif TipLay == 'hyperelliptical':
        hdom = Ly + 2 * pmlRect

        a = bdom / 2
        b = hdom / 2

        # Minimum allowed exponent
        x_rel = [0.5 * Lx + lmin, 0.5 * Ly + lmin]
        r = abs(x_rel[0] / a)**nexp + abs(x_rel[1] / b)**nexp
        # Exponent for ensuring pmlRect in the domain diagonal
        theta = np.arctan2(Ly, Lx)
        x_pml = [0.5 * Lx + pmlRect *
                 np.cos(theta), 0.5 * Ly + pmlRect * np.sin(theta)]
        rf = abs(x_pml[0] / a)**nexp + abs(x_pml[1] / b)**nexp
        print(
            'HypEll Limits: p:{}-r:{:3.4f}-rf:{:3.4f}'.format(nexp, r, rf))

        # Verification of hyperellipse exponent
        print('************************************')
        print('Minimum Exponent for Hyperellipse')
        p = calcp(x_rel, a, b, 'MIN')
        print('Maximum Exponent for Hyperellipse')
        pf = calcp(x_pml, a, b, 'MAX')
        if pf == p:
            pf += 1
        condA = nexp < p
        condB = nexp >= p and nexp > pf

        if condA or condB:
            print('Current Exponent: {}'.format(nexp))
            print('Minimum Exponent for Hyperellipse: {}'.format(p))
            print('Maximum Exponent for Hyperellipse: {}'.format(pf))
            if condA:
                sys.exit('Low Exponent for Hyperellipse')
            elif condB:
                sys.exit('High Exponent for Hyperellipse')
        else:
            print('Minimum Exponent for Hyperellipse: {}'.format(p))
            print('Current Exponent: {}'.format(nexp))
            print('Maximum Exponent for Hyperellipse: {}'.format(pf))

        print('************************************')

    # Size of damping layer
    pml = F_L * lref
    
    return F_L, pml


def detFref(histPcrit, f0, it_fwi, dt):
    '''
    Determines the reference frequency for a new layer length
    histPcrit: Transient response in at critical coordinates "posCrit"
    f0: Theorical central Ricker source frequency
    it_fwi: Iteration number of inversion process
    '''

    if it_fwi > 0:

        # Zero Padding for increasing smoothing in FFT
        y = np.concatenate([np.zeros(4*len(histPcrit)), histPcrit])
        # Number of sample points
        N = len(y)
        # Calculate the response in frequency domain at critical point
        yf = fft(y)  # FFT
        fe = 1.0 / (2.0 * dt)
        pfft = N//2 + N % 2
        xf = np.linspace(0.0, fe, pfft)

        # Minimun frequency excited
        fref = xf[np.abs(yf[0:pfft]).argmax()]

        del y, N, xf, yf
    else:  # Initial guess
        # Theorical central Ricker source frequency
        fref = f0

    return fref


def habc_size(HABC):
    '''
    Determines the size of the absorbing layer
    Lx, Ly: Original domain dimensions
    posCrit: Coordinates of critical point given by Eikonal
    possou: Positions of sources
    histPcrit: Transient response in at critical coordinates "posCrit"
    f0: Theorical central Ricker source frequency
    it_fwi: Iteration unmber of inversion process
    lmin: Minimal dimension of finite element
    Z: Inverse of minimum Eikonal
    TipLay: Layer damping type ('rectangular' or 'hyperelliptical')
    nexp: Hyperellipse exponent for damping layer. nexp = NaN for rectangular layers
    '''

    Lx = HABC.Lz
    Ly = HABC.Lx
    posCrit = HABC.posCrit
    possou = HABC.possou
    f0 = 3.37#HABC.initial_frequency
    it_fwi = HABC.it_fwi
    lmin = HABC.h_min
    dt = HABC.dt
    Z = HABC.Z
    nexp = HABC.nexp

    # Critical position for reference
    lref = detLref(posCrit, possou)
    # Determining the reference frequency
    fref = detFref(HABC.get_histPcrit(), f0, it_fwi, dt)
    
    # Absorbing layer size
    F_L, pml = CalcFL(HABC.TipLay, Lx, Ly, fref, lmin, lref, Z, nexp)

    ###############
    # Remesh of the domain adding the distance "pml" according to the case
    ##############

    return fref, F_L, pml, lref
