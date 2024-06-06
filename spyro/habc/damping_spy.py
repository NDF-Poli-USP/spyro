# from fenics import Function, pi, File
from firedrake import Function, pi, File, FunctionSpace, CellDiameter
from scipy.special import mathieu_modcem1
from scipy.optimize import broyden1
from math import gamma
import numpy as np
import os
import sys

# Work from Ruben Andres Salas,
# Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva, Hybrid absorbing scheme based on hyperel-
# liptical layers with non-reflecting boundary conditions in scalar wave equations, Applied Mathematical
# Modelling (2022), doi: https://doi.org/10.1016/j.apm.2022.09.014

###############
# Execute after remeshing domain with absorbing layer
##############


class Damping_field_calculator:
    """
    Calcules damping function across the field.

    Attributes
    ----------

    Methods
    -------

    """
    def __init__(self, Wave_obj):
        """
        Initializes 
        """
        self.mesh = Wave_obj.mesh
        self.pad_length = Wave_obj.abc_pad_length
        self.function_space = Wave_obj.function_space
        self.lref = Wave_obj.neik_reference_length
        self.layer_type = "rectangle"
        self.eta = Function(Wave_obj.function_space)
        self.length_z = Wave_obj.length_z
        self.length_x = Wave_obj.length_x
        self.neik_length_factor = Wave_obj.neik_length_factor
        self.neik_reference_velocity = Wave_obj.neik_velocity_value
        self.neik_time_value = Wave_obj.neik_time_value
        self.frequency = Wave_obj.frequency
        self.neik_location = Wave_obj.neik_location
        self.neik_dof = Wave_obj.noneikonal_dof
        self.neik_edge_length = self.get_neik_edge_length()
        self.damping_function()

    def get_neik_edge_length(self):
        V = self.function_space
        u = Function(V)
        cd = CellDiameter(self.mesh)
        u.interpolate(cd)
        cd_across_mesh = u.dat.data[:]
        return cd_across_mesh[self.neik_dof]

    def damping_function(self):
        """
        Determines the distribution of the damping within the absorbing
        layer from the calcultions of the maximum damping value
        mesh_Fin: Final mesh with absorbing layer
        TipLay: Layer damping type (Rectangular: 'REC' or Hyperelliptical: 'HYP')
        Lx, Ly: Original domain dimensions
        lref: Reference length for the size of the absorbing layer
        pml: Length of absrobing layer
        Z: Inverse of minimum Eikonal
        fref: Reference frequency
        lmin: Minimal dimension of finite element
        F_L: Parameterized length of absorbing layer
        c_ref: Valocity at critical point wit minimum Eikonal
        nexp: Hyperellipse exponent for damping layer. nexp = NaN for rectangular layers
        """
        print("Determining Damping Distribution")
        eta = self.eta
        eta_array = eta.dat.data[:]
        # Factor for frequency correction
        factLen = calcFact2D(
            self.layer_type,
            self.length_z,
            self.length_x,
            self.pad_length,
            0,
        )  # Change 0 to atribute nexp in future for hyper-ellipse
        # Min Eikonal = 1 / Z. Tau = kw*eikmin
        etacr = calcCritDamp(
            1 / self.neik_time_value,
            self.length_z,
            self.length_x,
            self.lref,
            factLen,
        )
        # Damping Function
        etalim = funDamp(
            self.mesh,
            self.layer_type,
            self.length_z,
            self.length_x,
            self.pad_length,
            0,  # NEXP change for kargs in all places
            eta_array,
            etacr,
            self.neik_time_value,
            self.frequency,
            self.neik_edge_length,
            self.neik_length_factor,
            self.neik_reference_velocity,
        )

        damp_file = File("/out/Damp.pvd")
        damp_file.write(eta)

        return eta, etalim


def regFreq(factLen, nexp, a, b):
    """
    Frequency factor for hiperellipses and hipercircles
    """
    factRec = (pi / 2) * (
        1 / a**2 + 1 / b**2
    ) ** 0.5  # Eval factor rectangle
    relfact = factLen / factRec
    p = 1.69453512285879
    return factRec * (1 - (1 - relfact) * (2 / nexp) ** p)


def minSemiAx(Ly, pml, CamComp=False):
    """
    CamComp: camada completa
    Calculating the minor semi-axis for rectangular layer
    """
    if CamComp:
        npml = 1
    else:
        npml = 0.5
    return 0.5 * Ly + npml * pml


def calcFact2D(TipLay, Lx, Ly, pml, nexp, CamComp=False):
    """
    Computing length factor for natural frequency
    """
    a = 0.5 * Lx + pml  # Major semi-axis
    if TipLay == "rectangle":
        """
        Rectangular layers
        https://www.sc.ehu.es/sbweb/fisica3/ondas/membrana_1/membrana_1.html
        """
        b = minSemiAx(Ly, pml)
        factLen = (pi / 2) * (1 / a**2 + 1 / b**2) ** 0.5  # Eval factor
    elif TipLay == "HYP":
        """
        Hyperlliptical layers
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.mathieu_modcem1.html#scipy.special.mathieu_modcem1
        """
        b = minSemiAx(Ly, pml)
        f = (a**2 - b**2) ** 0.5

        def F(x, m=0, psi0=np.arccosh(a / f)):
            """
            Modified Mathieiu's Function
            psi0 = arccosh(a/sqrt(a^2 - b^2)) = arccosh(a/f)
            mathieu_modcem1(m=0, x=2.6750449521966490, psi0=np.arccosh(2/(3)**0.5)=0.5493061443340551)[0]= 5.363046165143026e-17
            mathieu_modcem1(m=0, x=1.6748563428285737, psi0=0.7061880927645094)[
                            0] = 4.036310483679603e-16
            mathieu_modcem1(m, x, psi0)[0], x = q parameter = M01 (unknown)
            """
            # print(m, psi0, x, mathieu_modcem1(m, x, psi0)[0])
            return mathieu_modcem1(m, x, psi0)[0]

        """
        Modified Mathieiu's Function Order Zero, First Root
        alpha = Initial step from (2/f)* (alpha)**0.5 = (pi/2)*(1/a**2 + 1/b**2)**0.5
        """
        M01 = float(
            broyden1(
                F,
                0,
                f_tol=1e-14,
                alpha=(a**4 - b**4) * (pi / (4 * a * b)) ** 2,
            )
        )
        factLen = (2 / f) * (M01) ** 0.5  # Eval factor

        # Frequency factor for hiperellipses and hipercircles
        if (CamComp and nexp > 2) or not CamComp:
            factLen = regFreq(factLen, nexp, a, b)

    return factLen


def calcCritDamp(eikmin, Lx, Ly, lref, factLen):
    """
    Computing critical damping
    """
    # Aspect Ratio
    AspRatio = Lx / Ly
    # Geometry factor for frequency in original domain
    factEik = pi * (1 / Lx**2 + 1 / Ly**2) ** 0.5
    # Factor for estimation of frequency based on eikonal
    kwn = pi * (lref / Lx) * (1 + AspRatio**2) ** 0.5  # (lowest)
    # Estimated frequency for original domain
    weik = 1 / (kwn * eikmin)  # (lowest)
    # Estimated frequency for domain with layer
    wlay = weik * (factLen / factEik)
    # Critical damping
    etacr = 2 * wlay

    return etacr


def testPosDamp(mesh, Lx, Ly, pml, ref):
    """
    ref: ponderação da distribuição de amortecimento
    Mapping of positions inside of absorbing layer
    """
    mesh
    # Mesh coordinates
    mesh_coord = mesh.coordinates.dat.data[:]

    # identificar se esta na camada (posso retirar depois)
    refamx = np.abs(mesh_coord[:, 0] - (0.5 * Lx + pml))
    refamy = np.abs(mesh_coord[:, 1] - (0.5 * Ly + pml))
    condAmo = (refamx > 0.5 * Lx) | (refamy > 0.5 * Ly)

    # identificar os cantos e modificar amortecimento
    testc = (refamx >= 0.5 * Lx) & (refamy >= 0.5 * Ly)
    ref[testc] = (
        (refamx[testc] - 0.5 * Lx) ** 2 + (refamy[testc] - 0.5 * Ly) ** 2
    ) ** 0.5

    testx = (refamx > 0.5 * Lx) & (refamy < 0.5 * Ly)
    ref[testx] = refamx[testx] - 0.5 * Lx
    testy = (refamx < 0.5 * Lx) & (refamy > 0.5 * Ly)
    ref[testy] = refamy[testy] - 0.5 * Ly
    ref = ref / pml  # ref no artigo é a equação 12 xiref

    return condAmo


def coeDampFun(CRmin, kCR, d, psi):
    """
    Coefficients for damping functions
    """

    if CRmin == 0:
        psimin = 0
    else:
        psimin = kCR / (1 / CRmin - 1) ** 0.5
    aq = (psimin - d * psi) / (d**2 - d)
    bq = psi - aq
    return aq, bq, psimin


def CRminQua(
    d, kCR, xCR=None, typ="QUA", alpha=1, p=None, psi=None, a=None, F_L=None
):
    """
    Minimum coeficient reflection in QUA damping.
    Option 'MEF' is an auxiliary function to determine xCR
    """
    if typ == "QUA":  # Minimum coefficient reflection
        return (xCR * d) ** 2 / (kCR**2 + (xCR * d) ** 2) # Começando uma regreção e calulando o ponto inicia

    elif typ == "MEF":  # Unidimensional spourious reflection

        def Zi(p, alpha, Mass="COM"):  # Z1 e Z2 da equação 20 LUM é lumped com é consistente
            if Mass == "COM":
                m1 = 1 / 3
                m2 = 1 / 6
            elif Mass == "LUM":
                m1 = 1 / 2
                m2 = 0
            return m2 * (np.cos(alpha * p) - 1) / (m1 * (np.cos(alpha * p) + 1))

        # Dimensionless wavenumbers p_i
        p1 = p
        CRMdf = np.tan(p1 / 4) ** 2
        psiMdf = coeDampFun(CRMdf, kCR, d, psi)[2]
        p2 = p * (
            1 + 1 / 8 * (psiMdf * a / F_L) ** 2
        )  # For fundamental mode m=1
        Z1 = Zi(p1, alpha)
        Z2 = Zi(p2, alpha)
        num = (1 - Z1) * np.sin(p1) + (alpha * Z2 - 1) * np.sin(
            alpha * p2
        ) / alpha
        den = (1 - Z1) * np.sin(p1) - (alpha * Z2 - 1) * np.sin(
            alpha * p2
        ) / alpha
        CRMef = abs(num / den)
        return CRMef, coeDampFun(CRMef, kCR, d, psi)[2] / d


def xCRVert(x0, x1, x2, y0, y1, y2, xCRInf):
    """
    Quadratic regression for CRmin
    """
    m1 = (y1 - y0) / (x1 - x0)
    m2 = (y2 - y0) / (x2 - x0)
    a = (m2 - m1) / (x2 - x1)
    b = m2 - a * (x2 + x0)
    c = y2 - x2 * (m2 - a * x0)

    # Vertex or minimum positive root
    if -b / (2 * a) < xCRInf:  # Da equação 28
        ind = np.where(np.roots([a, b, c]) >= xCRInf)[0][0]
        return np.roots([a, b, c])[ind], 0
    else:
        return -b / (2 * a), -(b**2) / (4 * a) + c


def estXCR(psi, d, kCR, pCR, CRmax, a, F_L, Ra, Fa):
    """
    pcd: 
        numero de onda adimensional (eq. 22)
    Estimation of xCR
    CRref > 0, because always have both reflections: physical and spurious)
    """
    # Reference values for regression
    xCRIni = psi * (d + 1) / 2  # eq 18
    xCRInf = psi * d  # eq. 17
    xCRSup = psi * (2 - d)  # Eq 19
    CRIni = CRminQua(d, kCR, xCR=xCRIni)  # Eq. 16
    CRMef, xCRMef = CRminQua(d, kCR, typ="MEF", p=pCR, psi=psi, a=a, F_L=F_L)

    # Vertex or minimum positive root da equação quadrática
    xCRreg, CRreg = xCRVert(xCRIni, xCRSup, xCRMef, CRIni, CRmax, CRMef, xCRInf)
    # Ini primeiro ponto(a=b da eq quadratica), Sup máximo, Mef devido a essa reflexão espúria,Inf valor de referencia para o infinto 
    # Errors
    err1 = abs(np.sin(pCR / 2))  # Eq 26 e2
    err2 = abs(-1 + np.cos(pCR))  # Eq 26 e1

    # Eq 27 abaixo
    # Reference for calculation of interval of xCR
    if CRreg != 0:
        xCRref = xCRreg
        CRref = CRreg
    else:
        CRref = CRIni
    # Bounds: Minimmum errors
    lb1 = min(coeDampFun(CRref * (1 - err1), kCR, d, psi)[2] / d, xCRSup)
    lb2 = min(coeDampFun(CRref * (1 - err2), kCR, d, psi)[2] / d, xCRSup)
    if CRreg != 0:
        xCRmin = max(min(lb1, lb2), xCRInf)
    else:
        xCRmin = xCRreg
        xCRref = max(min(lb1, lb2), xCRInf)
    # Bounds: Maximum errors
    ub1 = min(coeDampFun(CRref * (1 + err1), kCR, d, psi)[2] / d, xCRSup)
    ub2 = min(coeDampFun(CRref * (1 + err2), kCR, d, psi)[2] / d, xCRSup)
    xCRmax = max(ub1, ub2)

    cad1 = "Range Values for 1D-xCR Factor: "
    cad2 = "RefMin:{:2.2f} - RefMax:{: 2.2f}"
    cad = cad1 + cad2
    # mp.my_print(cad.format(xCRmin, xCRmax))
    # Factors: 1/sqrt(1 + Ra), Ra/sqrt(1 + Ra) and their inverses
    fRamin = Fa / 4 * min(1 / (1 + Ra**2) ** 0.5, Ra / (1 + Ra**2) ** 0.5)
    fRamax = 4 / Fa * max((1 + Ra**2) ** 0.5, (1 + Ra**2) ** 0.5 / Ra)  # Eq. 29
    # Reference value
    xCRmin = max(xCRmin * fRamin, xCRInf)
    xCRmax = min(xCRmax * fRamax, xCRSup)

    return xCRref, xCRInf, xCRSup, xCRmin, xCRmax


# Reler 2.4 e ler ref 73
def funDamp(
    mesh,
    layer_type,
    length_z,
    length_x,
    pad_length,
    nexp,
    eta_array,
    etacr,
    neik_time_value,
    fref,
    neik_edge_length,
    neik_length_factor,
    neik_reference_velocity,
    psi=0.999,
    m=1,
):
    """
    Z: 1/eik
    lmin: do ponto critico

    Calculates the damping distribution within layer
    psi: Damping ratio. psi < 1: Underdamped regime
    m : Vibration mode
    """
    ref = np.empty_like(eta_array)  # Reference for damping
    condAmo = testPosDamp(mesh, length_z, length_x, pad_length, ref)

    a_star = 1 / (neik_time_value * fref)  # (apendece E eq e.10 e eq9 do corpo do texto)
    d = neik_edge_length / pad_length
    kCR = 4 * neik_length_factor / (a_star * m)  # eq 10 Aparece varias vezes no artigo 4Fl/ma_star
    psiaux = psi * (2 * d - d**2)

    arec = length_x + 2 * pad_length
    brec = 2 * minSemiAx(length_z, pad_length)  # Para realocar o centroide da hiperelipse

    Ra = arec / brec  # Aspect ratio of domain with layer

    Fa = 4

    # Estimation of xCR
    CRmax = abs(psiaux**2 / (psiaux**2 + kCR**2))

    # numero de onda adimensional a partir da eq 20
    pCR = 2 * pi * fref * neik_edge_length / neik_reference_velocity

    # Ler artigo da eq 20 pois elementos variam e alpha dif de 1
    # Olhar alpha como fração entre tamanho dos elementos
    xCR, xCRInf, xCRSup = estXCR(psi, d, kCR, pCR, CRmax, a_star, neik_length_factor, Ra, Fa)[0:3]

    # Minimum Reflection Coefficient
    CRmin = min(CRminQua(d, kCR, xCR=xCR), CRmax)
    aq, bq = coeDampFun(CRmin, kCR, d, psi)[0:2]
    # Damping distribution: ref = aq*x**2 + bq*x
    refamo = abs(aq * (ref[condAmo]) ** 2 + bq * (ref[condAmo]))
    del ref

    # Applying distribution damping
    etalim = psi * etacr
    eta_array[condAmo] = etacr * refamo
    del condAmo, refamo

    return etalim


def etafun(
    mesh_Fin,
    Lx,
    Ly,
    lref,
    pml,
    Z,
    fref,
    lmin,
    F_L,
    cref,
    TipLay="REC",
    nexp=np.nan,
):
    """
    Determines the distribution of the damping within the absorbing
    layer from the calcultions of the maximum damping value
    mesh_Fin: Final mesh with absorbing layer
    TipLay: Layer damping type (Rectangular: 'REC' or Hyperelliptical: 'HYP')
    Lx, Ly: Original domain dimensions
    lref: Reference length for the size of the absorbing layer
    pml: Length of absrobing layer
    Z: Inverse of minimum Eikonal
    fref: Reference frequency
    lmin: Minimal dimension of finite element
    F_L: Parameterized length of absorbing layer
    c_ref: Valocity at critical point wit minimum Eikonal
    nexp: Hyperellipse exponent for damping layer. nexp = NaN for rectangular layers
    """
    print("Determining Damping Distribution")
    eta = Function(V, name="eta (1/ms)")
    eta_array = eta.vector().get_local()
    # Factor for frequency correction
    factLen = calcFact2D(TipLay, Lx, Ly, pml, nexp)
    # Min Eikonal = 1 / Z. Tau = kw*eikmin
    etacr = calcCritDamp(1 / Z, Lx, Ly, lref, factLen)
    # Damping Function
    etalim = funDamp(
        mesh_Fin,
        TipLay,
        Lx,
        Ly,
        pml,
        nexp,
        eta_array,
        etacr,
        Z,
        fref,
        lmin,
        F_L,
        cref,
    )

    # Damping field
    eta.vector().set_local(eta_array)
    eta.vector().apply("insert")

    damp_file = File("/out/Damp.pvd")
    damp_file << eta

    return eta, etalim
