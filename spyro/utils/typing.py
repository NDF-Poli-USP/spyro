"""This module defines the enums and decorators used for typing in Spyro."""

from enum import Enum


def override(func):
    '''
    This decorator should be replaced by typing.override when Python
    version is updated to 3.12
    '''
    return func


class WaveType(Enum):
    """Enum for different types of wave equations that can be solved.

    NONE: No wave equation.
    ISOTROPIC_ACOUSTIC: Isotropic acoustic wave equation.
    ISOTROPIC_ELASTIC: Isotropic elastic wave equation for Isotropic media.
    ANISOTROPIC_VTI_ELASTIC: Anisotropic elastic wave equation for VTI media.
    ANISOTROPIC_TTI_ELASTIC: Anisotropic elastic wave equation for TTI media.
    """
    NONE = 0
    ISOTROPIC_ACOUSTIC = 1
    ISOTROPIC_ELASTIC = 2
    ANISOTROPIC_VTI_ELASTIC = 3
    ANISOTROPIC_TTI_ELASTIC = 4


class AdjointType(Enum):
    """Enum for the type of adjoint solver to use.

    NONE: No adjoint solver.
    AUTOMATED_ADJOINT: Use the automated adjoint solver via `firedrake.adjoint`.
    IMPLEMENTED_ADJOINT: Use the manually implemented adjoint solver.
    """

    NONE = 0
    AUTOMATED_ADJOINT = 1
    IMPLEMENTED_ADJOINT = 2


class RieszMapType(Enum):
    r"""Enum for the type of Riesz map used to recover the FE gradient.

    Let :math:`m_h \in V_h` be the control in a finite element space and let
    :math:`DJ(m_h) \in V_h'` be the reduced-functional derivative, which lives
    naturally in the dual space. The gradient :math:`g_h \in V_h` depends on
    the chosen inner product and is defined by the Riesz relation

    .. math::

        (g_h, v_h)_X = DJ(m_h)[v_h]\qquad \forall v_h \in V_h,

    where :math:`X` is one of the inner products below. In a FE basis
    :math:`\{\phi_i\}`, if :math:`b` is the assembled derivative vector and
    :math:`g` the coefficient vector of :math:`g_h`, then each option changes
    the linear system that turns the dual derivative into a primal gradient.

    L2:
        :math:`L^2` gradient. Solve :math:`Mg = b`, where
        :math:`M_{ij} = \int_\Omega \phi_i \phi_j \, dx` is the mass matrix.
        This accounts for element size and basis overlap, so the gradient is
        the Riesz representer in the FE function space.
    H1:
        :math:`H^1` gradient. Solve :math:`Ag = b`, where
        :math:`A_{ij} = \int_\Omega \phi_i \phi_j \, dx +
        \int_\Omega \nabla \phi_i \cdot \nabla \phi_j \, dx`.
        Relative to :math:`L^2`, this applies an elliptic smoothing that tends
        to damp high-frequency oscillations in the gradient.
    l2:
        Discrete :math:`\ell^2` coefficient-space gradient. Use :math:`g = b`
        directly, without inverting a FE mass or stiffness matrix. This is
        basis-dependent and acts on the vector of degrees of freedom rather
        than on the continuous FE inner product.

    In Spyro's automated-adjoint path, :class:`RieszMapType.L2` currently maps
    to ``compute_gradient()`` and :class:`RieszMapType.l2` to
    ``compute_derivative()``. :class:`RieszMapType.H1` captures the FE
    mathematical distinction above but is not yet implemented in that path.
    """

    L2 = 0
    H1 = 1
    l2 = 2


class ElasticMaterialParameter(Enum):
    """Supported isotropic elastic material parameter names."""

    DENSITY = "density"
    LAMBDA = "lambda"
    MU = "mu"
    P_WAVE_VELOCITY = "p_wave_velocity"
    S_WAVE_VELOCITY = "s_wave_velocity"


class ElasticMaterialParameterization(Enum):
    """Supported isotropic elastic inversion control parameterizations."""

    LAME = "lame"
    VELOCITY = "velocity"


class FunctionalType(Enum):
    """Enum for different types of functionals that can be computed.

    L2Norm : L2 norm functional, commonly used in classical full waveform
    inversion (FWI) as the measure to be minimized.
    """
    L2Norm = 0


class FunctionalEvaluationMode(Enum):
    """The mode in which to evaluate the functional.

    PER_TIMESTEP: Evaluate the functional at every time step during the time integration.
    AFTER_SOLVE: Evaluate the functional after the time integration is complete.
    """
    PER_TIMESTEP = "per_timestep"
    AFTER_SOLVE = "after_solve"


class LayerShapeType(Enum):
    """Enum for different types of absorbing layer shapes for ABCs.

    NOLAYER: No absorbing layer, i.e., no ABCs applied.
    RECTANGULAR: Rectangular absorbing layer`around the computational domain.
    HYPERSHAPE: Hypershape absorbing layer around the computational domain
    """
    NOLAYER = "no_layer"
    RECTANGULAR = "rectangular"
    HYPERSHAPE = "hypershape"


class LayerSizeRefFrequency(Enum):
    """Enum for different reference frequencies for sizing the hybrid absorbing layer.

    SOURCE: Size based on dominant source frequency.
    BOUNDARY: Size based on wave frequency at the critical boundary point (Eikonal min.)
    """
    SOURCE = "source"
    BOUNDARY = "boundary"


class HyperLayerDegreeType(Enum):
    """Enum for different types of hypershape degrees for HABCs.

    REAL: Hypershape degree can take real values >= 2.0 with one decimal place precision.
    INTEGER: Hypershape degree is restricted to integer values >= 2
    """
    REAL = "real"
    INTEGER = "integer"
