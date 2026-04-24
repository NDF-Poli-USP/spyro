from enum import Enum


def override(func):
    '''
    This decorator should be replaced by typing.override when Python
    version is updated to 3.12
    '''
    return func


class WaveType(Enum):
    NONE = 0
    ISOTROPIC_ACOUSTIC = 1
    ISOTROPIC_ELASTIC = 2


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
