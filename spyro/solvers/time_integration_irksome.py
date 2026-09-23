"""Runge-Kutta-Nystrom time integration of the wave equations with Irksome.

`Irksome <https://www.firedrakeproject.org/Irksome/>`_ discretizes in time a
variational form written with symbolic time derivatives, ``Dt(u)`` and
``Dt(u, 2)``, from a Butcher tableau. The wave equations spyro solves are
second order in time, so this module uses Irksome's Nystrom steppers: the
state is the field ``u`` and its time derivative ``u_t``, and each step
solves for the stage accelerations of a Runge-Kutta-Nystrom method.

Three families of tableaux are supported, each with the stepper that suits
it:

* explicit tableaux (``"rk4"``, ``"classic_nystrom4"``) with the stage
  coupled :class:`irksome.StageDerivativeNystromTimeStepper` (through
  :class:`StageCoupledNystromStepper`). The stage system is block lower
  triangular, so a multiplicative field split with the mass-matrix solver
  of the spatial method solves it exactly in one sweep, which on the
  mass-lumped elements is a diagonal scaling per stage. A step costs a few
  central-difference steps, the stages being assembled together;
* diagonally implicit tableaux (``"backward_euler"``, ``"alexander"``,
  ``"qin_zhang"``) with :class:`irksome.DIRKNystromTimeStepper`, which
  solves one shifted mass-stiffness system per stage, matrix-free;
* fully implicit tableaux (``"gauss_legendre"``, ``"radau_iia"``,
  ``"lobatto_iiia"``, ``"lobatto_iiic"``) with the stage coupled stepper
  and a direct solve of the coupled stages, factorized once.

The defaults are chosen so that the automated adjoint, which solves the
transposed stage systems with the same parameters, is exact as well; see
:func:`default_solver_parameters`.

The wave solver contributes its equation through
:meth:`~spyro.solvers.wave.Wave.weak_form`; the point sources are added
here as the product of the wavelet, written as a UFL expression of the
time, with the Riesz representer of the point-source cofunction, so that
the stages evaluate the source at their own times. Irksome must be imported
before any UFL form is processed, which is why :mod:`spyro` imports this
module when it is imported; when the import fails the reason is kept and
raised as soon as the scheme is requested.

The scheme is selected with ``time_axis["time_integration_scheme"] =
"irksome"`` and configured through ``time_axis["irksome"]``, see
:class:`~spyro.io.time_io.IrksomeOptions`.
"""

import numpy as np
import firedrake as fire
from firedrake import dx, inner, TestFunction, TrialFunction
from pyadjoint import stop_annotating

from .time_integration_central_difference import _propagate_forward
from ..io.time_io import IrksomeOptions
from ..utils.typing import AdjointType

#: What this module needs from Irksome, beyond importing: the Nystrom steppers
#: for second-order-in-time equations and the tableau module they live behind.
#: An Irksome that predates them imports fine and then fails deep inside a
#: solve, so it is rejected here instead, with the feature reporting itself as
#: unavailable exactly as it does when Irksome is absent.
REQUIRED_ATTRIBUTES = (
    "StageDerivativeNystromTimeStepper",
    "DIRKNystromTimeStepper",
    "ClassicNystrom4Tableau",
)
REQUIRED_MODULES = ("irksome.tableaux.ButcherTableaux", "irksome.nystrom_stepper")


def _check_irksome_api(module):
    """Return why ``module`` cannot drive the Nystrom steppers, or ``None``.

    Parameters
    ----------
    module : module
        The imported :mod:`irksome`.

    Returns
    -------
    ImportError or None
        The reason the installed Irksome is too old, or ``None`` when it
        offers everything this module uses.
    """
    import importlib

    missing = [name for name in REQUIRED_ATTRIBUTES if not hasattr(module, name)]
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if not missing:
        return None
    return ImportError(
        "The installed Irksome predates the Runge-Kutta-Nystrom methods this "
        f"module is built on (missing: {', '.join(missing)}). They arrived "
        "with the 2026.0.0 release; the Irksome bundled with Firedrake "
        "2025.4.1 and earlier does not have them. Note that the bilinear "
        "form spyro hands the stepper also needs an Irksome whose time "
        "derivative accepts an Argument."
    )


try:
    import irksome
    from irksome import Dt
except ImportError as error:
    irksome = None
    Dt = None
    IRKSOME_IMPORT_ERROR = error
except Exception as error:  # noqa: BLE001
    # ``irksome.ufl.deriv.IrksomeImportOrderException``: a UFL algorithm ran
    # before this import, so its node type could not be registered.
    irksome = None
    Dt = None
    IRKSOME_IMPORT_ERROR = error
else:
    IRKSOME_IMPORT_ERROR = _check_irksome_api(irksome)
    if IRKSOME_IMPORT_ERROR is not None:
        irksome = None
        Dt = None


#: Stepper families, decided by the structure of the Butcher matrix.
EXPLICIT = "explicit"
DIRK = "diagonally_implicit"
FULLY_IMPLICIT = "fully_implicit"


if irksome is not None:

    class StageCoupledNystromStepper(irksome.StageDerivativeNystromTimeStepper):
        """Stage-coupled Nystrom stepper whose stage form reads copies of the state.

        Irksome builds the stage form on the state fields themselves and
        updates them in place through ``subfunctions`` views. Under the
        automated adjoint that is fragile in two ways. For a vector-valued
        field Firedrake's ``SubfunctionBlock`` addresses a view with
        ``sub``, which on a non-mixed vector space picks one component
        rather than the field. And with a checkpoint schedule that
        recomputes the forward, the gradient came out different from the
        one of the plain tape (by ``4e-4`` relative, through the
        velocity-dependent boundary term) whenever the stage solve read a
        field that the same step overwrites afterwards; Irksome's DIRK
        stepper, whose stage form reads fresh stage arguments instead, was
        unaffected. So here the stage form reads copies of the field and
        its rate taken at the start of the step, and the update writes the
        fields themselves from the stage vector, read symbolically and
        interpolated, which is exact within one space.

        Parameters
        ----------
        F : ufl.Form
            Semidiscrete form in the trial function, with ``Dt``.
        tableau : object
            Irksome Butcher or Nystrom tableau.
        t, dt : firedrake.Function
            Time and time step on the Real space.
        u0, ut0 : firedrake.Function
            The field and its rate, advanced in place by :meth:`advance`.
        **kwargs
            Passed on to :class:`irksome.StageDerivativeNystromTimeStepper`.
        """

        def __init__(self, F, tableau, t, dt, u0, ut0, **kwargs):
            V = u0.function_space()
            self.state = u0
            self.rate = ut0
            stage_state = fire.Function(V, name=f"{u0.name()} at step start")
            stage_rate = fire.Function(V, name=f"{ut0.name()} at step start")
            super().__init__(F, tableau, t, dt, stage_state, stage_rate, **kwargs)

        def advance(self):
            """Advance the field and its rate by one time step.

            Returns
            -------
            None
                ``state`` and ``rate`` are updated in place.
            """
            self.u0.assign(self.state)
            self.ut0.assign(self.rate)
            super().advance()

        def _update(self):
            """Update the field and its rate from the stage accelerations.

            Returns
            -------
            None
                ``state`` and ``rate`` are updated in place.
            """
            if self.num_fields != 1:
                raise NotImplementedError(
                    "The stage-coupled stepper of spyro is written for a "
                    "single field."
                )
            b = self.updateb
            bbar = self.updatebbar
            ns = self.tableau.num_stages
            dt = self.dt
            V = self.state.function_space()
            ks = fire.split(self.stages) if ns > 1 else (self.stages,)
            self.state.assign(fire.assemble(fire.interpolate(
                self.u0 + self.ut0 * dt
                + sum(ks[s] * (bbar[s] * dt**2) for s in range(ns)), V,
            )))
            self.rate.assign(fire.assemble(fire.interpolate(
                self.ut0 + sum(ks[s] * (b[s] * dt) for s in range(ns)), V,
            )))


def require_irksome():
    """Raise a helpful error when Irksome is not usable.

    Returns
    -------
    None

    Raises
    ------
    ImportError
        If Irksome is not installed, or was imported too late to register
        its UFL node type.
    """
    if irksome is not None:
        return
    if isinstance(IRKSOME_IMPORT_ERROR, ImportError):
        raise ImportError(
            "The 'irksome' time integration scheme needs Irksome 2026.0.0 or "
            "newer, which is either not installed or too old here "
            f"({IRKSOME_IMPORT_ERROR}). Install it with 'pip install IRKsome' "
            "(or from https://github.com/firedrakeproject/Irksome) into the "
            "Firedrake environment."
        ) from IRKSOME_IMPORT_ERROR
    raise ImportError(
        "Irksome could not be imported when spyro was: it registers a UFL "
        "node type, which has to happen before any form is processed. "
        "Import spyro before creating meshes, function spaces or forms."
    ) from IRKSOME_IMPORT_ERROR


def classic_rk4_tableau():
    """Return the classical fourth-order Runge-Kutta Butcher tableau.

    Irksome converts it to a Nystrom tableau, so the method keeps its
    fourth order on equations with velocity-dependent terms, such as the
    absorbing boundary conditions, unlike :class:`irksome.ClassicNystrom4Tableau`,
    which is built for ``u'' = f(t, u)``.

    Returns
    -------
    irksome.tableaux.ButcherTableaux.ButcherTableau
        The explicit four-stage tableau.
    """
    require_irksome()
    from irksome.tableaux.ButcherTableaux import ButcherTableau

    A = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    b = np.array([1.0, 2.0, 2.0, 1.0]) / 6.0
    c = np.array([0.0, 0.5, 0.5, 1.0])
    return ButcherTableau(A, b, None, c, 4, None, None)


def tableau_catalogue():
    """Return the tableaux selectable by name.

    Returns
    -------
    dict
        Maps each name to ``(factory, minimum_stages)``. ``factory`` takes
        the number of stages when ``minimum_stages`` is an integer and no
        argument when it is ``None``, in which case the method has a fixed
        number of stages.
    """
    require_irksome()
    return {
        "rk4": (classic_rk4_tableau, None),
        "classic_nystrom4": (irksome.ClassicNystrom4Tableau, None),
        "gauss_legendre": (irksome.GaussLegendre, 1),
        "radau_iia": (irksome.RadauIIA, 1),
        "lobatto_iiia": (irksome.LobattoIIIA, 2),
        "lobatto_iiic": (irksome.LobattoIIIC, 2),
        "backward_euler": (irksome.BackwardEuler, None),
        "alexander": (irksome.Alexander, None),
        "qin_zhang": (irksome.QinZhang, None),
    }


def make_tableau(options: IrksomeOptions):
    """Build the Runge-Kutta tableau the options describe.

    Parameters
    ----------
    options : IrksomeOptions
        The ``time_axis["irksome"]`` options.

    Returns
    -------
    object
        An Irksome ``ButcherTableau`` or ``NystromTableau``.

    Raises
    ------
    ValueError
        If the name is not in :func:`tableau_catalogue`, if ``stages`` is
        given for a method with a fixed number of stages or is below the
        minimum of a collocation family, or if a tableau object comes with
        ``stages``.
    TypeError
        If ``tableau`` is neither a name nor a tableau object.
    """
    require_irksome()
    tableau = options.tableau
    if isinstance(tableau, str):
        catalogue = tableau_catalogue()
        try:
            factory, minimum_stages = catalogue[tableau]
        except KeyError:
            raise ValueError(
                f"Unknown Irksome tableau {tableau!r}; choose one of "
                f"{sorted(catalogue)} or pass an Irksome tableau object."
            ) from None
        if minimum_stages is None:
            if options.stages is not None:
                raise ValueError(
                    f"The {tableau!r} tableau has a fixed number of stages; "
                    "do not set time_axis['irksome']['stages'] for it."
                )
            return factory()
        stages = minimum_stages if options.stages is None else options.stages
        if stages < minimum_stages:
            raise ValueError(
                f"The {tableau!r} family needs at least {minimum_stages} "
                f"stages, got {stages}."
            )
        return factory(stages)

    from irksome.nystrom_stepper import NystromTableau
    from irksome.tableaux.ButcherTableaux import ButcherTableau

    if not isinstance(tableau, (ButcherTableau, NystromTableau)):
        raise TypeError(
            "time_axis['irksome']['tableau'] must be a name or an Irksome "
            f"ButcherTableau/NystromTableau, got {type(tableau).__name__}."
        )
    if options.stages is not None:
        raise ValueError(
            "time_axis['irksome']['stages'] cannot be combined with a "
            "tableau object, whose number of stages is fixed."
        )
    return tableau


def stepper_family(tableau) -> str:
    """Classify a tableau by the structure of its Butcher matrix.

    Parameters
    ----------
    tableau : object
        An Irksome ``ButcherTableau`` or ``NystromTableau``.

    Returns
    -------
    str
        :data:`EXPLICIT`, :data:`DIRK` or :data:`FULLY_IMPLICIT`.
    """
    if tableau.is_explicit:
        return EXPLICIT
    if tableau.is_diagonally_implicit:
        return DIRK
    return FULLY_IMPLICIT


def mass_solver_parameters(wave) -> dict:
    """Return the PETSc options that invert the mass matrix of ``wave``.

    Parameters
    ----------
    wave : Wave
        The wave solver. Its ``solver_parameters`` are the mass-matrix
        options of its spatial method: a diagonal scaling on the mass-lumped
        elements. Methods without a default get a conjugate gradient solve.

    Returns
    -------
    dict
        Options for one mass-matrix solve, without a ``mat_type``.
    """
    if wave.solver_parameters is not None:
        return dict(wave.solver_parameters)
    return {"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-12}


def default_solver_parameters(wave, family: str) -> dict:
    """Return the PETSc options of the stage system for a stepper family.

    Parameters
    ----------
    wave : Wave
        The wave solver, see :func:`mass_solver_parameters`.
    family : str
        One of :data:`EXPLICIT`, :data:`DIRK` and :data:`FULLY_IMPLICIT`.

    Returns
    -------
    dict
        Solver parameters, see :class:`~spyro.io.time_io.IrksomeOptions`.

    Notes
    -----
    Under the automated adjoint the same parameters solve the transposed
    stage system, so the defaults are exact, or converge to a tight
    tolerance, for the transpose too. User-supplied parameters have to
    keep that property for the gradient to be right.
    """
    if family == EXPLICIT:
        # The stages couple only downwards, so a multiplicative field split
        # is a block forward substitution: exact in a single sweep, with one
        # mass solve per stage. The adjoint solves the transposed system,
        # which couples upwards and needs the backward sweep as well; the
        # symmetric variant is exact for both, at the price of a second
        # sweep in the forward solve.
        if wave.adjoint_type is AdjointType.AUTOMATED_ADJOINT:
            fieldsplit_type = "symmetric_multiplicative"
        else:
            fieldsplit_type = "multiplicative"
        parameters = {
            "mat_type": "matfree",
            "ksp_type": "preonly",
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": fieldsplit_type,
        }
        for key, value in mass_solver_parameters(wave).items():
            parameters[f"fieldsplit_{key}"] = value
        return parameters
    if family == DIRK:
        # Each stage solves ``M + a dt^2 K``, symmetric positive definite and
        # close to the mass matrix for the time steps a wave solve takes.
        return {
            "mat_type": "matfree",
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "ksp_rtol": 1e-10,
        }
    return {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }


class IrksomeIntegrator:
    """Irksome time stepper of one wave solver.

    Built by the solver's ``matrix_building`` in place of the
    central-difference operators. It owns the time constants, the state
    fields and the point-source representer, and wraps the Irksome stepper
    that advances them.

    Parameters
    ----------
    wave : Wave
        The wave solver. It must have its function space, quadrature rules,
        material parameters, boundary conditions and sources ready, since
        the variational form is built here.

    Attributes
    ----------
    wave : Wave
        The wave solver being integrated.
    options : IrksomeOptions
        The options the stepper was built from.
    tableau : object
        The Irksome tableau.
    family : str
        The stepper family of the tableau, see :func:`stepper_family`.
    t : firedrake.Function
        The time, a function on the Real space of the mesh, shared with the
        solver as ``wave.time`` so that its UFL source expressions are
        evaluated at the stage times. A coefficient rather than a
        ``Constant`` because only coefficients are recorded as dependencies
        of the stage solves by the automated adjoint, and the source has
        to be replayed at the right time.
    dt : firedrake.Function
        The time step, on the same Real space.
    u : firedrake.Function
        The field, also ``wave.u_n``.
    u_t : firedrake.Function
        Its time derivative, also ``wave.u_t``.
    point_source : firedrake.Function or None
        Riesz representer of the point sources of the current shot, with
        respect to the mass matrix of the spatial method; ``None`` when the
        solver has no point sources.
    stepper : object
        The Irksome stepper.
    """

    def __init__(self, wave):
        require_irksome()
        self.wave = wave
        self.options = wave.irksome_options
        self.tableau = make_tableau(self.options)
        self.family = stepper_family(self.tableau)

        V = wave.function_space
        name = wave.get_function_name()
        self.u = fire.Function(V, name=name)
        self.u_t = fire.Function(V, name=f"{name} rate")
        self.t = wave.time
        self.t.assign(0.0)
        self.dt = fire.Function(self.t.function_space(), name="dt")
        self.dt.assign(wave.dt)

        u = TrialFunction(V)
        v = TestFunction(V)
        F = wave.weak_form(u, Dt(u), Dt(u, 2), v)
        self.point_source = None
        if wave.sources is not None:
            self.point_source = fire.Function(V, name="point source")
            wavelet = wave.sources.wavelet_expression(self.t)
            F = F - wavelet * inner(self.point_source, v) * dx(
                **wave.quadrature_rule
            )

        bc_type = self.options.bc_type
        if bc_type is None:
            bc_type = "dDAE" if self.family == EXPLICIT else "DAE"
        solver_parameters = self.options.solver_parameters
        if solver_parameters is None:
            solver_parameters = default_solver_parameters(wave, self.family)

        if self.family == DIRK:
            self.stepper = irksome.DIRKNystromTimeStepper(
                F, self.tableau, self.t, self.dt, self.u, self.u_t,
                bcs=wave.bcs, bc_type=bc_type,
                solver_parameters=solver_parameters,
            )
        else:
            # The form is linear with coefficients fixed for the whole
            # solve, so the stage operator is assembled, and for the direct
            # solver factorized, only once.
            self.stepper = StageCoupledNystromStepper(
                F, self.tableau, self.t, self.dt, self.u, self.u_t,
                bcs=wave.bcs, bc_type=bc_type,
                solver_parameters=solver_parameters,
                constant_jacobian=True,
            )

    def update_point_source(self) -> None:
        """Rebuild the point-source representer for the active sources.

        The point sources enter the variational form as the field whose
        mass-matrix product is their cofunction, so that the term is an
        integral like the rest of the form and the stage substitutions
        apply to it. With the mass-lumped elements the representer is the
        cofunction scaled by the inverse diagonal, so the product is
        exact. The active sources are ``wave.sources.current_sources``.

        Returns
        -------
        None
            ``point_source`` is updated in place.
        """
        if self.point_source is None:
            return
        wave = self.wave
        V = wave.function_space
        u = TrialFunction(V)
        v = TestFunction(V)
        mass = inner(u, v) * dx(**wave.quadrature_rule)
        solver_parameters = {"mat_type": "matfree", **mass_solver_parameters(wave)}
        # The representer only depends on where the sources are, never on
        # a control, so there is nothing for the adjoint to record.
        with stop_annotating():
            cofunction = wave.sources.point_source_cofunction()
            fire.solve(
                mass == cofunction, self.point_source,
                solver_parameters=solver_parameters,
            )

    def reset(self) -> None:
        """Zero the state before propagating another shot.

        Returns
        -------
        None
            ``u`` and ``u_t`` are zeroed in place.
        """
        self.u.assign(0.0)
        self.u_t.assign(0.0)

    def advance(self) -> None:
        """Advance the state by one time step and move the time forward.

        Returns
        -------
        None
            ``u``, ``u_t`` and ``t`` are updated in place.
        """
        self.stepper.advance()
        self.t.assign(float(self.t) + float(self.dt))


def _propagate_forward_irksome(wave, source_ids):
    """Advance the forward solve with the Irksome stepper of ``wave``.

    This is an internal helper used by :meth:`wave.wave_propagator`. It updates
    the solver state in place.

    Parameters
    ----------
    wave: Wave
        The wave solver object containing all necessary information to perform
        the forward solve.
    source_ids: list of int
        List of source IDs to simulate.

    Returns
    -------
    None
        The solver state, receiver data and functional are updated in place.
    """
    integrator = wave.irksome_integrator
    if integrator is None:
        raise ValueError(
            "The Irksome stepper has not been built; call matrix_building() "
            "or forward_solve()."
        )
    if wave.sources is not None:
        wave.sources.current_sources = source_ids
        integrator.update_point_source()
    integrator.dt.assign(wave.dt)
    integrator.t.assign(wave.current_time)

    def advance(step, t, nt):
        """Take one Runge-Kutta-Nystrom step from time level ``step``."""
        integrator.advance()
        # The time step has to close right after the stage solve and the
        # state update, before anything else is recorded. The final step is
        # closed by pyadjoint itself when taping ends, hence ``nt - 1``.
        if wave.adjoint_type == AdjointType.AUTOMATED_ADJOINT and step < nt - 1:
            wave.automated_adjoint.end_timestep()
        return float(integrator.t)

    _propagate_forward(wave, source_ids, advance)
