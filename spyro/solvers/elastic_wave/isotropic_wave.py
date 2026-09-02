import numpy as np

from firedrake import (assemble, Constant, curl, DirichletBC, div, Function,
                       project)
from pyadjoint import AdjFloat, Tape

from .elastic_wave import ElasticWave
from .forms import (isotropic_elastic_without_pml,
                    isotropic_elastic_with_pml)
from .functionals import mechanical_energy_form
from ...utils.physical_parameters import PhysicalParameters
from ...utils.typing import (AdjointType, ElasticMaterialParameter,
                             ElasticMaterialParameterization, AbsorbingBCsType,
                             RieszMapType)
from ...domains.space import create_function_space


PHYSICAL_PARAMETERIZATION = {
    ElasticMaterialParameterization.LAME: (
        ElasticMaterialParameter.DENSITY,
        ElasticMaterialParameter.LAMBDA,
        ElasticMaterialParameter.MU,
    ),
    ElasticMaterialParameterization.VELOCITY: (
        ElasticMaterialParameter.DENSITY,
        ElasticMaterialParameter.P_WAVE_VELOCITY,
        ElasticMaterialParameter.S_WAVE_VELOCITY,
    ),
}


def _format_physical_parameters(parameters):
    """Format material-parameter enum values for error messages.

    Parameters
    ----------
    parameters : iterable of ElasticMaterialParameter
        Material-parameter enum values to display.

    Returns
    -------
    str
        Human-readable set-like representation using public parameter names.

    Examples
    --------
    ``(ElasticMaterialParameter.DENSITY, ElasticMaterialParameter.MU)``
    becomes ``"{density, mu}"``.
    """
    return "{" + ", ".join(parameter.value for parameter in parameters) + "}"


class IsotropicWave(ElasticWave):
    '''Isotropic elastic wave propagator'''

    #: An isotropic elastic medium is described by density plus either the
    #: two Lame parameters or the two wave speeds; whichever pair is not
    #: declared is computed from the other.
    _physical_parameter_names = frozenset(ElasticMaterialParameter)

    def __init__(self, dictionary, comm=None):
        super().__init__(dictionary, comm=comm)
        self.rho = None   # Density
        self.lmbda = None  # First Lame parameter
        self.mu = None    # Second Lame parameter
        self.c_s = None   # Secondary wave velocity

        self.u_n = None   # Current displacement field
        self.u_nm1 = None  # Displacement field in previous iteration
        self.u_nm2 = None  # Displacement field at iteration n-2
        self.u_np1 = None  # Displacement field in next iteration

        # Volumetric sources (defined through UFL)
        self.body_forces = None

        # Boundary conditions
        self.bcs = []

        # Variables for logging the P-wave
        self.p_wave = None
        self.D_h = None
        self.field_logger.add_field("p-wave", "P-wave",
                                    lambda: self.update_p_wave())

        # Variables for logging the S-wave
        self.s_wave = None
        self.C_h = None
        self.field_logger.add_field("s-wave", "S-wave",
                                    lambda: self.update_s_wave())

        self.mechanical_energy = None
        self.field_logger.add_functional("mechanical_energy",
                                         lambda: assemble(self.mechanical_energy))

    def initialize_model_parameters_from_object(self, synthetic_data_dict: dict):
        """Initialize isotropic elastic material parameters from a dictionary.

        The dictionary must define exactly one supported material
        parameterization: either density with Lame parameters, or density with
        P- and S-wave velocities. The missing dependent parameters are
        computed from the provided set.

        This runs more than once: :meth:`enable_automated_adjoint` and
        :meth:`initialize_physical_parameters` both call it, and so does the
        forward solve itself for absorbing-boundary settings that rebuild
        the material properties. It reads the model only on the first of
        those. The parameters built then are the objects the assembled
        forms, the adjoint and any inversion refer to, so rebuilding them
        would replace those objects and reset an inversion's current iterate
        to the model's initial values. Replacing the model clears the
        parameters, which is what allows them to be built again.

        Parameters
        ----------
        synthetic_data_dict : dict
            Material parameter dictionary using the public Spyro model schema.
            Valid combinations are ``density``, ``lambda`` (or ``lame_first``),
            and ``mu`` (or ``lame_second``); or ``density``,
            ``p_wave_velocity``, and ``s_wave_velocity``. Values may be
            scalars, Firedrake ``Constant`` objects, Firedrake ``Function``
            objects, or UFL expressions. The dictionary is only read from.

        Returns
        -------
        None
            The method assigns ``rho``, ``lmbda``, ``mu``, ``c``, ``c_s``, and
            the active physical parameterization on ``self``.
        """
        if self._physical_parameters:
            return

        def declared(parameter, *aliases):
            """Return the model value of ``parameter``, or ``None``."""
            for key in (parameter.value, *aliases):
                if key in synthetic_data_dict:
                    value = synthetic_data_dict[key]
                    return Constant(value) if np.isscalar(value) else value
            return None

        self.rho = declared(ElasticMaterialParameter.DENSITY)
        self.lmbda = declared(ElasticMaterialParameter.LAMBDA, "lame_first")
        self.mu = declared(ElasticMaterialParameter.MU, "lame_second")
        self.c = declared(ElasticMaterialParameter.P_WAVE_VELOCITY)
        self.c_s = declared(ElasticMaterialParameter.S_WAVE_VELOCITY)

        # Exactly one set must be declared, and it names the parameters
        # that carry the material data. ``is not None`` rather than
        # truthiness:
        # every UFL object is unconditionally true, so ``bool`` would only
        # ever be testing whether the key was present.
        lame = (
            self.rho is not None
            and self.lmbda is not None
            and self.mu is not None
            and self.c is None
            and self.c_s is None
        )
        velocity = (
            self.rho is not None
            and self.c is not None
            and self.c_s is not None
            and self.lmbda is None
            and self.mu is None
        )

        if lame:
            declared_parameterization = ElasticMaterialParameterization.LAME
        elif velocity:
            declared_parameterization = (
                ElasticMaterialParameterization.VELOCITY
            )
        else:
            options = " or (exclusive) ".join(
                _format_physical_parameters(parameters)
                for parameters in PHYSICAL_PARAMETERIZATION.values()
            )
            raise ValueError(
                "Inconsistent selection of isotropic elastic wave "
                "parameters:\n"
                f"    Density        : {self.rho is not None}\n"
                f"    Lame first     : {self.lmbda is not None}\n"
                f"    Lame second    : {self.mu is not None}\n"
                f"    P-wave velocity: {self.c is not None}\n"
                f"    S-wave velocity: {self.c_s is not None}\n"
                f"The valid options are {options}",
            )
        self.set_physical_parameterization(declared_parameterization)

    def set_physical_parameterization(
        self, parameterization: ElasticMaterialParameterization,
    ) -> None:
        """Set which elastic parameters carry the material data.

        All five are read whatever this is set to: the variational form is
        written in density and the Lame parameters, while the absorbing
        boundary conditions and the stable timestep estimate are written in
        the two wave speeds. The chosen three become scalar ``Function``
        objects and the other two become UFL expressions of them, recomputed
        wherever they appear, so updating one of the chosen parameters
        carries through to the computed ones and to the assembled forms.

        This is a change of variables on the solver, not an edit of the
        model: the input dictionary is left as the user wrote it, and the
        set chosen here survives because initialization does not read the
        model a second time.

        Parameters
        ----------
        parameterization : ElasticMaterialParameterization
            Set of elastic parameters to carry the data.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the mesh has not been created, or the set of parameters is
            not one this solver supports.
        """
        space = None if self.mesh is None else create_function_space(
            self.mesh, self.method, self.degree, dim=1,
        )

        def as_function(value, parameter):
            """Return ``value`` as the independent field of ``parameter``.

            Before a mesh exists there is no space to build a ``Function``
            in, so the value is left as the scalar or ``Constant`` it came
            in as, and this set still carries the data.
            """
            if space is None or isinstance(value, Function):
                return value
            return Function(space, name=parameter.value).interpolate(value)

        if parameterization is ElasticMaterialParameterization.LAME:
            self.rho = as_function(self.rho, ElasticMaterialParameter.DENSITY)
            self.lmbda = as_function(self.lmbda, ElasticMaterialParameter.LAMBDA)
            self.mu = as_function(self.mu, ElasticMaterialParameter.MU)
            self.c = ((self.lmbda + 2*self.mu)/self.rho)**0.5
            self.c_s = (self.mu/self.rho)**0.5
        elif parameterization is ElasticMaterialParameterization.VELOCITY:
            self.rho = as_function(self.rho, ElasticMaterialParameter.DENSITY)
            self.c = as_function(
                self.c, ElasticMaterialParameter.P_WAVE_VELOCITY,
            )
            self.c_s = as_function(
                self.c_s, ElasticMaterialParameter.S_WAVE_VELOCITY,
            )
            self.mu = self.rho*self.c_s**2
            self.lmbda = self.rho*self.c**2 - 2*self.mu
        else:
            raise ValueError(
                "Unsupported elastic material parameterization: "
                f"{parameterization}.",
            )

        add = self._physical_parameters.add
        add(ElasticMaterialParameter.DENSITY, self.rho)
        add(ElasticMaterialParameter.LAMBDA, self.lmbda)
        add(ElasticMaterialParameter.MU, self.mu)
        add(ElasticMaterialParameter.P_WAVE_VELOCITY, self.c)
        add(ElasticMaterialParameter.S_WAVE_VELOCITY, self.c_s)

    def gradient_solve(
        self,
        misfit=None,
        forward_solution=None,
        adjoint_type=AdjointType.AUTOMATED_ADJOINT,
        riesz_map=RieszMapType.L2,
    ) -> PhysicalParameters:
        """Compute automated-adjoint elastic material derivatives.

        Only the automated adjoint is available for the elastic wave so far.
        The implemented adjoint -- the backward integration written out by
        hand, which the acoustic solver already offers -- is intended to
        follow, and ``misfit`` and ``forward_solution`` are the two inputs it
        needs. They are part of the signature so that callers written against
        :meth:`~spyro.solvers.acoustic_wave.AcousticWave.gradient_solve` keep
        working once it lands, and are unused until then.

        Parameters
        ----------
        misfit : array_like, optional
            Difference between observed and simulated receiver data. The
            implemented adjoint drives the backward equation with it. The
            automated adjoint does not need it: it differentiates the
            functional recorded during the forward solve, which already
            accumulated the misfit.
        forward_solution : firedrake.Function, optional
            Forward wavefield. The implemented adjoint integrates the adjoint
            equation backwards against it, so passing it saves a forward
            solve. The automated adjoint recovers the wavefield on its own.
        adjoint_type : AdjointType, optional
            Must be :attr:`AdjointType.AUTOMATED_ADJOINT` until the
            implemented adjoint is available for elastic waves.
        riesz_map : RieszMapType, optional
            ``L2`` returns primal gradients and ``l2`` raw derivatives.

        Returns
        -------
        PhysicalParameters
            Derivatives keyed by the selected elastic parameter enums.

        Raises
        ------
        NotImplementedError
            If a hand-implemented adjoint or unsupported Riesz map is requested.
        ValueError
            If no valid annotated functional is available.
        """
        if adjoint_type is not AdjointType.AUTOMATED_ADJOINT:
            raise NotImplementedError(
                "Elastic gradients only support the automated adjoint.",
            )
        if not isinstance(self.functional_value, AdjFloat):
            raise ValueError(
                "Functional value must be an AdjFloat for automated adjoint "
                "gradient computation.",
            )
        if self.automated_adjoint is None:
            raise ValueError(
                "Enable the automated adjoint before the elastic forward solve.",
            )
        if (
            self.automated_adjoint.reduced_functional is None
            and isinstance(self.automated_adjoint._tape, Tape)
        ):
            self.automated_adjoint.create_reduced_functional(
                self.functional_value,
            )

        if riesz_map is RieszMapType.L2:
            derivatives = self.automated_adjoint.compute_gradient()
        elif riesz_map is RieszMapType.l2:
            derivatives = self.automated_adjoint.compute_derivative()
        else:
            raise NotImplementedError(
                f"Riesz map {riesz_map} not implemented for automated adjoint.",
            )
        return self.automated_adjoint.label_derivatives(derivatives)

    def initialize_model_parameters_from_file(self, synthetic_data_dict):
        raise NotImplementedError

    def _create_function_space(self):
        return create_function_space(self.mesh, self.method, self.degree,
                                     dim=self.dimension)

    def _set_vstate(self, vstate):
        self.u_n.assign(vstate)

    def _get_vstate(self):
        return self.u_n

    def _set_prev_vstate(self, vstate):
        if self.u_nm2 is not None:
            self.u_nm2.assign(self.u_nm1)
        self.u_nm1.assign(vstate)

    def _get_prev_vstate(self):
        return self.u_nm1

    def _set_next_vstate(self, vstate):
        self.u_np1.assign(vstate)

    def _get_next_vstate(self):
        return self.u_np1

    def get_forward_solution_receivers(self):
        if self.abc_type == AbsorbingBCsType.PML:
            raise NotImplementedError
        else:
            data_with_halos = self.u_n.dat.data_ro_with_halos[:]
        return self.receivers.interpolate(data_with_halos)

    def get_function(self):
        return self.u_n

    def get_function_name(self):
        return "Displacement"

    def matrix_building(self):
        self.current_time = 0.0

        self.u_n = Function(self.function_space,
                            name=self.get_function_name())
        self.u_nm1 = Function(self.function_space,
                              name=self.get_function_name())
        self.u_np1 = Function(self.function_space,
                              name=self.get_function_name())

        abc_dict = self.input_dictionary.get("absorving_boundary_conditions", None)
        if abc_dict is not None:
            abc_active = abc_dict.get("status", False)
            if abc_active:
                dt_scheme = abc_dict.get("nrbc", {}).get("dt_scheme", None)
                if dt_scheme == "backward_2nd":
                    self.u_nm2 = Function(self.function_space,
                                          name=self.get_function_name())

        self.mechanical_energy = mechanical_energy_form(self)

        self.parse_initial_conditions()
        self.parse_boundary_conditions()
        self.parse_volumetric_forces()

        if self.abc_type in [AbsorbingBCsType.NRBC, AbsorbingBCsType.NOABCS]:
            isotropic_elastic_without_pml(self)
        elif self.abc_type == AbsorbingBCsType.PML:
            isotropic_elastic_with_pml(self)

    def rhs_no_pml(self):
        if self.abc_type == AbsorbingBCsType.PML:
            raise NotImplementedError
        else:
            return self.B

    def rhs_no_pml_source(self):
        if self.abc_type == AbsorbingBCsType.PML:
            raise NotImplementedError
        else:
            return self.source_function

    def parse_initial_conditions(self):
        time_dict = self.input_dictionary["time_axis"]
        initial_condition = time_dict.get("initial_condition", None)
        if initial_condition is not None:
            x_vec = self.get_spatial_coordinates()
            self.u_n.interpolate(initial_condition(x_vec, 0 - self.dt))
            self.u_nm1.interpolate(initial_condition(x_vec, 0 - 2*self.dt))

    def parse_boundary_conditions(self):
        bc_list = self.input_dictionary.get("boundary_conditions", [])
        for tag, idbc, value in bc_list:
            if tag == "u":
                subspace = self.function_space
            elif tag == "uz":
                subspace = self.function_space.sub(0)
            elif tag == "ux":
                subspace = self.function_space.sub(1)
            elif tag == "uy":
                subspace = self.function_space.sub(2)
            else:
                raise Exception(
                    f"Unsupported boundary condition with tag: {tag}")
            self.bcs.append(DirichletBC(subspace, value, idbc))

    def parse_volumetric_forces(self):
        acquisition_dict = self.input_dictionary["acquisition"]
        body_forces_data = acquisition_dict.get("body_forces", None)
        if body_forces_data is not None:
            x_vec = self.get_spatial_coordinates()
            self.body_forces = body_forces_data(x_vec, self.time)

    def update_p_wave(self):
        if self.p_wave is None:
            self.D_h = create_function_space(self.mesh, "DG0", 0)
            self.p_wave = Function(self.D_h)

        self.p_wave.assign(project(div(self.get_function()), self.D_h))

        return self.p_wave

    def update_s_wave(self):
        if self.s_wave is None:
            if self.dimension == 2:
                self.C_h = create_function_space(self.mesh, "DG0", 0)
            else:
                self.C_h = create_function_space(self.mesh, "DG0", 0,
                                                 dim=self.dimension)
            self.s_wave = Function(self.C_h)

        self.s_wave.assign(project(curl(self.get_function()), self.C_h))

        return self.s_wave
