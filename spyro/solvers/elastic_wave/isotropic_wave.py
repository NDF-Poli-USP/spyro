import numpy as np

from firedrake import (assemble, Cofunction, Constant, curl, DirichletBC, div,
                       Function, project)

from .elastic_wave import ElasticWave
from .forms import (isotropic_elastic_without_pml,
                    isotropic_elastic_with_pml)
from .functionals import mechanical_energy_form
from .material import (ATTRIBUTE_BY_PARAMETER, DERIVATIVES_BY_PARAMETERIZATION,
                       KEYS_BY_PARAMETER, PARAMETERS_BY_PARAMETERIZATION,
                       ElasticControlSet, resolve_parameterization)
from ...utils.typing import (AdjointType, ElasticMaterialParameter,
                             ElasticMaterialParameterization, AbsorbingBCsType,
                             RieszMapType, override)
from ...domains.space import create_function_space


class IsotropicWave(ElasticWave):
    '''Isotropic elastic wave propagator'''

    def __init__(self, dictionary, comm=None):
        super().__init__(dictionary, comm=comm)
        self.rho = None   # Density
        self.lmbda = None  # First Lame parameter
        self.mu = None    # Second Lame parameter
        self.c_s = None   # Secondary wave velocity
        self._control_parameterization = None
        self._material_parameter_function_space = None

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

    @override
    def initialize_model_parameters_from_object(self, synthetic_data_dict: dict):
        """Initialize isotropic elastic material parameters from a dictionary.

        The declaration must define exactly one complete family. The optional
        ``parameterization`` entry then re-expresses the equation in the other
        family, which is what makes its parameters differentiable; the subset
        actually inverted for is chosen later, with
        ``enable_automated_adjoint(controls=...)``.

        Parameters
        ----------
        synthetic_data_dict : dict
            Material parameter dictionary using the public Spyro model schema.
            Valid material combinations are ``density``, ``lambda`` (or
            ``lame_first``), and ``mu`` (or ``lame_second``); or ``density``,
            ``p_wave_velocity``, and ``s_wave_velocity``. The optional
            ``parameterization`` entry is ``"lame"`` or ``"velocity"``.

        Returns
        -------
        None
            The method assigns every material field and records the active
            parameterization.

        Examples
        --------
        Declaring ``{"density": ..., "lambda": ..., "mu": ...,
        "parameterization": "velocity"}`` builds the equation from Lame input
        but keeps ``c`` and ``c_s`` as the independent fields.
        """
        values = {
            parameter: next(
                (synthetic_data_dict[key]
                 for key in keys if key in synthetic_data_dict),
                None,
            )
            for parameter, keys in KEYS_BY_PARAMETER.items()
        }

        declared = [
            parameterization
            for parameterization, family in PARAMETERS_BY_PARAMETERIZATION.items()
            if {parameter for parameter, value in values.items()
                if value is not None} == set(family)
        ]
        if not declared:
            raise ValueError(
                "Inconsistent selection of isotropic elastic wave parameters:\n"
                + "".join(
                    f"    {parameter.value:<16}: {value is not None}\n"
                    for parameter, value in values.items()
                )
                + "The valid options are {Density, Lame first, Lame second} "
                "or (exclusive) {Density, P-wave velocity, S-wave velocity}",
            )

        for parameter in PARAMETERS_BY_PARAMETERIZATION[declared[0]]:
            self._store_material_parameter(parameter, values[parameter])
        self._control_parameterization = declared[0]
        self._derive_complementary_parameters(declared[0])

        parameterization = synthetic_data_dict.get("parameterization")
        if parameterization is not None:
            self.set_control_parameterization(parameterization)

    @property
    def control_parameterization(self):
        """Family of parameters held as independent fields by the equation.

        Returns
        -------
        ElasticMaterialParameterization or None
            Family whose parameters are Firedrake ``Function`` objects in the
            variational form, or ``None`` before the material parameters have
            been initialized. The two remaining parameters are UFL expressions
            of this family, so gradients can be taken with respect to it.
        """
        return self._control_parameterization

    def set_control_parameterization(self, parameterization):
        """Re-express the equation parameters in another material family.

        The parameters of ``parameterization`` are currently UFL expressions of
        the active family; interpolating them into the scalar control space
        turns them into independent fields, and the previous family is then
        re-derived from them. This is what allows inverting for wave velocities
        on a model declared with Lame moduli, and vice versa.

        Parameters
        ----------
        parameterization : str or ElasticMaterialParameterization
            Target family, ``"lame"`` or ``"velocity"``.

        Returns
        -------
        None
            Rewrites every material attribute when the family changes.

        Raises
        ------
        ValueError
            If ``parameterization`` is not a supported family, or if the
            material parameters have not been initialized yet.

        Examples
        --------
        ``wave.set_control_parameterization("velocity")`` on a model declared
        with ``lambda`` and ``mu`` makes ``c`` and ``c_s`` the independent
        fields, so the automated adjoint can differentiate with respect to
        wave velocities.
        """
        parameterization = resolve_parameterization(parameterization)
        if self._control_parameterization is None:
            raise ValueError(
                "Material parameters must be initialized before changing the "
                "elastic parameterization.",
            )
        if parameterization is self._control_parameterization:
            return
        if self.mesh is None:
            # There is no space to interpolate into yet. The request stays in
            # the model dictionary and is replayed once the mesh exists.
            self.input_dictionary["synthetic_data"]["parameterization"] = (
                parameterization.value
            )
            return

        for parameter in PARAMETERS_BY_PARAMETERIZATION[parameterization]:
            self._set_material_parameter(
                parameter,
                self._as_control_field(
                    self._get_material_parameter(parameter), parameter,
                ),
            )
        self._control_parameterization = parameterization
        self._derive_complementary_parameters(parameterization)
        self._record_material_parameters()

    def _store_material_parameter(self, parameter, value):
        """Store one declared material parameter as an equation coefficient.

        Scalars and ``Constant`` values become scalar material ``Function``
        objects once a mesh exists, and stay as ``Constant`` before that so the
        regular model initialization flow can continue. Any other value —
        a ``Function`` or a UFL expression — is stored unchanged.

        An existing compatible ``Function`` is updated in place rather than
        replaced. ``forward_solve()`` re-initializes the model on every call,
        and the automated adjoint holds references to these objects; rebuilding
        them would silently detach the recorded controls from the ones the
        variational form actually uses.

        Parameters
        ----------
        parameter : ElasticMaterialParameter
            Parameter being stored.
        value : scalar, firedrake.Constant, firedrake.Function, or UFL expression
            Value read from the model dictionary.

        Returns
        -------
        None
            Assigns the corresponding material attribute.
        """
        if not (np.isscalar(value) or isinstance(value, Constant)):
            self._set_material_parameter(parameter, value)
            return
        if self.mesh is None:
            self._set_material_parameter(
                parameter, Constant(value) if np.isscalar(value) else value,
            )
            return
        self._set_material_parameter(
            parameter, self._as_control_field(value, parameter),
        )

    def _derive_complementary_parameters(self, parameterization):
        """Express the non-control parameters in terms of the control family.

        The complementary parameters are stored as UFL expressions rather than
        interpolated fields, so the algebraic link between the two families
        stays inside the variational form and pyadjoint differentiates through
        it, whichever family the user selected as controls.

        Parameters
        ----------
        parameterization : ElasticMaterialParameterization
            Family currently held as independent ``Function`` objects.

        Returns
        -------
        None
            Assigns the two remaining material attributes.
        """
        if parameterization is ElasticMaterialParameterization.LAME:
            self.c = ((self.lmbda + 2*self.mu)/self.rho)**0.5
            self.c_s = (self.mu/self.rho)**0.5
        else:
            self.mu = self.rho*self.c_s**2
            self.lmbda = self.rho*self.c**2 - 2*self.mu

    def _get_material_parameter(self, parameter):
        """Return one material field or expression."""
        return getattr(self, ATTRIBUTE_BY_PARAMETER[parameter])

    def _set_material_parameter(self, parameter, value):
        """Set one material field or expression."""
        setattr(self, ATTRIBUTE_BY_PARAMETER[parameter], value)

    @override
    def initialize_model_parameters_from_file(self, synthetic_data_dict):
        raise NotImplementedError

    @override
    def _create_function_space(self):
        return create_function_space(self.mesh, self.method, self.degree,
                                     dim=self.dimension)

    @override
    def _set_vstate(self, vstate):
        self.u_n.assign(vstate)

    @override
    def _get_vstate(self):
        return self.u_n

    @override
    def _set_prev_vstate(self, vstate):
        if self.u_nm2 is not None:
            self.u_nm2.assign(self.u_nm1)
        self.u_nm1.assign(vstate)

    @override
    def _get_prev_vstate(self):
        return self.u_nm1

    @override
    def _set_next_vstate(self, vstate):
        self.u_np1.assign(vstate)

    @override
    def _get_next_vstate(self):
        return self.u_np1

    @override
    def get_forward_solution_receivers(self):
        if self.abc_type == AbsorbingBCsType.PML:
            raise NotImplementedError
        else:
            data_with_halos = self.u_n.dat.data_ro_with_halos[:]
        return self.receivers.interpolate(data_with_halos)

    @override
    def get_function(self):
        return self.u_n

    @override
    def get_function_name(self):
        return "Displacement"

    def get_control_parameter_function_space(self):
        """Return the scalar space used for elastic material controls.

        Elastic displacement is vector-valued, but density, Lame parameters,
        and wave speeds are scalar material fields. This method creates and
        returns the scalar space used for those controls.

        Returns
        -------
        firedrake.FunctionSpace
            Scalar material-parameter function space.

        Raises
        ------
        ValueError
            If the mesh has not been created yet.

        Examples
        --------
        ``Function(wave.get_control_parameter_function_space())`` creates a
        scalar density or Lame-parameter control compatible with
        ``set_control_parameters``.
        """
        if self.mesh is None:
            raise ValueError(
                "Mesh must be set before creating elastic control parameter spaces.",
            )
        space = self._material_parameter_function_space
        if space is None or space.mesh() is not self.mesh:
            space = create_function_space(self.mesh, self.method, self.degree)
            self._material_parameter_function_space = space
        return space

    def _as_control_field(self, value, parameter):
        """Return a material control as a scalar Firedrake Function.

        Elastic material parameters are scalar fields, while the elastic
        displacement solution lives in a vector function space. This helper
        keeps the inversion controls in the scalar space returned by
        ``get_control_parameter_function_space()`` so density, Lame
        parameters, and velocity controls can be flattened, rebuilt, written,
        and reassigned consistently during FWI.

        Accepted values are Firedrake ``Function`` objects, constants, scalar
        values, or UFL expressions. Functions already in the target space are
        copied with ``assign``; all other values are interpolated.

        The field currently held by the attribute is reused whenever it lives
        in the control space, so the objects referenced by the variational form
        and by :class:`AutomatedAdjoint` survive re-initialization. A new
        ``Function`` is built only when there is nothing compatible to write
        into.

        Parameters
        ----------
        value : firedrake.Function, firedrake.Constant, scalar, or UFL expression
            Material control value to represent in the scalar control space.
        parameter : ElasticMaterialParameter
            Material parameter being represented; identifies both the field to
            reuse and the name given to a newly created one.

        Returns
        -------
        firedrake.Function or None
            Scalar control field in the material-parameter function space. If
            ``value`` is ``None``, returns ``None``.
        """
        if value is None:
            return None

        V = self.get_control_parameter_function_space()
        field = self._get_material_parameter(parameter)
        if not (isinstance(field, Function) and field.function_space() == V):
            field = Function(V, name=parameter.value)
        if isinstance(value, Function) and value.function_space() == V:
            field.assign(value)
        else:
            field.interpolate(value)
        return field

    def get_control_parameters(self):
        """Return the isotropic elastic material fields available as controls.

        These are the three independent parameters of the active
        parameterization: every field the automated adjoint may differentiate
        with respect to. Which of them are actually differentiated is chosen
        with ``enable_automated_adjoint(controls=...)`` and stored on
        :class:`AutomatedAdjoint`, not here.

        Returns
        -------
        dict or None
            Dictionary mapping material-parameter enum values to scalar
            Firedrake ``Function`` controls. Returns ``None`` if material
            parameters have not been initialized.

        Examples
        --------
        Under the velocity parameterization this returns
        ``{DENSITY: rho, P_WAVE_VELOCITY: c, S_WAVE_VELOCITY: c_s}``.
        """
        parameterization = self._control_parameterization
        if parameterization is None:
            if self.rho is None:
                return None
            # Material attributes were assigned directly, bypassing the model
            # dictionary; fall back to the default Lame family.
            parameterization = ElasticMaterialParameterization.LAME
        return {
            parameter: self._get_material_parameter(parameter)
            for parameter in PARAMETERS_BY_PARAMETERIZATION[parameterization]
        }

    @override
    def _align_control_parameterization(self, parameters=None):
        """Re-express the equation in the family the controls belong to.

        Selecting parameters from the family that is not currently active
        makes them independent ``Function`` objects, which is what pyadjoint
        needs to differentiate with respect to them. Selecting ``density``
        alone is ambiguous and keeps the active family unchanged.

        Parameters
        ----------
        parameters : list, tuple, or None, optional
            Control names or :class:`ElasticMaterialParameter` values.
            ``None`` keeps the active family.

        Returns
        -------
        None

        Examples
        --------
        Requesting ``["lambda", "mu"]`` on a model declared with wave
        velocities switches the equation to the Lame family.
        """
        if parameters is None:
            return
        self.set_control_parameterization(
            ElasticControlSet.select(
                parameters, default=self._control_parameterization,
            ).parameterization,
        )

    @override
    def _select_control_parameters(self, parameters=None):
        """Resolve an elastic control selection into labeled fields.

        Only the parameters of the active family can be resolved: the other
        two are UFL expressions of them, not independent variables. Their
        gradients are still available after the fact, through the change of
        variables applied by :meth:`gradient_solve`.

        Parameters
        ----------
        parameters : list, tuple, or None, optional
            Control names (``"mu"``, ``"c_s"``, ``"lame_first"``, ...) or
            :class:`ElasticMaterialParameter` values. ``None`` selects the
            three parameters of the active family.

        Returns
        -------
        dict
            Ordered mapping from :class:`ElasticMaterialParameter` to the
            scalar ``Function`` differentiated for it.

        Raises
        ------
        ValueError
            If the selection is empty, has duplicates, mixes both families,
            names an unknown parameter, or belongs to the family that is not
            currently active.

        Examples
        --------
        ``wave._select_control_parameters(["mu"])`` under the Lame
        parameterization returns ``{MU: mu}``.
        """
        selection = ElasticControlSet.select(
            parameters, default=self._control_parameterization,
        )
        if selection.parameterization is not self._control_parameterization:
            raise ValueError(
                "Elastic controls "
                + ", ".join(parameter.value for parameter in selection)
                + f" belong to the '{selection.parameterization.value}' "
                "family, but the equation is parameterized with "
                f"'{self._control_parameterization.value}', so they are not "
                "independent variables of the recorded model. Use "
                "gradient_solve() to obtain their gradients through the "
                "change of variables, or set_control_parameterization() "
                "before recording the forward solve.",
            )
        return {
            parameter: self._get_material_parameter(parameter)
            for parameter in selection
        }

    @override
    def gradient_solve(
        self,
        misfit=None,
        forward_solution=None,
        adjoint_type=AdjointType.AUTOMATED_ADJOINT,
        riesz_map=RieszMapType.L2,
        controls=None,
    ):
        """Compute the adjoint gradient of the elastic misfit functional.

        Only the automated adjoint is available for elastic media: the
        gradient is obtained by replaying the pyadjoint tape recorded during
        an annotated ``forward_solve()``.

        Controls from the active parameterization are read straight off the
        tape. Controls from the other family are obtained from the full active
        family by the exact change of variables in :mod:`.material`, so no
        second forward solve is needed either way.

        Parameters
        ----------
        misfit : array_like, optional
            Accepted for signature compatibility with
            :meth:`AcousticWave.gradient_solve`; the automated adjoint reads
            the misfit from the recorded tape instead.
        forward_solution : firedrake.Function, optional
            Accepted for signature compatibility; unused for the same reason.
        adjoint_type : AdjointType, optional
            Must be :attr:`AdjointType.AUTOMATED_ADJOINT`.
        riesz_map : RieszMapType, optional
            ``L2`` returns gradients (``Function``), ``l2`` returns raw
            derivatives (``Cofunction``). See :class:`RieszMapType`.
        controls : list, tuple, or None, optional
            Restrict the gradient to a subset of the material parameters, from
            either family. ``None`` uses the selection made in
            ``enable_automated_adjoint()``. Passing a selection re-registers it
            on the automated adjoint, since that is where the choice lives; a
            request from the other family registers the whole active family,
            which is what the tape is actually read for.

        Returns
        -------
        dict
            Derivative of the functional with respect to each selected
            control, keyed by :class:`ElasticMaterialParameter`.

        Raises
        ------
        NotImplementedError
            If a hand-implemented adjoint is requested.

        Examples
        --------
        After ``enable_automated_adjoint(controls=["lambda", "mu"])``,
        ``gradient_solve(controls=["mu"])`` returns only ``{MU: dJ_dmu}`` and
        ``gradient_solve(controls=["c_s"])`` returns ``{S_WAVE_VELOCITY: ...}``
        by converting the Lame gradients, both reusing the same tape.
        """
        if adjoint_type is not AdjointType.AUTOMATED_ADJOINT:
            raise NotImplementedError(
                "Elastic media only support the automated adjoint; "
                f"got {adjoint_type}.",
            )
        selection = ElasticControlSet.select(
            controls, default=self._control_parameterization,
        )
        if selection.parameterization is self._control_parameterization:
            return self._automated_adjoint_derivatives(
                riesz_map=riesz_map, controls=controls,
            )

        # The requested family is not on the tape. Differentiate the whole
        # active family, which is, and change variables afterwards.
        derivatives = self._automated_adjoint_derivatives(
            riesz_map=RieszMapType.l2,
            controls=PARAMETERS_BY_PARAMETERIZATION[
                self._control_parameterization
            ],
        )
        converted = self._change_derivative_variables(
            derivatives, selection.parameterization,
        )
        return {
            parameter: (
                converted[parameter]
                if riesz_map is RieszMapType.l2
                else converted[parameter].riesz_representation("L2")
            )
            for parameter in selection
        }

    def _change_derivative_variables(self, derivatives, parameterization):
        """Chain-rule dual derivatives into the other material family.

        The conversion between the two families is applied node by node when
        the material is re-expressed, so its Jacobian is diagonal and the
        chain rule is an exact, coefficient-wise operation on the dual vectors.
        That is why this works on raw derivatives (``Cofunction``) rather than
        on Riesz representers: scaling coefficients does not commute with
        inverting the mass matrix, so the Riesz map must be applied afterwards.

        Parameters
        ----------
        derivatives : dict
            Raw derivatives for every parameter of the active family, keyed by
            :class:`ElasticMaterialParameter`. All three are required, since
            each converted parameter mixes several of them.
        parameterization : ElasticMaterialParameterization
            Family to convert to.

        Returns
        -------
        dict
            Raw derivatives with respect to ``parameterization``, keyed by
            :class:`ElasticMaterialParameter`.
        """
        V = self.get_control_parameter_function_space()
        coefficients = DERIVATIVES_BY_PARAMETERIZATION[parameterization](
            self.rho, self.c, self.c_s,
        )
        converted = {}
        for parameter, terms in coefficients.items():
            dual = Cofunction(V.dual(), name=parameter.value)
            for source, coefficient in terms.items():
                nodal = Function(V).interpolate(coefficient).dat.data_ro
                dual.dat.data[:] += derivatives[source].dat.data_ro * nodal
            converted[parameter] = dual
        return converted

    def set_control_parameters(self, controls):
        """Assign isotropic elastic material controls.

        Control dictionaries must use :class:`ElasticMaterialParameter` keys
        and may contain any non-empty subset of a single material family.
        Parameters omitted from the dictionary keep their current values, and
        the complementary family is re-derived from the result.

        Parameters
        ----------
        controls : dict
            Dictionary containing a non-empty subset of density/Lame controls
            or density/P- and S-wave velocity controls.

        Returns
        -------
        None
            The method updates the assigned fields, the complementary material
            expressions, and the active parameterization.

        Raises
        ------
        TypeError
            If ``controls`` is not a dictionary or if any key is not an
            ``ElasticMaterialParameter``.
        ValueError
            If the dictionary is empty or mixes both parameter families.

        Examples
        --------
        A subset of Lame controls is passed as::

            {
                ElasticMaterialParameter.LAMBDA: lmbda,
                ElasticMaterialParameter.MU: mu,
            }

        A single velocity control is passed as::

            {
                ElasticMaterialParameter.S_WAVE_VELOCITY: c_s,
            }
        """
        if not isinstance(controls, dict):
            raise TypeError(
                "IsotropicWave controls must be provided as a dictionary.",
            )

        if not all(isinstance(key, ElasticMaterialParameter) for key in controls):
            raise TypeError(
                "IsotropicWave control keys must be ElasticMaterialParameter "
                "enum values.",
            )

        selection = ElasticControlSet.select(
            list(controls),
            default=self._control_parameterization,
        )
        if self._control_parameterization is None:
            # A dictionary covering a complete family fully determines the
            # material; a partial one only makes sense against an existing
            # material, so the model dictionary is used to build one first.
            if len(selection) == len(
                PARAMETERS_BY_PARAMETERIZATION[selection.parameterization]
            ):
                self._control_parameterization = selection.parameterization
            else:
                self._initialize_model_parameters()
        self.set_control_parameterization(selection.parameterization)

        for parameter, value in controls.items():
            self._set_material_parameter(
                parameter,
                self._as_control_field(value, parameter),
            )
        self._derive_complementary_parameters(selection.parameterization)
        self._record_material_parameters()

    def _record_material_parameters(self):
        """Write the active material state back to the model dictionary.

        ``forward_solve()`` re-initializes the model from ``synthetic_data`` on
        every call, so the dictionary must keep describing the current state:
        the fields as they stand now, and the family holding them. Keys of the
        other family are dropped, because a declaration naming both families is
        rejected as ambiguous.

        Storing the ``Function`` objects themselves — rather than the values
        the model was built from — is also what preserves their identity across
        re-initialization, so the controls recorded by
        :class:`AutomatedAdjoint` keep pointing at the fields in the form.

        Returns
        -------
        None
            Updates ``self.input_dictionary["synthetic_data"]`` in place.
        """
        synthetic_data = self.input_dictionary["synthetic_data"]
        synthetic_data["parameterization"] = (
            self._control_parameterization.value
        )
        active = PARAMETERS_BY_PARAMETERIZATION[self._control_parameterization]
        for parameter, keys in KEYS_BY_PARAMETER.items():
            for key in keys:
                synthetic_data.pop(key, None)
            if parameter in active:
                synthetic_data[keys[0]] = self._get_material_parameter(parameter)

    @override
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

    @override
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
