import firedrake as fire

from .wave import Wave
from pyadjoint import Tape, AdjFloat
from ..io.basicio import ensemble_gradient
from ..io import interpolate
from .acoustic_solver_construction_no_pml import (
    construct_solver_or_matrix_no_pml,
)
from .acoustic_solver_construction_with_pml import (
    construct_solver_or_matrix_with_pml,
)
from .backward_time_integration import (
    backward_wave_propagator,
)
from ..domains.space import create_function_space
from ..utils.typing import (
    AdjointType, RieszMapType, override, WaveType,
)
from ..utils import write_hdf5_velocity_model
from .functionals import acoustic_energy


class AcousticWave(Wave):
    def __init__(self, dictionary, comm=None):
        super().__init__(dictionary, comm=comm)
        self.wave_type = WaveType.ISOTROPIC_ACOUSTIC

        # In case sources and reeivers were initialized in super we have to pass the wave_type
        if getattr(self, 'sources', None) is not None:
            self.sources.wave_type = self.wave_type
        if getattr(self, 'receivers', None) is not None:
            self.receivers.wave_type = self.wave_type
        self.acoustic_energy = None
        self.field_logger.add_functional(
            "acoustic_energy", lambda: fire.assemble(self.acoustic_energy))

    def save_current_velocity_model(self, file_name=None):
        if self.c is None:
            raise ValueError("C not loaded")
        if file_name is None:
            file_name = "velocity_model.pvd"
        fire.VTKFile(file_name).write(
            self.c, name="velocity"
        )

    @override
    def matrix_building(self):
        """Builds solver operators. Doesn't create mass matrices if
        matrix_free option is on,
        which it is by default.
        """
        self.current_time = 0.0

        abc_type = self.abc_boundary_layer_type

        # Just to document variables that will be overwritten
        self.trial_function = None
        self.u_nm1 = None
        self.u_n = None
        self.u_np1 = fire.Function(self.function_space)
        self.lhs = None
        self.solver = None
        self.rhs = None
        self.B = None
        if abc_type is None or abc_type == "local" or abc_type == "hybrid":
            construct_solver_or_matrix_no_pml(self)
        elif abc_type == "PML":
            V = self.function_space
            Z = fire.VectorFunctionSpace(V.ufl_domain(), V.ufl_element())
            self.vector_function_space = Z
            self.X_np1 = None
            self.X_n = None
            self.X_nm1 = None
            construct_solver_or_matrix_with_pml(self)

        self.acoustic_energy = acoustic_energy(self)

    @ensemble_gradient
    def gradient_solve(
        self, misfit=None, forward_solution=None,
        adjoint_type=AdjointType.UFL_DERIVED_ADJOINT,
        riesz_map=RieszMapType.L2,
    ):
        """Compute the adjoint-based gradient.

        Parameters:
        -----------
        misfit: Firedrake 'Function' or numpy array (optional)
            The misfit between the observed and predicted data. If not provided,
            it will be computed as the difference between the real shot record and
            the forward solution at the receivers. If the real shot record is not
            available, the method will raise an error.
        forward_solution: Firedrake 'Function' (optional)
            The forward solution of the wave equation. If not provided, it will be
            computed by calling the forward solver. Providing the forward solution
            can save computational time if it has already been
            computed for the current velocity model, as it avoids redundant forward solves.
        adjoint_type: AdjointType enum
            Whether to use automated adjoint differentiation.
        riesz_map: RieszMapType enum (default: RieszMapType.L2)
            The type of Riesz map to use for the gradient. More details in the documentation of the
            :class:`RieszMapType` enum.

        Returns:
        --------
        dJ: Firedrake 'Function' or Firedrake 'Cofunction'
            Gradient (Function) or derivative (Cofunction) of the functional with respect to the velocity model,
            depending on the chosen Riesz map.
        """
        if adjoint_type == AdjointType.AUTOMATED_ADJOINT:
            return self._automated_adjoint_gradient(riesz_map=riesz_map)
        if not adjoint_type.is_implemented:
            raise NotImplementedError(
                f"Adjoint type {adjoint_type} is not implemented for gradients.",
            )

        if riesz_map != RieszMapType.L2:
            raise NotImplementedError(
                f"Riesz map {riesz_map} not implemented for implemented adjoint."
            )

        self._prepare_implemented_adjoint(
            misfit=misfit, forward_solution=forward_solution,
            adjoint_type=adjoint_type,
        )
        return backward_wave_propagator(
            self,
            adjoint_type=adjoint_type,
        )

    def _automated_adjoint_gradient(self, riesz_map=RieszMapType.L2):
        """Compute the gradient using the automated adjoint.

        Parameters:
        -----------
        riesz_map: RieszMapType enum (default: RieszMapType.L2)
            The type of Riesz map to use for the gradient. More details in the documentation of the
            :class:`RieszMapType` enum.

        Returns:
        --------
        dJ: Firedrake 'Function' or Firedrake 'Cofunction'
            Gradient (Function) or derivative (Cofunction) of the functional with respect to the velocity model,
            depending on the chosen Riesz map.
        """
        if not isinstance(self.functional_value, AdjFloat):
            raise ValueError(
                "Functional value must be an AdjFloat for automated adjoint gradient computation."
            )

        if not self.automated_adjoint:
            self.enable_automated_adjoint()
            self.automated_adjoint.clear_tape()
            self.forward_solve()
            self.automated_adjoint.create_reduced_functional(
                self.functional_value
            )
        elif (
            self.automated_adjoint.reduced_functional is None
            and isinstance(self.automated_adjoint._tape, Tape)
        ):
            self.automated_adjoint.create_reduced_functional(
                self.functional_value
            )

        if riesz_map == RieszMapType.L2:
            return self.automated_adjoint.compute_gradient()
        elif riesz_map == RieszMapType.l2:
            return self.automated_adjoint.compute_derivative()
        else:
            raise NotImplementedError(
                f"Riesz map {riesz_map} not implemented for automated adjoint."
            )

    def reset_pressure(self):
        if self.abc_boundary_layer_type == "PML":
            self.X_n.assign(0.0)
            self.X_nm1.assign(0.0)
        else:
            self.u_nm1.assign(0.0)
            self.u_n.assign(0.0)

    @override
    def _initialize_model_parameters(self):
        if self.initial_velocity_model is None:
            if self.initial_velocity_model_file is None:
                if getattr(self.mesh_parameters, "grid_velocity_data", None) is not None:
                    self.initial_velocity_model = interpolate(
                        self,
                        self.mesh_parameters.grid_velocity_data,
                        self.function_space.sub(0),
                    )
                    if self.debug_output:
                        fire.VTKFile("initial_velocity_model.pvd").write(
                            self.initial_velocity_model, name="velocity"
                        )
                    self.c = self.initial_velocity_model
                    return
                raise ValueError("No velocity model or velocity file to load.")

            if self.initial_velocity_model_file.endswith(".segy"):
                self.initial_velocity_model_file = write_hdf5_velocity_model(self, self.initial_velocity_model_file)

            if self.initial_velocity_model_file.endswith((".hdf5", ".h5")):
                self.initial_velocity_model = interpolate(
                    self,
                    self.initial_velocity_model_file,
                    self.function_space.sub(0),
                )

            if self.debug_output:
                fire.VTKFile("initial_velocity_model.pvd").write(
                    self.initial_velocity_model, name="velocity"
                )

        self.c = self.initial_velocity_model

    @override
    def _set_vstate(self, vstate):
        if self.abc_boundary_layer_type == "PML":
            self.X_n.assign(vstate)
        else:
            self.u_n.assign(vstate)

    @override
    def _get_vstate(self):
        if self.abc_boundary_layer_type == "PML":
            return self.X_n
        else:
            return self.u_n

    @override
    def _set_prev_vstate(self, vstate):
        if self.abc_boundary_layer_type == "PML":
            self.X_nm1.assign(vstate)
        else:
            self.u_nm1.assign(vstate)

    @override
    def _get_prev_vstate(self):
        if self.abc_boundary_layer_type == "PML":
            return self.X_nm1
        else:
            return self.u_nm1

    @override
    def _set_next_vstate(self, vstate):
        if self.abc_boundary_layer_type == "PML":
            self.X_np1.assign(vstate)
        else:
            self.u_np1.assign(vstate)

    @override
    def _get_next_vstate(self):
        if self.abc_boundary_layer_type == "PML":
            return self.X_np1
        else:
            return self.u_np1

    @override
    def get_forward_solution_receivers(self):
        if self.abc_boundary_layer_type == "PML":
            data_with_halos = self.X_n.dat.data_ro_with_halos[0][:]
        else:
            data_with_halos = self.u_n.dat.data_ro_with_halos[:]
        return self.receivers.interpolate(data_with_halos)

    @override
    def get_function(self, state: fire.Function = None) -> fire.Function:
        """Return the wave equation solution.

        If `state` is provided, return the wave field corresponding to that
        state (e.g., for PML, the first component of the state vector). If `state`
        is `None`, return the wave field corresponding to the time step ``n``.
        For PML, this corresponds to the first component of X_n.

        Parameters:
        -----------
        state : Firedrake 'Function' (optional)
            The state for which to return the wave field. If None, returns the
            wave field corresponding to the time step ``n``.

        Returns:
        --------
        Firedrake 'Function'
            The scalar wave field corresponding to the specified `state` or the time step ``n``.
        """
        if state is None:
            if self.abc_boundary_layer_type == "PML":
                return self.X_n.sub(0)
            else:
                return self.u_n
        else:
            if self.abc_boundary_layer_type == "PML":
                return state.sub(0)
            else:
                return state

    def get_scalar_function_space(self) -> fire.FunctionSpace:
        """Return the scalar space where the pressure equation is solved."""
        if self.scalar_function_space is not None:
            return self.scalar_function_space
        else:
            raise ValueError("Scalar function space not found in wave object.")

    @override
    def get_function_name(self):
        return "Pressure"

    @override
    def _create_function_space(self):
        return create_function_space(self.mesh, self.method, self.degree)

    @override
    def rhs_no_pml(self):
        if self.abc_boundary_layer_type == "PML":
            return self.B.sub(0)
        else:
            return self.B

    def rhs_no_pml_source(self):
        """Return the source cofunction added to the variational right-hand
        side.
        """
        if self.abc_boundary_layer_type == "PML":
            return self.source_function.sub(0)
        else:
            return self.source_function

    def pressure_for_receivers(self):
        """Return the expression to be interpolated at receiver locations.

        For the PML formulation the pressure field is the first component of
        the mixed Function ``X_n``. We deliberately return the UFL ``split``
        expression rather than the subfunction ``Function`` view so that
        pyadjoint's annotation of ``X_n.assign(X_np1)`` (the only operation
        that updates state between time steps) is sufficient to make the
        receiver interpolation reflect the time-stepping during tape replay.
        For the non-mixed case ``self.u_n`` is already the form coefficient
        Function, which is updated directly by ``u_n.assign(...)``.
        """
        if self.abc_boundary_layer_type == "PML":
            return fire.split(self.X_n)[0]
        return self.u_n

    def get_control_parameters(self):
        """Return the acoustic inversion control.

        For acoustic FWI the control is the velocity model stored in
        ``initial_velocity_model``.

        Returns
        -------
        firedrake.Function or None
            Current acoustic velocity model.

        Examples
        --------
        After ``set_initial_velocity_model(constant=2.0)``, this method returns
        the velocity ``Function`` filled with ``2.0``.
        """
        return self.initial_velocity_model

    def set_control_parameters(self, controls):
        """Assign the acoustic inversion control.

        Parameters
        ----------
        controls : firedrake.Function, firedrake.Constant, scalar, or UFL expression
            Velocity model control. Non-``Function`` values are interpolated
            into the acoustic function space.

        Returns
        -------
        None
            The method updates ``initial_velocity_model`` and ``c`` and clears
            ``initial_velocity_model_file``.

        Examples
        --------
        ``set_control_parameters(fire.Constant(2.0))`` creates a velocity
        ``Function`` in the acoustic function space and fills it with ``2.0``.
        """
        if self.function_space is None:
            self.force_rebuild_function_space()

        if isinstance(controls, fire.Function):
            name = controls.name()
            velocity = fire.Function(self.function_space, name=name)
            if controls.function_space() == self.function_space:
                velocity.assign(controls)
            else:
                velocity.interpolate(controls)
        else:
            velocity = fire.Function(self.function_space, name="velocity")
            velocity.interpolate(controls)

        self.initial_velocity_model = velocity
        self.initial_velocity_model_file = None
        self.c = self.initial_velocity_model

    def get_control_parameter_function_space(self):
        """Return the function space used by acoustic controls.

        Returns
        -------
        firedrake.FunctionSpace
            Acoustic solver function space. If it has not been built yet, it is
            created before being returned.

        Examples
        --------
        ``fire.Function(wave.get_control_parameter_function_space())`` creates
        a velocity control compatible with ``set_control_parameters``.
        """
        if self.function_space is None:
            self.force_rebuild_function_space()
        return self.function_space
