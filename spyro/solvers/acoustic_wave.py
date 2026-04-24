import firedrake as fire

from .wave import Wave

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
from ..utils.typing import AdjointType, RieszMapType, override, WaveType
from ..utils import write_hdf5_velocity_model
from .functionals import acoustic_energy


class AcousticWave(Wave):
    def __init__(self, dictionary, comm=None):
        super().__init__(dictionary, comm=comm)
        self.wave_type = WaveType.ISOTROPIC_ACOUSTIC
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
        self, guess=None, misfit=None, forward_solution=None,
        adjoint_type=AdjointType.IMPLEMENTED_ADJOINT,
        riesz_map=RieszMapType.L2
    ):
        """Solve the adjoint problem to calculate the gradient.

        Parameters:
        -----------
        guess: Firedrake 'Function' (optional)
            Initial guess for the velocity model. If not mentioned uses the
            one currently in the wave object.

        Returns:
        --------
        dJ: Firedrake 'Function'
            Gradient of the cost functional.
        """
        if adjoint_type == AdjointType.AUTOMATED_ADJOINT:
            if self.automated_adjoint.reduced_functional is None:
                self.forward_solve()
                self.automated_adjoint.create_reduced_functional(
                    self.functional_value
                )
            else:
                self.functional_value = self.automated_adjoint.recompute_functional(
                    self.c
                )

            if riesz_map == RieszMapType.L2:
                dJ = self.automated_adjoint.compute_gradient()
            elif riesz_map == RieszMapType.l2:
                dJ = self.automated_adjoint.compute_derivative()
            else:
                raise NotImplementedError(
                    f"Riesz map {riesz_map} not implemented for automated adjoint."
                )

            if isinstance(dJ, fire.Cofunction):
                return fire.Function(self.function_space, val=dJ)
            return dJ

        if adjoint_type != AdjointType.IMPLEMENTED_ADJOINT:
            self.enable_implemented_adjoint()

        if misfit is not None:
            self.misfit = misfit
        if forward_solution is not None:
            self.forward_solution = forward_solution
        elif self.current_time == 0.0:
            self.forward_solve()
        elif self.misfit is None:
            raise ValueError("Please load or calculate a real shot record first")

        if self.misfit is None:
            self.misfit = self.real_shot_record - self.forward_solution_receivers
        return backward_wave_propagator(self)

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
