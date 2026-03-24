import firedrake as fire
import warnings

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
from ..utils.typing import override, WaveType, AdjointType, RieszMapType
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
        self, misfit=None, forward_solution=None, riesz_map=RieszMapType.L2
    ):
        """Solves the adjoint problem to calculate de gradient.

        Parameters:
        -----------
        misfit: list of Firedrake Functions or list of numpy arrays
            A list containing the misfit at each time step.
            If not provided, it will be calculated as the difference between
            the real shot record and the forward solution shots at the
            receiver locations.
        forward_solution: list of Firedrake Functions or list of numpy arrays
            A list containing the forward solution at each time step.
            If not provided, the forward problem will be solved to obtain it.
        riesz_map: RieszMapType, optional
            The type of Riesz map to apply to the gradient. If using the
            automated adjoint and the optimization library is scipy, this will
            be set to RieszMapType.l2 regardless of the value passed here,
            since scipy's optimizers expect derivatives rather than gradients.
            Default is RieszMapType.L2, which applies the L2 Riesz map to
            the computed derivative to obtain the gradient in the appropriate
            inner product space.

        Notes:
        ------
        - For automated adjoint, the misfit and forward_solution are not needed
        as they are handled internally by the recording and recomputation
        mechanism.

        Returns:
        --------
        dJ: Firedrake 'Function'
            Gradient of the cost functional.
        """
        if self.adjoint_type == AdjointType.AUTOMATED_ADJOINT:
            if self.automated_adjoint.reduced_functional is None:
                self.forward_solve()
                self.automated_adjoint.create_reduced_functional(
                    self.functional_value
                )
            else:
                if self.comm.ensemble_size == 1 and self.number_of_sources > 1:
                    # In serial runs, the source cofunction must be passed back
                    # into the reduced functional so it is refreshed with the
                    # current source values before recomputing the functional.
                    # The returned derivative is still taken only with respect
                    # to the velocity-model control: in
                    # ``AutomatedAdjoint.create_reduced_functional()``, both
                    # controls are registered, but ``derivative_components`` is
                    # set to ``(model_control_index,)``. That keeps the source
                    # cofunction synchronized during recomputation while
                    # restricting the reported gradient to ``self.c``.
                    controls = [self.c, self.source_cofunction]
                else:
                    controls = self.c
                self.automated_adjoint.recompute_functional(controls)
            if riesz_map == RieszMapType.L2:
                gradient = self.automated_adjoint.compute_gradient()
            elif riesz_map == RieszMapType.l2:
                gradient = self.automated_adjoint.compute_derivative()
            else:
                raise ValueError(f"Unsupported Riesz map type: {riesz_map}")
            if isinstance(gradient, (list, tuple)):
                if len(gradient) == 0:
                    raise ValueError(
                        "Automated adjoint returned no derivative "
                        "components for the model control."
                    )
                gradient = gradient[0]
            if isinstance(gradient, fire.Cofunction):
                gradient_function = fire.Function(self.function_space)
                gradient_function.dat.data[:] = gradient.dat.data_ro[:]
                return gradient_function
            return gradient
        # Implemented adjoint case
        self.enable_implemented_adjoint(
            misfit=misfit, forward_solution=forward_solution)
        if misfit is None:
            self.forward_solve()
            self.misfit = (
                self.real_shot_record - self.forward_solution_receivers)
        return backward_wave_propagator(self)

    def reset_pressure(self):
        try:
            self.u_nm1.assign(0.0)
            self.u_n.assign(0.0)
        except Exception:
            warnings.warn("No pressure to reset")

    @override
    def _initialize_model_parameters(self):
        if self.initial_velocity_model is None:
            if self.initial_velocity_model_file is None:
                raise ValueError("No velocity model or velocity file to load.")

            if self.initial_velocity_model_file.endswith(".segy"):
                self.initial_velocity_model_file = write_hdf5_velocity_model(
                    self, self.initial_velocity_model_file)

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
    def get_receivers_output(self):
        if self.abc_boundary_layer_type == "PML":
            data_with_halos = self.X_n.dat.data_ro_with_halos[0][:]
        else:
            data_with_halos = self.u_n.dat.data_ro_with_halos[:]
        return self.receivers.interpolate(data_with_halos)

    @override
    def get_function(self):
        if self.abc_boundary_layer_type == "PML":
            return self.X_n.sub(0)
        else:
            return self.u_n

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
            return self.source_cofunction.sub(0)
        else:
            return self.source_cofunction
