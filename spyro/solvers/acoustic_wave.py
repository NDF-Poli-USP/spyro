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
from ..utils.typing import override, WaveType
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
            self.X = None
            self.X_n = None
            self.X_nm1 = None
            self.X_np1 = fire.Function(V * Z)
            construct_solver_or_matrix_with_pml(self)

        self.acoustic_energy = acoustic_energy(self)

    @ensemble_gradient
    def gradient_solve(self, guess=None, misfit=None, forward_solution=None):
        """Solves the adjoint problem to calculate de gradient.

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
        if misfit is not None:
            self.misfit = misfit
        elif self.current_time == 0.0:
            self.forward_solve()
            self.misfit = self.real_shot_record - self.forward_solution_receivers
        else:
            raise ValueError("Please load or calculate a real shot record first")
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
