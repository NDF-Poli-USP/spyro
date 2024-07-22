import firedrake as fire
import warnings

from .wave import Wave
from .time_integration import time_integrator
from ..io.basicio import ensemble_propagator, ensemble_gradient
from ..domains.quadrature import quadrature_rules
from .acoustic_solver_construction_no_pml import (
    construct_solver_or_matrix_no_pml,
)
from .acoustic_solver_construction_with_pml import (
    construct_solver_or_matrix_with_pml,
)
from .backward_time_integration import (
    backward_wave_propagator,
)
from ..utils.typing import override

class AcousticWave(Wave):
    def save_current_velocity_model(self, file_name=None):
        if self.c is None:
            raise ValueError("C not loaded")
        if file_name is None:
            file_name = "velocity_model.pvd"
        fire.File(file_name).write(
            self.c, name="velocity"
        )

    def forward_solve(self):
        """Solves the forward problem.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        if self.function_space is None:
            self.force_rebuild_function_space()

        self._initialize_model_parameters()
        self.c = self.initial_velocity_model
        self.matrix_building()
        self.wave_propagator()

    def force_rebuild_function_space(self):
        if self.mesh is None:
            self.mesh = self.get_mesh()
        self._build_function_space()
        self._map_sources_and_receivers()

    def matrix_building(self):
        """Builds solver operators. Doesn't create mass matrices if
        matrix_free option is on,
        which it is by default.
        """
        self.current_time = 0.0
        quad_rule, k_rule, s_rule = quadrature_rules(self.function_space)
        self.quadrature_rule = quad_rule
        self.stiffness_quadrature_rule = k_rule
        self.surface_quadrature_rule = s_rule

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
        if abc_type is None:
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

    @ensemble_propagator
    def wave_propagator(self, dt=None, final_time=None, source_num=0):
        """Propagates the wave forward in time.
        Currently uses central differences.

        Parameters:
        -----------
        dt: Python 'float' (optional)
            Time step to be used explicitly. If not mentioned uses the default,
            that was estabilished in the wave object.
        final_time: Python 'float' (optional)
            Time which simulation ends. If not mentioned uses the default,
            that was estabilished in the wave object.

        Returns:
        --------
        usol: Firedrake 'Function'
            Pressure wavefield at the final time.
        u_rec: numpy array
            Pressure wavefield at the receivers across the timesteps.
        """
        if final_time is not None:
            self.final_time = final_time
        if dt is not None:
            self.dt = dt

        self.current_source = source_num
        usol, usol_recv = time_integrator(self, source_id=source_num)

        return usol, usol_recv

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
        if self.real_shot_record is None:
            warnings.warn("Please load or calculate a real shot record first")
        if self.current_time == 0.0:
            self.forward_solve()
            self.misfit = self.real_shot_record - self.forward_solution_receivers
        return backward_wave_propagator(self)

    def reset_pressure(self):
        try:
            self.u_nm1.assign(0.0)
            self.u_n.assign(0.0)
        except:
            warnings.warn("No pressure to reset")

    @override
    def _initialize_model_parameters(self):
        if self.initial_velocity_model is not None:
            return None

        if self.initial_velocity_model_file is None:
            raise ValueError("No velocity model or velocity file to load.")

        if self.initial_velocity_model_file.endswith(".segy"):
            vp_filename, vp_filetype = os.path.splitext(
                self.initial_velocity_model_file
            )
            warnings.warn("Converting segy file to hdf5")
            write_velocity_model(
                self.initial_velocity_model_file, ofname=vp_filename
            )
            self.initial_velocity_model_file = vp_filename + ".hdf5"

        if self.initial_velocity_model_file.endswith((".hdf5", ".h5")):
            self.initial_velocity_model = interpolate(
                self,
                self.initial_velocity_model_file,
                self.function_space.sub(0),
            )

        if self.debug_output:
            fire.File("initial_velocity_model.pvd").write(
                self.initial_velocity_model, name="velocity"
            )

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
