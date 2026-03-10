import firedrake as fire
import firedrake.adjoint as fire_adj
import warnings
import os

from .wave import Wave
from .automatic_differentiation_solver import SpyroReducedFunctional

from ..io.basicio import ensemble_gradient, switch_serial_shot
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
from .functionals import acoustic_energy

try:
    from SeismicMesh import write_velocity_model
    SEISMIC_MESH_AVAILABLE = True
except ImportError:
    SEISMIC_MESH_AVAILABLE = False


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

    def _resolve_true_receivers(self, true_receivers=None):
        if true_receivers is not None:
            self.true_receivers = true_receivers
        elif self.true_receivers is None and hasattr(self, "real_shot_record"):
            self.true_receivers = self.real_shot_record

        return self.true_receivers

    def _build_misfit_from_observed_receivers(self, observed_receivers):
        if observed_receivers is None:
            raise ValueError(
                "Please set wave.true_receivers or wave.real_shot_record before computing the gradient."
            )

        if self.parallelism_type == "spatial" and self.number_of_sources > 1:
            misfit = []
            for source_number in range(self.number_of_sources):
                switch_serial_shot(self, source_number)
                misfit.append(
                    observed_receivers[source_number] - self.forward_solution_receivers
                )
            return misfit

        return observed_receivers - self.forward_solution_receivers

    def _validate_automatic_adjoint_gradient_inputs(self):
        if not self.compute_functional:
            raise ValueError(
                "Set wave.compute_functional to True before calling compute_gradient() with automatic adjoint."
            )

        if self.abc_boundary_layer_type == "PML":
            raise NotImplementedError(
                "Automatic adjoint with SpyroReducedFunctional is not supported for PML acoustic waves."
            )

        if self._resolve_true_receivers() is None:
            raise ValueError(
                "Set wave.true_receivers before calling compute_gradient() with automatic adjoint."
            )

        if not self.use_vertex_only_mesh:
            raise ValueError(
                "Automatic adjoint requires acquisition.use_vertex_only_mesh=True."
            )

        if self.number_of_sources != 1:
            raise NotImplementedError(
                "Automatic adjoint currently supports only single-source propagations."
            )

    def _compute_gradient_with_automatic_adjoint(
        self,
        store_receivers_output=True,
        **kwargs,
    ):
        self._validate_automatic_adjoint_gradient_inputs()

        fire_adj.pause_annotation()
        tape = fire_adj.get_working_tape()
        if tape is not None:
            tape.clear_tape()

        fire_adj.continue_annotation()
        try:
            self.forward_solve(
                store_receivers_output=store_receivers_output,
                compute_functional=True,
                true_receivers=self.true_receivers,
                **kwargs,
            )
        finally:
            fire_adj.pause_annotation()

        if self.functional_evaluation is None:
            raise RuntimeError(
                "Forward solve did not produce a functional evaluation for automatic adjoint differentiation."
            )

        self.misfit = self.true_receivers - self.forward_solution_receivers
        reduced_functional = SpyroReducedFunctional(
            self.functional_evaluation,
            self.c,
        )
        return reduced_functional.compute_gradient()

    @ensemble_gradient
    def _compute_gradient_with_discrete_adjoint(
        self,
        misfit,
        forward_solution=None,
    ):
        self.misfit = misfit
        if forward_solution is None:
            forward_solution = self.forward_solution

        return backward_wave_propagator(self, forward_solution=forward_solution)

    def compute_gradient(
        self,
        store_receivers_output=True,
        compute_functional=None,
        true_receivers=None,
        guess=None,
        misfit=None,
        forward_solution=None,
        **kwargs,
    ):
        """Compute the gradient with either the discrete or automatic adjoint.

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
        if guess is not None:
            self.initial_velocity_model = guess

        if compute_functional is not None:
            self.compute_functional = compute_functional

        observed_receivers = self._resolve_true_receivers(true_receivers)

        if self.automatic_adjoint:
            return self._compute_gradient_with_automatic_adjoint(
                store_receivers_output=store_receivers_output,
                **kwargs,
            )

        needs_forward_propagation = (
            forward_solution is None and len(self.forward_solution) == 0
        ) or (misfit is None and self.forward_solution_receivers is None)

        if needs_forward_propagation:
            self.forward_solve(
                store_receivers_output=store_receivers_output,
                compute_functional=False,
                **kwargs,
            )

        if misfit is None:
            misfit = self._build_misfit_from_observed_receivers(observed_receivers)

        return self._compute_gradient_with_discrete_adjoint(
            misfit=misfit,
            forward_solution=forward_solution,
        )

    def gradient_solve(
        self,
        store_receivers_output=True,
        compute_functional=None,
        guess=None,
        misfit=None,
        forward_solution=None,
        true_receivers=None,
        **kwargs,
    ):
        return self.compute_gradient(
            store_receivers_output=store_receivers_output,
            compute_functional=compute_functional,
            true_receivers=true_receivers,
            guess=guess,
            misfit=misfit,
            forward_solution=forward_solution,
            **kwargs,
        )

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
                if not SEISMIC_MESH_AVAILABLE:
                    raise ImportError("SeismicMesh is required to convert segy files.")
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
            return self.source_function.sub(0)
        else:
            return self.source_function
