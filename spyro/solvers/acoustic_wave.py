import firedrake as fire
import warnings
import os
from mpi4py import MPI

from .wave import Wave
from .automatic_differentiation_solver import AutomatedAdjoint

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

    @ensemble_gradient
    def gradient_solve(self, misfit=None, **kwargs):
        """Solves the adjoint problem to calculate de gradient.

        Returns:
        --------
        dJ: Firedrake 'Function'
            Gradient of the cost functional.
        """
        forward_solution = kwargs.pop("forward_solution", None)
        automated_adjoint = kwargs.pop("automated_adjoint", None)
        source_nums = kwargs.pop("source_nums", None)
        true_recv = kwargs.pop("true_recv", None)
        if forward_solution is not None:
            self.forward_solution = forward_solution
        self._validate_solve_kwargs(kwargs, "gradient_solve")
        if self.automatic_adjoint:
            self._evaluate_automatic_functional(
                true_recv=true_recv,
                automated_adjoint=automated_adjoint,
                source_nums=source_nums,
                **kwargs,
            )
            return self.automated_adjoint.compute_gradient()

        else:
            if misfit is not None:
                self.misfit = misfit
            elif self.current_time == 0.0:
                self.forward_solve(**kwargs)
                self.misfit = self.real_shot_record - self.receivers_data
            else:
                raise ValueError("Please load or calculate a real shot record first")
            return backward_wave_propagator(self)

    def _automatic_model_control(self, c=None):
        control = self.c
        if control is None:
            control = self.initial_velocity_model
        if control is None:
            raise ValueError("No velocity model is available for AD control.")
        if c is None:
            return control

        if isinstance(c, fire.Function):
            control.assign(c)
        else:
            control.dat.data[:] = c
        return control

    def _uses_serial_shot_source_control(self):
        return (
            self.use_vertex_only_mesh
            and self.parallelism_type == "spatial"
            and self.number_of_sources > 1
            and self.sources is not None
        )

    def _automatic_control_values(self, source_nums=None, c=None):
        model_control = self._automatic_model_control(c=c)
        if not self._uses_serial_shot_source_control():
            return model_control
        source_control = self.update_source_control(source_nums=source_nums)
        return [model_control, source_control]

    def _get_or_create_automated_adjoint(
        self,
        control_values,
        automated_adjoint=None,
    ):
        if automated_adjoint is None:
            automated_adjoint = self.automated_adjoint

        if isinstance(control_values, list):
            num_controls = len(control_values)
            model_control = control_values[0]
        else:
            num_controls = 1
            model_control = control_values

        if (
            automated_adjoint is None
            or automated_adjoint.control is not model_control
            or len(automated_adjoint.controls) != num_controls
            or automated_adjoint.ensemble is not self.comm
        ):
            automated_adjoint = AutomatedAdjoint(
                control_values,
                ensemble=self.comm,
            )

        self.automated_adjoint = automated_adjoint
        return automated_adjoint

    def _evaluate_automatic_functional(
        self,
        true_recv,
        automated_adjoint=None,
        source_nums=None,
        c=None,
        reduce_output=False,
        **kwargs,
    ):
        control_values = self._automatic_control_values(
            source_nums=source_nums,
            c=c,
        )
        automated_adjoint = self._get_or_create_automated_adjoint(
            control_values,
            automated_adjoint=automated_adjoint,
        )

        previous_compute_functional = self.compute_functional
        self.compute_functional = True
        try:
            if automated_adjoint.reduced_functional is None:
                with automated_adjoint.fresh_tape():
                    automated_adjoint.start_recording()
                    try:
                        solve_kwargs = dict(kwargs, true_recv=true_recv)
                        if source_nums is not None:
                            solve_kwargs["source_nums"] = source_nums
                        self.forward_solve(**solve_kwargs)
                    finally:
                        automated_adjoint.stop_recording()
                    automated_adjoint.create_reduced_functional(self.functional)
                    if reduce_output and self.comm.ensemble_comm.size > 1:
                        self.functional = (
                            self.comm.ensemble_comm.allreduce(
                                float(self.functional),
                                op=MPI.SUM,
                            )
                            * automated_adjoint.reduction_scale
                        )
            else:
                self.update_true_receiver_data(true_recv)
                if self._uses_serial_shot_source_control():
                    self.update_source_control(source_nums=source_nums)
                self.functional = automated_adjoint.recompute_functional(
                    control_values
                )
        finally:
            self.compute_functional = previous_compute_functional

        return self.functional

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
