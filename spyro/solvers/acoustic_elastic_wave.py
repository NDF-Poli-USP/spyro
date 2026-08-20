import firedrake as fire
import warnings
import os

from .wave import Wave
from .acoustic_elastic_solver_no_pml import construct_acoustic_elastic # to implement
from ..utils.typing import override, WaveType
from ..domains.space import create_function_space
from ..domains.quadrature import quadrature_rules
from ..plots.general_plots import plot acoustic_elastic_snapshot # to implement

class AcousticElasticWave(Wave):
    def __init__(self, dictionary, comm=None):
        self.fluid_id = dictionary["mesh"].get("fluid_id", 1)
        self.solid_id  = dictionary["mesh"].get("solid_id", 2)
        self.interface_x = dictionary["mesh"].get("interface_x", None)

        super().__init__(dictionary, comm=comm)
        self.wave_type = WaveType.ISOTROPIC_ACOUSTIC_ELASTIC # to implement
        self.field_logger.add_field("displacement", "SolidDisplacement", lambda: self.X_n.sub(1))

        self.c           = None # fluid
        self.rho         = None # solid
        self.lmbda       = None # solid
        self.mu          = None # solid
        self.body_forces = None # solid

        self._save_pressure_only = True
        self._use.mixed_source   = True

        self._setup_snapshots(dictionary)

    def _mark_mesh_regions(self):
        if self.interface_x in None:
            raise ValueError(
                "dictionary['mesh']['interface_x'] must be set when "
                "using an automatically-generated mesh (mesh_filen=None) "
                "with AcousticElasticWave."
            )
        dq0 = fire.FunctionSpace(self.mesh, "DG", 0)

        indicator_fluid = fire.Function(dq0).interpolate(
            fire.conditional(self.mesh_x >= self.interface_x, 1, 0)
        )
        self.mesh_mark_entities(indicator_fluid, self.fluid_id)
        indicator_solid = fire.Function(dq0).interpolate(
            fire.conditional(self.mesh_x < self.interface_x, 1, 0)
        )
        self.mesh.mark_entities(indicator_solid, self.solid_id)

    def _build_submeshes(self):
        dim = self.dimension
        self.submesh_fluid = fire.Submesh(self.mesh, dim, self.fluid_id)
        self.submesh_solid = fire.Submesh(self.mesh, dim, self.solid_id)

        # Restore the lost negative z-sign in the Submesh
        self.submesh_fluid.coordinates.dat.data[:, 0] *= -1.0
        self.submesh_solid.coordinates.dat.data[:, 0] *= -1.0

    def _build_measures(self):
        self.dx_fluid = fire.Measure("dx", domain=self.submesh_fluid)
        self.dx_solid = fire.Measure("dx", domain=self.submesh_solid)
        self.
    
    @override
    def _create_function_space(self):
        is_automatic_mesh = self.input_dictionary["mesh"].get("mesh_file") is None
        if is_automatic_mesh:
            self._mark_mesh_regions()
        else:
            # TODO: mesh-file case not implemented yet.
            pass

        self._build_submeshes()
        self._build_interface_measures()

        self.scalar_function_space = create_function_space(
            self.submesh_fluid, self.method, self.degree, dim=1
        )
        self.vector_function_space = create_function_space(
            self.submesh_solid, self.method, self.degree, dim=self.dimension
        )
        mixed_space = self.scalar_function_space * self.vector_function_space
        return mixed_space

    # =====BEGIN TEMPORARY=====
    @override
    def building_mesh_derived_parameters(self):
        coodinates = self.mesh_ops._set_spatial_coordinates(self.mesh)
        self.mesh_z, self.mesh_x = coordinates[0], coordinates[1]
        if self.dimension ==3:
            self.mesh_y = coordinates[2]
        self._build_function_space()
        self._map_sources_and_receivers()
        self.mesh_ops.func_space_type = 'mixed'
        self.mesh_parameters.boundary_idx_map = {}
    # ======END TEMPORARY======

    @override
    def _initialize_model_parameters(self):
        synthetic_data = self.input_dictionary.get("synthetic_data", {})

        velocity_fluid_value  = synthetic_data["velocity_fluid"]  # fluid
        rho_value             = synthetic_data["density_solid"]   # solid
        p_wave_velocity_value = synthetic_data["p_wave_velocity"] # solid
        s_wave_velocity_value = synthetic_data["s_wave_velocity"] # solid

        self.c      = fire.Function(self.scalar_function_space)
        self.c.interpolate(fire.Constant(velocity_fluid_value))
        self.rho    = fire.Constant(rho_value)
        mu_value    = rho_value * s_wave_velocity_value**2
        lmbda_value = rho_valuev * p_wave_velocity_value**2 - 2.0 * mu_value
        self.mu     = fire.Constant(mu_value)
        self.lmda   = fire.Constant(lmbda_value)

    @override
    def matrix_building(self):
        self.current_time = 0.0
        self.X_nm1        = fire.Function(self.function_space)
        self.X_n          = fire.Function(self.function_space)
        self.X_np1        = fire.Function(self.function_space)
        construct_acoustic_elastic(self)

    @override
    def _get_vstate(self):
        return self.X_n

    @override
    def _set_vstate(self):
        self.X_n.assign(vstate)

    @override
    def _get_prev_vstate(self):
        return self.X_nm1

    @override
    def _set_prev_vstate(self, vstate):
        self.X_nm1.assign(vstate)

    @override
    def _get_next_vstate(self):
        return self.X_np1

    @override
    def _set_next_vstate(self, vstate):
        self.X_np1.assign(vstate)

    @override
    def get_forward_solution_receivers(self):
        data_with_halos = self.X_n.sub(0).dat.data_ro_with_halos[:]
        return self.receivers.interpolate(data_with_halos)

    @override
    def get_functions(self):
        return self.X_n.sub(0)

    @override
    def get_function_name(self):
        return "AcousticElastic" # temporary name

    @override
    def rhs_no_pml(self):
        return self.rhs

    @override
    def rhs_no_pml_source(self):
        return self.source_function
        
    @override
    def _build_function_space(self):
        self.function_space = self._create_function_space()
        self._setup_quadrature_rules()

    def _setup_quadrature_rules(self):
        self.quadrature_rule_fluid, \
            self.stiffness_quadrature_rule_fluid, \
            self.surface_quadrature_rule_fluid = quadrature_rules(self.scalar_function_space)
        for qr in (self.quadrature_rule_fluid, self.stiffness_quadrature_rule_fluid,
                   self.surface_quadrature_rule_fluid):
            qr["domain"] = self.submesh_fluid
        
        self.quadrature_rule_solid, \
            self.stiffness_quadrature_rule_solid, \
            self.surface_quadrature_rule_solid = quadrature_rules(self.vector_function_space)
        for qr in (self.quadrature_rule_solid, self.stiffness_quadrature_rule_solid,
                   self.surface_quadrature_rule_solid):
            qr["domain"] = self.submesh_solid

    @override
    def update_source_expression(self, t):
        self._handle_snapshot()

    def _handle_snapshot(self):
        if self._snapshot_every is not None and self._snapshot_step % self._snapshot_every == 0:
            os.makedirs(self._snapshot_dir, exist_ok=True)
            plot_acoustic_elastic_snapshot(
                self, filename=f"{self._snapshot_dir}/snapshot_{self._snapshot_step:04d}.png"
            )
        self._snapshot_step += 1
        
        
        











            