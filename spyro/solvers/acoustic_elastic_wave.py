import firedrake as fire
import warnings
import os

from .wave import Wave
from .acoustic_elastic_solver_no_pml import construct_acoustic_elastic
from ..utils.typing import override, WaveType
from ..domains.space import create_function_space
from ..domains.quadrature import quadrature_rules
from ..receivers.Receivers import Receivers
# from ..plots.general_plots import plot acoustic_elastic_snapshot # to implement

def _extract_interface_markers(parent_mesh, child_mesh):
    parent_exterior = {int(m) for m in parent_mesh.exterior_facets.unique_markers}
    child_exterior  = {int(m) for m in child_mesh.exterior_facets.unique_markers}
    return tuple(sorted(child_exterior - parent_exterior))

class AcousticElasticWave(Wave):
    def __init__(self, dictionary, comm=None):
        self.fluid_id    = 1
        self.solid_id    = 2
        self.interface_x = dictionary["mesh"].get("interface_x", None)

        super().__init__(dictionary, comm=comm)
        self.wave_type = WaveType.NONE
        self.field_logger.add_field("displacement", "SolidDisplacement", lambda: self.X_n.sub(1))

        self.p_equivalent_space = fire.FunctionSpace(self.submesh_solid, "CG", self.degree)
        self.p_equivalent_function = fire.Function(self.p_equivalent_space, name="EquivalentPressure")
        self.field_logger.add_field("p_equivalent", "EquivalentPressure", self._compute_p_equivalent)

        self.c           = None # fluid
        self.rho         = None # solid
        self.lmbda       = None # solid
        self.mu          = None # solid
        self.body_forces = None # solid

        self._setup_snapshots(dictionary)

    def _mark_mesh_regions(self):
        if self.interface_x is None:
            raise ValueError(
                "dictionary['mesh']['interface_x'] must be set when "
                "using an automatically-generated mesh (mesh_filen=None) "
                "with AcousticElasticWave."
            )
        dq0 = fire.FunctionSpace(self.mesh, "DG", 0)

        indicator_fluid = fire.Function(dq0).interpolate(
            fire.conditional(self.mesh_x >= self.interface_x, 1, 0)
        )
        self.mesh.mark_entities(indicator_fluid, self.fluid_id)
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

        iface_fluid = _extract_interface_markers(self.mesh, self.submesh_fluid)
        iface_solid = _extract_interface_markers(self.mesh, self.submesh_solid)
        assert iface_fluid == iface_solid, (
            f"Inconsistent interface markers: {iface_fluid} vs {iface_solid}"
        )
        self.interface_id = iface_solid[0] if len(iface_solid) == 1 else iface_solid

    def _build_measures(self):
        self.dx_fluid = fire.Measure("dx", domain=self.submesh_fluid)
        self.dx_solid = fire.Measure("dx", domain=self.submesh_solid)
        self.ds_int   = fire.Measure(
            "ds", domain=self.submesh_fluid,
            intersect_measures=(fire.Measure("ds", self.submesh_solid),)
        )
        self.n_f = fire.FacetNormal(self.submesh_fluid)
        self.n_s = fire.FacetNormal(self.submesh_solid)
        check = fire.assemble(
            fire.dot(self.n_s + self.n_f, self.n_s + self.n_f)
            * self.ds_int(self.interface_id)
        )
        assert check < 1e-12, f"Inconsistent interface normals: {check}"
    
    @override
    def _create_function_space(self):
        is_automatic_mesh = self.input_dictionary["mesh"].get("mesh_file") is None
        if is_automatic_mesh:
            self._mark_mesh_regions()
        else:
            # TODO: mesh-file case not implemented yet.
            pass

        self._build_submeshes()
        self._build_measures()

        self.scalar_function_space = create_function_space(
            self.submesh_fluid, self.method, self.degree, dim=1
        )
        self.vector_function_space = create_function_space(
            self.submesh_solid, self.method, self.degree, dim=self.dimension
        )
        mixed_space = self.scalar_function_space * self.vector_function_space
        return mixed_space

    def _setup_solid_receivers(self):
        solid_locs = self.input_dictionary["acquisition"].get("solid_receiver_locations")
        self.solid_receiver_history = []
        if not solid_locs:
            self.solid_receivers = None
            return

        saved_locs, saved_n = self.receiver_locations, self.number_of_receivers
        self.receiver_locations   = solid_locs
        self.number_of_receivers  = len(solid_locs)
        self.delta_projector_sub_index = 1
        self.solid_receivers = Receivers(self)
        self.delta_projector_sub_index = 0
        self.receiver_locations  = saved_locs
        self.number_of_receivers = saved_n

    # =====BEGIN TEMPORARY=====
    @override
    def building_mesh_derived_paramenters(self):
        coodinates = self.mesh_ops._set_spatial_coordinates(self.mesh)
        self.mesh_z, self.mesh_x = coodinates[0], coodinates[1]
        if self.dimension ==3:
            self.mesh_y = coodinates[2]
        self._build_function_space()
        self._setup_solid_receivers()
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

        self.c      = fire.Constant(velocity_fluid_value)
        self.rho    = fire.Constant(rho_value)
        mu_value    = rho_value * s_wave_velocity_value**2
        lmbda_value = rho_value * p_wave_velocity_value**2 - 2.0 * mu_value
        self.mu     = fire.Constant(mu_value)
        self.lmbda  = fire.Constant(lmbda_value)

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
    def _set_vstate(self, vstate):
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
    def get_function(self):
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
        # self._handle_snapshot()
        pass

    # def _handle_snapshot(self):
    #     if self._snapshot_every is not None and self._snapshot_step % self._snapshot_every == 0:
    #         os.makedirs(self._snapshot_dir, exist_ok=True)
    #         plot_acoustic_elastic_snapshot(
    #             self, filename=f"{self._snapshot_dir}/snapshot_{self._snapshot_step:04d}.png"
    #         )
    #     self._snapshot_step += 1

    @override
    def get_control_parameters(self):
        raise NotImplementedError

    @override
    def set_control_parameters(self, controls):
        raise NotImplementedError

    @override
    def gradient_solve(self, guess=None, misfit=None, forward_solution=None):
        raise NotImplementedError

    @override
    def get_control_parameter_function_space(self):
        raise NotImplementedError
        
    def _setup_snapshots(self, dictionary):
        vis = dictionary.get("visualization", {})
        self._snapshot_every = vis.get("snapshot_frequency", None)
        self._snapshot_dir = vis.get("snapshot_output_dir", "results/snapshots")
        self._snapshot_step = 0

    def solve(self):
        self.source_function_fluid.assign(self.source_function.sub(0))
        self.solid_solver.solve()
        self.fluid_solver.solve()

        if self.solid_receivers is not None:
            data = self.X_np1.sub(1).dat.data_ro_with_halos[:]
            self.solid_receiver_history.append(self.solid_receivers.interpolate(data))

    def _compute_p_equivalent(self):
        dim = self.dimension
        if dim == 2:
            K = self.lmbda + self.mu
        elif dim == 3:
            K = self.lmbda + (2.0/3.0) * self.mu
        else:
            raise ValueError(f"Unsupported dimension: {dim}")

        u = self.X_n.sub(1)
        self.p_equivalent_function.interpolate(-K * fire.div(u))
        return self.p_equivalent_function