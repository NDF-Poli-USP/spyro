import firedrake as fire
import numpy as np

from .wave import Wave
from .acoustic_elastic_solver_no_pml import construct_acoustic_elastic
from .acoustic_elastic_solver_monolithic import construct_acoustic_elastic_monolithic
from .acoustic_solver_construction_no_pml import velocity_fluid
from ..utils.physical_parameters import PhysicalParameters
from ..utils.typing import (
    override,
    AdjointType,
    ElasticMaterialParameter,
    FluidMaterialParameter,
    RieszMapType,
    WaveType,
)
from ..domains.space import create_function_space
from ..domains.quadrature import quadrature_rules
from ..receivers.Receivers import Receivers
from ..io.segy_io import read_segy_velocity_model


def _extract_interface_markers(parent_mesh, child_mesh):
    parent_exterior = {int(m) for m in parent_mesh.exterior_facets.unique_markers}
    child_exterior = {int(m) for m in child_mesh.exterior_facets.unique_markers}
    return tuple(sorted(child_exterior - parent_exterior))


class AcousticElasticWave(Wave):
    """Acoplamento fluido-sólido (formulação P - u_s)."""

    #: The physical parameters are those of both media: the fluid ones are
    #: fields on the fluid submesh and the elastic ones on the solid submesh.
    _physical_parameter_names = frozenset(FluidMaterialParameter) | frozenset(
        ElasticMaterialParameter
    )

    def __init__(self, dictionary, comm=None):
        self.fluid_id = 1
        self.solid_id = 2
        self.interface_z = dictionary["mesh"].get("interface_z", None)
        self.use_monolithic = False
        self.record_fluid_displacement = False
        self.fluid_displacement_history = []
        self._vf_normal = None
        self._uf_normal = None

        super().__init__(dictionary, comm=comm)
        self.wave_type = WaveType.NONE

        self.field_logger.add_field(
            "displacement", "SolidDisplacement", lambda: self.X_n.sub(1)
        )

        self.p_equivalent_space = fire.FunctionSpace(
            self.submesh_solid, "CG", self.degree
        )
        self.p_equivalent_function = fire.Function(
            self.p_equivalent_space, name="EquivalentPressure"
        )
        self.field_logger.add_field(
            "p_equivalent", "EquivalentPressure", self._compute_p_equivalent
        )

        self.sigma_zz_space = fire.FunctionSpace(self.submesh_solid, "CG", self.degree)
        self.sigma_zz_function = fire.Function(self.sigma_zz_space, name="SigmaZZ")
        self.field_logger.add_field("sigma_zz", "SigmaZZ", self._compute_sigma_zz)

        self.K = None
        self.rho_fluid = None
        self.c = None
        self.rho = None
        self.c_p = None
        self.c_s = None
        self.lmbda = None
        self.mu = None
        self.body_forces = None

        self._setup_snapshots(dictionary)

    # ------------------------------------------------------------------
    # Malha, submalhas e medidas
    # ------------------------------------------------------------------
    def _cell_centers(self):
        V = fire.VectorFunctionSpace(self.mesh, "DG", 0)
        centers = fire.Function(V).interpolate(fire.SpatialCoordinate(self.mesh))
        return centers.dat.data_ro_with_halos

    def _fluid_cells(self):
        mesh_dict = self.input_dictionary["mesh"]
        centers = self._cell_centers()
        z, x = centers[:, 0], centers[:, 1]

        if mesh_dict.get("mesh_type") == "firedrake_mesh":
            if self.interface_z is None:
                raise ValueError(
                    "dictionary['mesh']['interface_z'] must be set for "
                    "mesh_type='firedrake_mesh'."
                )
            return z >= self.interface_z

        vp, nz, nx = read_segy_velocity_model(mesh_dict["segy_velocity_model"])
        lz, lx = mesh_dict["length_z"], mesh_dict["length_x"]
        iz = np.clip(np.rint((z + lz) / lz * (nz - 1)).astype(int), 0, nz - 1)
        ix = np.clip(np.rint(x / lx * (nx - 1)).astype(int), 0, nx - 1)
        return np.isclose(vp[iz, ix], mesh_dict["water_search_value"], rtol=1e-5)

    def _has_region_tags(self):
        dm = self.mesh.topology_dm
        ids = set()
        if dm.hasLabel("Cell Sets"):
            ids = {int(i) for i in dm.getLabelIdIS("Cell Sets").indices}
        ids = set().union(*self.mesh.comm.allgather(ids))
        return {self.fluid_id, self.solid_id} <= ids

    def _mark_mesh_regions(self):
        is_fluid = self._fluid_cells()
        dq0 = fire.FunctionSpace(self.mesh, "DG", 0)
        for region_id, mask in ((self.fluid_id, is_fluid), (self.solid_id, ~is_fluid)):
            indicator = fire.Function(dq0)
            indicator.dat.data_with_halos[:] = mask
            self.mesh.mark_entities(indicator, region_id, label_name=self.region_label)

    def _build_submeshes(self):
        dim = self.dimension
        self.submesh_fluid = fire.Submesh(
            self.mesh, dim, self.fluid_id, label_name=self.region_label
        )
        self.submesh_solid = fire.Submesh(
            self.mesh, dim, self.solid_id, label_name=self.region_label
        )

        for sub in (self.submesh_fluid, self.submesh_solid):
            z = sub.coordinates.dat.data_with_halos[:, 0]
            if z.size and z.max() > 1e-12:
                z *= -1.0

        iface_fluid = _extract_interface_markers(self.mesh, self.submesh_fluid)
        iface_solid = _extract_interface_markers(self.mesh, self.submesh_solid)
        assert (
            iface_fluid == iface_solid
        ), f"Inconsistent interface markers: {iface_fluid} vs {iface_solid}"
        self.interface_id = iface_solid[0] if len(iface_solid) == 1 else iface_solid

    def _build_measures(self):
        self.dx_fluid = fire.Measure("dx", domain=self.submesh_fluid)
        self.dx_solid = fire.Measure("dx", domain=self.submesh_solid)
        self.ds_int = fire.Measure(
            "ds",
            domain=self.submesh_fluid,
            intersect_measures=(fire.Measure("ds", self.submesh_solid),),
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
        if self.input_dictionary["mesh"].get("mesh_file") or self._has_region_tags():
            self.region_label = None
        else:
            self.region_label = "fluid_solid_regions"
            self._mark_mesh_regions()

        self._build_submeshes()
        self._build_measures()

        self.fluid_function_space = create_function_space(
            self.submesh_fluid, self.method, self.degree, dim=1
        )
        self.solid_function_space = create_function_space(
            self.submesh_solid, self.method, self.degree, dim=self.dimension
        )
        self.scalar_function_space = self.fluid_function_space
        self.vector_function_space = self.solid_function_space
        return self.fluid_function_space * self.solid_function_space

    @override
    def _build_function_space(self):
        self.function_space = self._create_function_space()
        self._setup_quadrature_rules()

    def _setup_quadrature_rules(self):
        (
            self.quadrature_rule_fluid,
            self.stiffness_quadrature_rule_fluid,
            self.surface_quadrature_rule_fluid,
        ) = quadrature_rules(self.fluid_function_space)
        for qr in (
            self.quadrature_rule_fluid,
            self.stiffness_quadrature_rule_fluid,
            self.surface_quadrature_rule_fluid,
        ):
            qr["domain"] = self.submesh_fluid

        (
            self.quadrature_rule_solid,
            self.stiffness_quadrature_rule_solid,
            self.surface_quadrature_rule_solid,
        ) = quadrature_rules(self.solid_function_space)
        for qr in (
            self.quadrature_rule_solid,
            self.stiffness_quadrature_rule_solid,
            self.surface_quadrature_rule_solid,
        ):
            qr["domain"] = self.submesh_solid

    def _setup_solid_receivers(self):
        solid_locs = self.input_dictionary["acquisition"].get(
            "solid_receiver_locations"
        )
        self.solid_receiver_history = []
        if not solid_locs:
            self.solid_receivers = None
            return

        saved_locs, saved_n = self.receiver_locations, self.number_of_receivers
        self.receiver_locations = solid_locs
        self.number_of_receivers = len(solid_locs)
        self.delta_projector_sub_index = 1
        self.solid_receivers = Receivers(self)
        self.delta_projector_sub_index = 0
        self.receiver_locations = saved_locs
        self.number_of_receivers = saved_n

    @override
    def _map_sources_and_receivers(self):
        """Map the sources and the fluid receivers onto the fluid field.

        Both sit in the fluid and act on the scalar pressure, so they are
        handled as acoustic ones.

        Returns
        -------
        None
            ``self.sources`` and ``self.receivers`` are set in place.
        """
        super()._map_sources_and_receivers()
        if self.sources is not None:
            self.sources.wave_type = WaveType.ISOTROPIC_ACOUSTIC
        self.receivers.wave_type = WaveType.ISOTROPIC_ACOUSTIC

    # =====BEGIN TEMPORARY=====
    @override
    def building_mesh_derived_paramenters(self):
        coodinates = self.mesh_ops._set_spatial_coordinates(self.mesh)
        self.mesh_z, self.mesh_x = coodinates[0], coodinates[1]
        if self.dimension == 3:
            self.mesh_y = coodinates[2]
        self._build_function_space()
        self._setup_solid_receivers()
        self._map_sources_and_receivers()
        self.mesh_ops.func_space_type = "mixed"
        self.mesh_parameters.boundary_idx_map = {}

    # ======END TEMPORARY======

    # ------------------------------------------------------------------
    # Parâmetros e montagem
    # ------------------------------------------------------------------
    @override
    def _initialize_model_parameters(self):
        """Initialize the fluid and solid material parameters.

        Every material parameter is a field on the submesh of its medium and
        is registered as a physical parameter of the equation, so any of
        them can be selected as a control of the automated adjoint.

        - Fluid, on the fluid submesh: the density, which the interface
          condition needs, with either the wave speed or the bulk modulus,
          from which the wave speed is then computed. The ones the model
          declares are independent ``Function`` objects.
        - Solid, on the solid submesh: density and the P- and S-wave
          velocities are independent ``Function`` objects, and the Lame
          parameters are UFL expressions of them.

        This runs more than once: :meth:`enable_automated_adjoint` and the
        forward solve both call it. The model is read only on the first call,
        so the fields the assembled forms and the adjoint refer to are not
        replaced afterwards.

        Returns
        -------
        None
            The material parameters are assigned on ``self``.

        Raises
        ------
        ValueError
            If the model does not give the fluid density, or gives an
            ambiguous or incomplete set of fluid parameters.
        """
        if self._physical_parameters:
            return

        synthetic_data = self.input_dictionary.get("synthetic_data", {})
        fluid_space = create_function_space(
            self.submesh_fluid, self.method, self.degree, dim=1
        )
        solid_space = create_function_space(
            self.submesh_solid, self.method, self.degree, dim=1
        )

        def field(space, parameter):
            """Return the field of an optional fluid parameter.

            Parameters
            ----------
            space : firedrake.functionspaceimpl.WithGeometry
                Scalar space on the fluid submesh.
            parameter : FluidMaterialParameter
                Parameter to read, keyed in the model by its value.

            Returns
            -------
            firedrake.Function or None
                The parameter as a field, or ``None`` if the model does not
                declare it.
            """
            value = synthetic_data.get(parameter.value)
            if value is None:
                return None
            return fire.Function(space, name=parameter.value).assign(value)

        self.K = field(fluid_space, FluidMaterialParameter.BULK_MODULUS)
        self.rho_fluid = field(fluid_space, FluidMaterialParameter.DENSITY)
        self.c = field(fluid_space, FluidMaterialParameter.P_WAVE_VELOCITY)
        if self.rho_fluid is None:
            raise ValueError(
                "The coupled fluid-solid wave needs 'density_fluid': the "
                "fluid density enters the interface condition."
            )

        def solid_field(parameter, key):
            """Return the field of a solid parameter.

            Parameters
            ----------
            parameter : ElasticMaterialParameter
                Parameter the field carries, which also names it.
            key : str
                Model key the value is read from.

            Returns
            -------
            firedrake.Function
                The parameter as a field on the solid submesh.
            """
            solid = fire.Function(solid_space, name=parameter.value)
            return solid.assign(synthetic_data[key])

        self.rho = solid_field(ElasticMaterialParameter.DENSITY, "density_solid")
        self.c_p = solid_field(
            ElasticMaterialParameter.P_WAVE_VELOCITY, "p_wave_velocity"
        )
        self.c_s = solid_field(
            ElasticMaterialParameter.S_WAVE_VELOCITY, "s_wave_velocity"
        )
        self.mu = self.rho * self.c_s**2
        self.lmbda = self.rho * self.c_p**2 - 2.0 * self.mu

        add = self._physical_parameters.add
        add(FluidMaterialParameter.DENSITY, self.rho_fluid)
        # The bulk modulus is registered only when the model gives it in
        # place of the velocity.
        if self.K is not None:
            add(FluidMaterialParameter.BULK_MODULUS, self.K)
        # Also validates the combination of fluid parameters the model gives.
        add(FluidMaterialParameter.P_WAVE_VELOCITY, velocity_fluid(self))
        add(ElasticMaterialParameter.DENSITY, self.rho)
        add(ElasticMaterialParameter.LAMBDA, self.lmbda)
        add(ElasticMaterialParameter.MU, self.mu)
        add(ElasticMaterialParameter.P_WAVE_VELOCITY, self.c_p)
        add(ElasticMaterialParameter.S_WAVE_VELOCITY, self.c_s)

    @override
    def matrix_building(self):
        self.current_time = 0.0
        self.X_nm1 = fire.Function(self.function_space)
        self.X_n = fire.Function(self.function_space)
        self.X_np1 = fire.Function(self.function_space)

        if self.use_monolithic:
            construct_acoustic_elastic_monolithic(self)
        else:
            construct_acoustic_elastic(self)

        self.gradP_space = fire.VectorFunctionSpace(
            self.submesh_fluid, "CG", self.degree
        )
        self.gradP_function = fire.Function(self.gradP_space, name="gradP")
        n_recv = (
            len(self.receiver_locations) if self.receiver_locations is not None else 0
        )
        self._vf_normal = [0.0] * n_recv
        self._uf_normal = [0.0] * n_recv

    # ------------------------------------------------------------------
    # Estados e interface com Wave
    # ------------------------------------------------------------------
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
    def get_function(self):
        return self.X_n.sub(0)

    @override
    def get_function_name(self):
        return "AcousticElastic"

    @override
    def rhs_no_pml(self):
        return self.rhs

    @override
    def rhs_no_pml_source(self):
        return self.source_function

    @override
    def update_source_expression(self, t):
        pass

    @override
    def get_control_parameters(self):
        raise NotImplementedError

    @override
    def set_control_parameters(self, controls):
        raise NotImplementedError

    @override
    def gradient_solve(
        self,
        misfit=None,
        forward_solution=None,
        adjoint_type: AdjointType = AdjointType.AUTOMATED_ADJOINT,
        riesz_map: RieszMapType = RieszMapType.L2,
    ) -> PhysicalParameters:
        """Compute automated-adjoint derivatives of the material parameters.

        The controls are the physical parameters selected through
        :meth:`enable_automated_adjoint`, fields on the fluid or the solid
        submesh. ``misfit`` and ``forward_solution`` belong to the
        hand-implemented adjoint, which does not exist for the coupled
        problem; they are kept so the signature matches the other solvers.

        Parameters
        ----------
        misfit : array_like, optional
            Unused by the automated adjoint.
        forward_solution : firedrake.Function, optional
            Unused by the automated adjoint.
        adjoint_type : AdjointType, optional
            Must be :attr:`AdjointType.AUTOMATED_ADJOINT`.
        riesz_map : RieszMapType, optional
            ``L2`` returns primal gradients and ``l2`` raw derivatives.

        Returns
        -------
        PhysicalParameters
            Derivatives keyed by the selected physical parameters.

        Raises
        ------
        NotImplementedError
            If a hand-implemented adjoint is requested.
        """
        if adjoint_type is not AdjointType.AUTOMATED_ADJOINT:
            raise NotImplementedError(
                "The coupled acoustic-elastic solver only supports the "
                "automated adjoint.",
            )
        return self._automated_adjoint_derivatives(riesz_map)

    @override
    def get_control_parameter_function_space(self):
        raise NotImplementedError

    def _setup_snapshots(self, dictionary):
        vis = dictionary.get("visualization", {})
        self._snapshot_every = vis.get("snapshot_frequency", None)
        self._snapshot_dir = vis.get("snapshot_output_dir", "results/snapshots")
        self._snapshot_step = 0

    # ------------------------------------------------------------------
    # Passo de tempo e receptores
    # ------------------------------------------------------------------
    def solve(self):
        if self.solid_receivers is not None:
            self.solid_receiver_history.append(self._sample_solid_receivers())

        if self.record_fluid_displacement and self.receiver_locations is not None:
            self._record_fluid_displacement_from_pressure()

        if self.use_monolithic:
            self._monolithic_solver.solve()
        else:
            self.source_function_fluid.assign(self.source_function.sub(0))
            self.solid_solver.solve()
            self.fluid_solver.solve()

    @override
    def receiver_field(self):
        """Return the pressure, the field the fluid receivers sample.

        The fluid subfunction is used rather than ``split(self.X_n)[0]``:
        the latter is defined on the mixed fluid-solid mesh, and Firedrake
        cannot interpolate it onto a vertex-only mesh of the fluid submesh.
        The adjoint tape re-reads a subfunction from its parent every time
        it is used, so it follows the state as it is updated in time.

        Returns
        -------
        firedrake.Function
            Fluid component of the coupled state.
        """
        return self.X_n.sub(0)

    @override
    def get_forward_solution_receivers(self):
        data = self.X_n.sub(0).dat.data_ro_with_halos[:]
        values = []
        for rid in range(self.receivers.number_of_points):
            if self.receivers.is_local[rid] is None or data.shape[0] == 0:
                values.append(-99999.0)
            else:
                values.append(self.receivers.new_at(data, rid))
        return values

    def _sample_solid_receivers(self):
        data = self.X_n.sub(1).dat.data_ro_with_halos[:]
        values = []
        for rid in range(self.solid_receivers.number_of_points):
            if self.solid_receivers.is_local[rid] is None or data.shape[0] == 0:
                values.append(np.full(self.dimension, -99999.0))
            else:
                values.append(self.solid_receivers.new_at(data, rid))
        return values

    def _record_fluid_displacement_from_pressure(self):
        self.gradP_function.interpolate(fire.grad(self.X_np1.sub(0)))
        dt = self.dt
        step_uf = []
        for i, loc in enumerate(self.receiver_locations):
            af_z = -self.gradP_function.at(loc)[0] / self.rho_fluid.at(loc)
            self._vf_normal[i] += dt * af_z
            self._uf_normal[i] += dt * self._vf_normal[i]
            step_uf.append(self._uf_normal[i])
        self.fluid_displacement_history.append(step_uf)

    # ------------------------------------------------------------------
    # Campos derivados
    # ------------------------------------------------------------------
    def _compute_p_equivalent(self):
        dim = self.dimension
        if dim == 2:
            K = self.lmbda + self.mu
        elif dim == 3:
            K = self.lmbda + (2.0 / 3.0) * self.mu
        else:
            raise ValueError(f"Unsupported dimension: {dim}")

        u = self.X_n.sub(1)
        self.p_equivalent_function.interpolate(-K * fire.div(u))
        return self.p_equivalent_function

    def _compute_sigma_zz(self):
        u = self.X_n.sub(1)
        strain = fire.sym(fire.grad(u))
        sigma_zz_expr = self.lmbda * fire.div(u) + 2.0 * self.mu * strain[0, 0]
        self.sigma_zz_function.interpolate(sigma_zz_expr)
        return self.sigma_zz_function
