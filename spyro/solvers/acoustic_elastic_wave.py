import firedrake as fire

from .wave import Wave
from .acoustic_elastic_solver_no_pml import construct_acoustic_elastic
from .acoustic_elastic_solver_monolithic import construct_acoustic_elastic_monolithic
from .acoustic_elastic_solver_uu import construct_displacement_displacement
from ..utils.typing import override, WaveType
from ..domains.space import create_function_space
from ..domains.quadrature import quadrature_rules
from ..receivers.Receivers import Receivers

import numpy as np

FLUID_FORMULATIONS = ("pressure", "displacement")


def _extract_interface_markers(parent_mesh, child_mesh):
    parent_exterior = {int(m) for m in parent_mesh.exterior_facets.unique_markers}
    child_exterior = {int(m) for m in child_mesh.exterior_facets.unique_markers}
    return tuple(sorted(child_exterior - parent_exterior))


class AcousticElasticWave(Wave):
    """Acoplamento fluido-sólido.

    fluid_formulation (dictionary["options"]):
      "pressure"     -> P-us (fluido escalar, pressão)           [padrão]
      "displacement" -> u_f-u_s (fluido vetorial, deslocamento)
    """

    def __init__(self, dictionary, comm=None):
        self.fluid_id = 1
        self.solid_id = 2
        self.interface_x = dictionary["mesh"].get("interface_x", None)
        self.sigma_xx_history = []
        self.fluid_displacement_history = []
        self._vf_normal = None
        self._uf_normal = None
        self.fluid_pressure_history = []
        self.record_fluid_pressure = False
        self._uu_step = 0
        self.record_fluid_displacement = False        

        self.use_monolithic = False

        # Precisa existir ANTES do super().__init__ (que monta o espaço).
        options = dictionary.get("options", {})
        self.fluid_formulation = options.get("fluid_formulation", "pressure")
        if self.fluid_formulation not in FLUID_FORMULATIONS:
            raise ValueError(
                f"fluid_formulation inválida: {self.fluid_formulation!r}. "
                f"Use uma de {FLUID_FORMULATIONS}."
            )

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

        self.sigma_xx_space = fire.FunctionSpace(self.submesh_solid, "CG", self.degree)
        self.sigma_xx_function = fire.Function(self.sigma_xx_space, name="SigmaXX")
        self.field_logger.add_field("sigma_xx", "SigmaXX", self._compute_sigma_xx)

        if self.fluid_is_vector:
            self.fluid_pressure_space = fire.FunctionSpace(
                self.submesh_fluid, "CG", self.degree
            )
            self.fluid_pressure_function = fire.Function(
                self.fluid_pressure_space, name="FluidPressure"
            )
            self.field_logger.add_field(
                "fluid_pressure", "FluidPressure", self._compute_fluid_pressure
            )

        self.K = None
        self.rho_fluid = None
        self.c = None  # fluid
        self.rho = None  # solid
        self.lmbda = None  # solid
        self.mu = None  # solid
        self.body_forces = None  # solid

        self._setup_snapshots(dictionary)

    # ------------------------------------------------------------------
    @property
    def fluid_is_vector(self):
        return self.fluid_formulation == "displacement"

    # ------------------------------------------------------------------
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
        is_automatic_mesh = self.input_dictionary["mesh"].get("mesh_file") is None
        if is_automatic_mesh:
            self._mark_mesh_regions()
        else:
            # TODO: mesh-file case not implemented yet.
            pass

        self._build_submeshes()
        self._build_measures()

        fluid_dim = self.dimension if self.fluid_is_vector else 1
        self.fluid_function_space = create_function_space(
            self.submesh_fluid, self.method, self.degree, dim=fluid_dim
        )
        self.solid_function_space = create_function_space(
            self.submesh_solid, self.method, self.degree, dim=self.dimension
        )
        # compatibilidade com solvers/plots antigos
        self.scalar_function_space = self.fluid_function_space
        self.vector_function_space = self.solid_function_space
        return self.fluid_function_space * self.solid_function_space

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
        if self.fluid_is_vector:
            self._fix_moment_source_tabulations()
        self.mesh_ops.func_space_type = "mixed"
        self.mesh_parameters.boundary_idx_map = {}

    # ======END TEMPORARY======

    @override
    def _initialize_model_parameters(self):
        synthetic_data = self.input_dictionary.get("synthetic_data", {})

        K_value = synthetic_data.get("bulk_modulus")  # fluid
        rho_fluid_value = synthetic_data.get("density_fluid")  # fluid
        velocity_fluid_value = synthetic_data.get("velocity_fluid")  # fluid
        rho_value = synthetic_data["density_solid"]  # solid
        p_wave_velocity_value = synthetic_data["p_wave_velocity"]  # solid
        s_wave_velocity_value = synthetic_data["s_wave_velocity"]  # solid

        self.K = fire.Constant(K_value) if K_value is not None else None
        self.rho_fluid = (
            fire.Constant(rho_fluid_value) if rho_fluid_value is not None else None
        )
        self.c = (
            fire.Constant(velocity_fluid_value)
            if velocity_fluid_value is not None
            else None
        )
        self.rho = fire.Constant(rho_value)
        mu_value = rho_value * s_wave_velocity_value**2
        lmbda_value = rho_value * p_wave_velocity_value**2 - 2.0 * mu_value
        self.mu = fire.Constant(mu_value)
        self.lmbda = fire.Constant(lmbda_value)

        if self.fluid_is_vector:
            # u_f-u_s precisa de rho_f e kappa explicitamente.
            if self.rho_fluid is None:
                raise ValueError(
                    "synthetic_data['density_fluid'] é obrigatório com "
                    "fluid_formulation='displacement'."
                )
            if self.K is None:
                if self.c is None:
                    raise ValueError(
                        "Informe 'bulk_modulus' ou 'velocity_fluid' para o fluido."
                    )
                self.K = fire.Constant(rho_fluid_value * velocity_fluid_value**2)

    @override
    @override
    def matrix_building(self):
        self.current_time = 0.0
        self.X_nm1 = fire.Function(self.function_space)
        self.X_n = fire.Function(self.function_space)
        self.X_np1 = fire.Function(self.function_space)

        if self.fluid_is_vector:
            if self.use_monolithic:
                raise NotImplementedError(
                    "Monolítico ainda não implementado para u_f-u_s."
                )
            construct_displacement_displacement(self)

            # Fonte explosiva (amplitude = I): tempo = integral dupla do wavelet
            # do próprio Spyro (Cao et al., eqs. 3-4). A flag evita integrar
            # de novo se forward_solve for chamado mais de uma vez.
            if not getattr(self, "_wavelet_integrated", False):
                w = np.asarray(self.sources.wavelet)
                from scipy.integrate import cumulative_trapezoid
                w1 = cumulative_trapezoid(w, dx=self.dt, initial=0.0)
                self.sources.wavelet = cumulative_trapezoid(w1, dx=self.dt, initial=0.0)
                self._wavelet_integrated = True

            self._uu_step = 0

        elif self.use_monolithic:
            construct_acoustic_elastic_monolithic(self)
        else:
            construct_acoustic_elastic(self)

        if not self.fluid_is_vector:
            self.gradP_space = fire.VectorFunctionSpace(
                self.submesh_fluid, "CG", self.degree
            )
            self.gradP_function = fire.Function(self.gradP_space, name="gradP")
            n_recv = len(self.receiver_locations) if self.receiver_locations else 0
            self._vf_normal = [0.0] * n_recv
            self._uf_normal = [0.0] * n_recv

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
        # pressure: P nos receptores | displacement: u_f (vetor) nos receptores
        data_with_halos = self.X_n.sub(0).dat.data_ro_with_halos[:]
        return self.receivers.interpolate(data_with_halos)

    @override
    def get_function(self):
        return self.X_n.sub(0)

    @override
    def get_function_name(self):
        return "AcousticElastic"  # temporary name

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

    # ------------------------------------------------------------------
    def solve(self):
        # Registro em t = step*dt (estado ANTES do passo), alinhado com o loop do Spyro
        if self.solid_receivers is not None:
            data = self.X_n.sub(1).dat.data_ro_with_halos[:]
            self.solid_receiver_history.append(self.solid_receivers.interpolate(data))

        if (self.fluid_is_vector and self.record_fluid_pressure
                and self.receiver_locations):
            self.fluid_pressure_function.interpolate(
                -self.K * fire.div(self.X_n.sub(0)))
            self.fluid_pressure_history.append(
                [self.fluid_pressure_function.at(loc)
                 for loc in self.receiver_locations])

        if (not self.fluid_is_vector and self.record_fluid_displacement
                and self.receiver_locations):
            self._record_fluid_displacement_from_pressure()

        # Passo de tempo
        if self.use_monolithic:
            self._monolithic_solver.solve()
        else:
            self.source_function_fluid.assign(self.source_function.sub(0))
            self.solid_solver.solve()
            self.fluid_solver.solve()
            if self.fluid_is_vector:
                self._impose_interface_normal_continuity()


    def _record_fluid_displacement_from_pressure(self):
        # Só P-us: integra a_f = -grad(P)/rho_f duas vezes no tempo.
        self.gradP_function.interpolate(fire.grad(self.X_np1.sub(0)))
        dt = self.dt
        rho_f_val = float(self.rho_fluid)
        step_uf = []
        for i, loc in enumerate(self.receiver_locations):
            gx = self.gradP_function.at(loc)[1]  # índice 1 = componente x (z,x)
            af_x = -gx / rho_f_val
            self._vf_normal[i] += dt * af_x
            self._uf_normal[i] += dt * self._vf_normal[i]
            step_uf.append(self._uf_normal[i])
        self.fluid_displacement_history.append(step_uf)

    # ------------------------------------------------------------------
    def _compute_fluid_pressure(self):
        # Só u_f-u_s: p = -kappa * div(u_f)
        u_f = self.X_n.sub(0)
        self.fluid_pressure_function.interpolate(-self.K * fire.div(u_f))
        return self.fluid_pressure_function

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

    def _compute_sigma_xx(self):
        u = self.X_n.sub(1)
        strain = fire.sym(fire.grad(u))
        sigma_xx_expr = self.lmbda * fire.div(u) + 2.0 * self.mu * strain[1, 1]
        self.sigma_xx_function.interpolate(sigma_xx_expr)
        return self.sigma_xx_function

    def _impose_interface_normal_continuity(self):
        """Dof compartilhado na interface: soma das linhas do fluido e do
        sólido (média das previsões ponderada pelas massas lumped).
        Padrão: só a componente normal (deslizamento, eq. 10).
        Teste: synthetic_data["welded_interface_test"] = True compartilha
        todas as componentes (interface soldada)."""
        welded = self.input_dictionary["synthetic_data"].get(
            "welded_interface_test", False)
        comps = range(self.dimension) if welded else (self._iface_normal_comp,)

        jf, js = self._iface_f_nodes, self._iface_s_nodes
        mf, ms = self._m_f_iface, self._m_s_iface   # massa lumped: igual por componente
        uf = self.X_np1.sub(0).dat.data
        us = self.X_np1.sub(1).dat.data
        for c in comps:
            un = (mf * uf[jf, c] + ms * us[js, c]) / (mf + ms)
            uf[jf, c] = un
            us[js, c] = un

    def _fix_moment_source_tabulations(self, rel_eps=1e-6):
        """Converte as derivadas de referência da fonte de momento do Spyro
        (ordem 1) em derivadas físicas: dphi/dx_k = sum_j dphi/dxi_j dxi_j/dx_k.
        Não altera o Spyro; só corrige as tabulações já montadas."""
        from spyro.receivers.changing_coordinates import (
            change_to_reference_quad, change_to_reference_triangle,
        )
        src = self.sources
        tabs = np.asarray(src.cell_tabulations)
        if tabs.ndim != 3:          # não é fonte de momento (amplitude escalar/vetor)
            return

        if self.dimension != 2:
            raise NotImplementedError("Correção implementada só para 2D.")
        n_v, change = (4, change_to_reference_quad) if src.quadrilateral \
            else (3, change_to_reference_triangle)

        for i in range(src.number_of_points):
            if src.is_local[i] is None:
                continue
            p = np.asarray(src.point_locations[i], dtype=float)
            verts = src.cellVertices[i][0:n_v]
            eps = rel_eps * np.max(np.ptp(np.asarray(verts, dtype=float), axis=0))
            dxi_dx = np.zeros((2, 2))
            for k in range(2):
                e = np.zeros(2)
                e[k] = eps
                xi_p = np.asarray(change(tuple(p + e), verts), dtype=float)
                xi_m = np.asarray(change(tuple(p - e), verts), dtype=float)
                dxi_dx[:, k] = (xi_p - xi_m) / (2.0 * eps)
            tabs[i] = tabs[i] @ dxi_dx
        src.cell_tabulations = tabs