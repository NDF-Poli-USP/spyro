from contextlib import contextmanager

import numpy as np
import firedrake as fire
from firedrake import (
    Constant,
    LinearVariationalProblem,
    LinearVariationalSolver,
    TestFunction,
    TrialFunction,
    dot,
    dx,
)
from scipy.spatial import cKDTree

from spyro.solvers.elastic_wave.forms import build_elastic_form


@contextmanager
def _material(wave, rho, lmbda, mu):
    """Troca temporariamente (rho, lmbda, mu) do wave durante a montagem da
    forma (build_elastic_form lê esses atributos do wave)."""
    saved = (wave.rho, wave.lmbda, wave.mu)
    wave.rho, wave.lmbda, wave.mu = rho, lmbda, mu
    try:
        yield
    finally:
        wave.rho, wave.lmbda, wave.mu = saved


def _build_interface_node_map(wave, tol=1e-8):
    """Pareia os nós da interface do fluido e do sólido (malha conforme)."""
    Vf, Vs = wave.fluid_function_space, wave.solid_function_space
    f_nodes = fire.DirichletBC(Vf, 0.0, wave.interface_id).nodes
    s_nodes = fire.DirichletBC(Vs, 0.0, wave.interface_id).nodes

    xf = fire.Function(Vf).interpolate(fire.SpatialCoordinate(wave.submesh_fluid))
    xs = fire.Function(Vs).interpolate(fire.SpatialCoordinate(wave.submesh_solid))
    dist, idx = cKDTree(xs.dat.data_ro[s_nodes]).query(xf.dat.data_ro[f_nodes])
    if np.max(dist) > tol:
        raise RuntimeError(
            f"Nós da interface não coincidem (dist máx = {np.max(dist):.2e}). "
            "A malha precisa ser conforme e com o mesmo grau nos dois lados."
        )
    wave._iface_f_nodes = f_nodes
    wave._iface_s_nodes = s_nodes[idx]


def _lumped_mass_diagonal(rho, dt, V, quad_rule):
    """Diagonal da massa lumped (rho/dt^2) por dof, no formato (n_nós, dim).
    Como a massa GLL é diagonal, a soma de cada linha é a própria diagonal."""
    v = TestFunction(V)
    one = Constant(tuple([1.0] * V.value_size))
    m = fire.assemble((rho / dt**2) * dot(one, v) * dx(**quad_rule))
    return m.dat.data_ro.copy()


def construct_displacement_displacement(Wave_obj):
    """Formulação u_f-u_s (Cao et al., eq. 10) com a onda elástica do Spyro
    nos dois meios.

    Fluido: lambda_f = kappa - 2 mu_f, mu_f = rho_f * vs_f^2, com vs_f lido de
    synthetic_data["s_wave_velocity_fluid"] (padrão 0 -> fluido ideal, mu_f = 0).
    lambda_f + 2 mu_f = kappa mantém a velocidade P do fluido.

    Forma fraca consistente:
      - u_s.n = u_f.n é condição ESSENCIAL (entra no espaço): o deslocamento
        normal na interface é um grau de liberdade COMPARTILHADO.
      - A tração é NATURAL: com v_f.n = v_s.n, os termos de interface dos
        dois lados se cancelam. Não há integral de interface.
    Discretização (massa lumped, explícito): cada lado é resolvido sem termo
    de interface; no dof normal compartilhado, a equação é a SOMA das linhas
    do fluido e do sólido, isto é, a média das duas previsões ponderada pelas
    massas (feita em _impose_interface_normal_continuity, na classe).
    """
    uf_nm1, u_nm1 = fire.split(Wave_obj.X_nm1)
    uf_n, u_n = fire.split(Wave_obj.X_n)
    dt = Constant(Wave_obj.dt)

    # solver_parameters = {
    #     'ksp_type': 'preonly',
    #     'pc_type': 'lu',
    # }

    solver_parameters = {
        "ksp_type": "preonly",
        "pc_type": "jacobi",
        "mat_type": "matfree",
    }

    # ---- Sólido: onda elástica do Spyro (sem termo de interface) ----
    u_trial = TrialFunction(Wave_obj.solid_function_space)
    v_s = TestFunction(Wave_obj.solid_function_space)
    F_solid = build_elastic_form(
        Wave_obj, u_trial, v_s, u_n, u_nm1, Wave_obj.quadrature_rule_solid
    )
    Wave_obj.solid_solver = LinearVariationalSolver(
        LinearVariationalProblem(
            fire.lhs(F_solid), fire.rhs(F_solid), Wave_obj.X_np1.sub(1),
            constant_jacobian=True,
        ),
        solver_parameters=solver_parameters,
    )

    # ---- Fluido: onda elástica do Spyro; mu_f = rho_f * vs_f^2 (teste) ----
    vs_f = Wave_obj.input_dictionary["synthetic_data"].get("s_wave_velocity_fluid", 0.0)
    mu_f = Wave_obj.rho_fluid * vs_f**2
    lmbda_f = Wave_obj.K - 2.0 * mu_f

    uf_trial = TrialFunction(Wave_obj.fluid_function_space)
    v_f = TestFunction(Wave_obj.fluid_function_space)
    with _material(Wave_obj, Wave_obj.rho_fluid, lmbda_f, mu_f):
        F_fluid = build_elastic_form(
            Wave_obj, uf_trial, v_f, uf_n, uf_nm1, Wave_obj.quadrature_rule_fluid
        )

    def _rot2d(u):                        # convenção (z, x): du_x/dz - du_z/dx
        return u[1].dx(0) - u[0].dx(1)

    alpha_fac = Wave_obj.input_dictionary["synthetic_data"].get("rotation_penalty", 0.0)
    alpha = alpha_fac * Wave_obj.K
    F_fluid += alpha * _rot2d(uf_n) * _rot2d(v_f) * dx(**Wave_obj.quadrature_rule_fluid)

    Wave_obj.source_function = fire.Cofunction(Wave_obj.function_space.dual())
    Wave_obj.source_function_fluid = fire.Cofunction(
        Wave_obj.fluid_function_space.dual()
    )
    Wave_obj.fluid_solver = LinearVariationalSolver(
        LinearVariationalProblem(
            fire.lhs(F_fluid),
            fire.rhs(F_fluid) + Wave_obj.source_function_fluid,
            Wave_obj.X_np1.sub(0),
            constant_jacobian=True,
        ),
        solver_parameters=solver_parameters,
    )

    # ---- Dof normal compartilhado na interface ----
    _build_interface_node_map(Wave_obj)
    m_f = _lumped_mass_diagonal(
        Wave_obj.rho_fluid, dt, Wave_obj.fluid_function_space,
        Wave_obj.quadrature_rule_fluid,
    )
    m_s = _lumped_mass_diagonal(
        Wave_obj.rho, dt, Wave_obj.solid_function_space,
        Wave_obj.quadrature_rule_solid,
    )
    normal = 1  # componente x (convenção z, x): normal da interface x = cte
    Wave_obj._iface_normal_comp = normal
    Wave_obj._m_f_iface = m_f[Wave_obj._iface_f_nodes, normal]
    Wave_obj._m_s_iface = m_s[Wave_obj._iface_s_nodes, normal]

    Wave_obj.rhs = None
    Wave_obj.solver = Wave_obj