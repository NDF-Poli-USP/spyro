import firedrake as fire

from firedrake import (Cofunction, LinearVariationalProblem,
                       LinearVariationalSolver, TestFunction,
                       TrialFunction, dot, lhs, rhs)
from spyro.solvers.acoustic_solver_construction_no_pml import build_acoustic_form
from spyro.solvers.elastic_wave.forms import build_elastic_form

def construct_acoustic_elastic(Wave_obj):
    p_nm1, u_nm1 = fire.split(Wave_obj.X_nm1)
    p_n, u_n = fire.split(Wave_obj.X_n)

    dt = Wave_obj.dt
    n_s = Wave_obj.n_s
    ds_int = Wave_obj.ds_int
    iface = Wave_obj.interface_id

    # Solid sub-problem:
    u_trial = TrialFunction(Wave_obj.vector_function_space)
    v_s = TestFunction(Wave_obj.vector_function_space)

    F_elastic = build_elastic_form(
        Wave_obj,
        u_trial,
        v_s,
        u_n,
        u_nm1,
        Wave_obj.quadrature_rule_solid
    )

    F_interface_elastic  = + p_n * dot(v_s, n_s) * ds_int(iface)
    F_solid = F_elastic + F_interface_elastic

    solid_lhs = fire.lhs(F_solid)
    solid_rhs = fire.rhs(F_solid)

    solid_problem = LinearVariationalProblem(
        solid_lhs,
        solid_rhs,
        Wave_obj.X_np1.sub(1),
        constant_jacobian=True,
    )

    solver_parameters = dict(Wave_obj.solver_parameters)
    solver_parameters = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
    }

    Wave_obj.solid_solver = LinearVariationalSolver(
        solid_problem, 
        solver_parameters=solver_parameters
    )

    # Fluid sub-problem:
    p_trial = TrialFunction(Wave_obj.scalar_function_space)
    v_f = TestFunction(Wave_obj.scalar_function_space)

    F_acoustic = build_acoustic_form(
        Wave_obj, p_trial, v_f, p_n, p_nm1,
        Wave_obj.quadrature_rule_fluid, c=Wave_obj.c, K=Wave_obj.K, rho_fluid=Wave_obj.rho_fluid
    )

    u_tt = (Wave_obj.X_np1.sub(1) - 2.0*u_n + u_nm1) / dt**2
    F_interface_acoustic = - dot(u_tt, n_s) * v_f * ds_int(iface)
    F_fluid = F_acoustic + F_interface_acoustic

    fluid_lhs = fire.lhs(F_fluid)
    fluid_rhs = fire.rhs(F_fluid)

    Wave_obj.source_function = Cofunction(Wave_obj.function_space.dual())
    Wave_obj.source_function_fluid = Cofunction(Wave_obj.scalar_function_space.dual())

    fluid_problem = LinearVariationalProblem(
        fluid_lhs,
        fluid_rhs + Wave_obj.source_function_fluid,
        Wave_obj.X_np1.sub(0),
        constant_jacobian=True,
    )

    solver_parameters = dict(Wave_obj.solver_parameters)
    solver_parameters = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
    }

    Wave_obj.fluid_solver = LinearVariationalSolver(
        fluid_problem, 
        solver_parameters=solver_parameters
    )

    Wave_obj.solver = Wave_obj

    # ===== DIAGNÓSTICO TEMPORÁRIO — remover depois =====
    import numpy as np
    import matplotlib.pyplot as plt
    from firedrake import assemble, dx

    Np = Wave_obj.scalar_function_space.dim()
    Nu = Wave_obj.vector_function_space.dim()
    print(f"Tamanho espaço fluido (Np): {Np}")
    print(f"Tamanho espaço sólido (Nu): {Nu}")

    p_t, p_v = TrialFunction(Wave_obj.scalar_function_space), TestFunction(Wave_obj.scalar_function_space)
    u_t, u_v = TrialFunction(Wave_obj.vector_function_space), TestFunction(Wave_obj.vector_function_space)

    M_p_mat = assemble(p_t * p_v * dx(**Wave_obj.quadrature_rule_fluid), mat_type="aij").petscmat
    M_u_mat = assemble(dot(u_t, u_v) * dx(**Wave_obj.quadrature_rule_solid), mat_type="aij").petscmat
    Cpu_mat = assemble(dot(u_t, n_s) * p_v * ds_int(iface), mat_type="aij").petscmat

    Cup_form = p_t * dot(u_v, n_s) * ds_int(iface)
    Cup_mat = assemble(Cup_form, mat_type="aij").petscmat
    print(f"C_up shape: {Cup_mat.getSize()}")
    Cup_dense = Cup_mat[:, :]

    print(f"M_p shape: {M_p_mat.getSize()}")
    print(f"M_u shape: {M_u_mat.getSize()}")
    print(f"C_pu shape: {Cpu_mat.getSize()}")

    M_p_dense = M_p_mat[:, :]
    M_u_dense = M_u_mat[:, :]
    Cpu_dense = Cpu_mat[:, :]

    nnz_p = np.count_nonzero(np.abs(M_p_dense) > 1e-14)
    nnz_c = np.count_nonzero(np.abs(Cpu_dense) > 1e-14)
    print(f"M_p: {nnz_p}/{M_p_dense.size} não-nulos ({100*nnz_p/M_p_dense.size:.2f}%)")
    print(f"C_pu: {nnz_c}/{Cpu_dense.size} não-nulos ({100*nnz_c/Cpu_dense.size:.2f}%)")

    from firedrake.petsc import PETSc
    rank = PETSc.COMM_WORLD.rank

    max_dim = max(Np, Nu)

    if rank == 0:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].spy(M_p_dense, markersize=6)
        axes[0].set_title(f"M_p (fluido)\n{M_p_dense.shape[0]}x{M_p_dense.shape[1]}")

        axes[1].spy(M_u_dense, markersize=6)
        axes[1].set_title(f"M_u (sólido)\n{M_u_dense.shape[0]}x{M_u_dense.shape[1]}")

        axes[2].spy(Cpu_dense, markersize=6)
        axes[2].set_title(f"C_pu (acoplamento)\n{Cpu_dense.shape[0]}x{Cpu_dense.shape[1]}")

        axes[3].spy(Cup_dense, markersize=6)
        axes[3].set_title(f"C_up (transposta)\n{Cup_dense.shape[0]}x{Cup_dense.shape[1]}")

        plt.tight_layout()
        plt.savefig("/workspaces/spyro/sparsity_comparison.png", dpi=150)
        print("Salvo em sparsity_comparison.png")

        import numpy as np

        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

        im0 = axes2[0].imshow(np.abs(Cpu_dense), cmap="viridis", aspect="auto")
        axes2[0].set_title(f"|C_pu| (magnitude)\n{Cpu_dense.shape[0]}x{Cpu_dense.shape[1]}")
        plt.colorbar(im0, ax=axes2[0], label="valor absoluto")

        im1 = axes2[1].imshow(np.abs(Cup_dense), cmap="viridis", aspect="auto")
        axes2[1].set_title(f"|C_up| (magnitude)\n{Cup_dense.shape[0]}x{Cup_dense.shape[1]}")
        plt.colorbar(im1, ax=axes2[1], label="valor absoluto")

        plt.tight_layout()
        plt.savefig("/workspaces/spyro/coupling_magnitude.png", dpi=150)
        print("Salvo em /workspaces/spyro/coupling_magnitude.png")
    # ===== FIM DIAGNÓSTICO =====