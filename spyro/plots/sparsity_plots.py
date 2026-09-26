import os
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import firedrake as fire
from firedrake import TrialFunction, TestFunction, dot, div, dx
from firedrake.petsc import PETSc


def _to_scipy(form, tol=1e-14):
    """Monta a forma como matriz esparsa (sem densificar)."""
    mat = fire.assemble(form, mat_type="aij").petscmat
    indptr, indices, data = mat.getValuesCSR()
    A = sp.csr_matrix((data, indices, indptr), shape=mat.getSize())
    A.data[np.abs(A.data) < tol] = 0.0
    A.eliminate_zeros()
    return A


def _style(ax, title):
    ax.set_title(title, fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def _matrices_pressure(wave):
    Vf, Vs = wave.fluid_function_space, wave.solid_function_space
    n_s, ds_i = wave.n_s, wave.ds_int(wave.interface_id)
    p_t, p_v = TrialFunction(Vf), TestFunction(Vf)
    u_t, u_v = TrialFunction(Vs), TestFunction(Vs)

    M_f = _to_scipy(p_t * p_v * dx(**wave.quadrature_rule_fluid))
    M_s = _to_scipy(dot(u_t, u_v) * dx(**wave.quadrature_rule_solid))
    C_f = _to_scipy(dot(u_t, n_s) * p_v * ds_i)   # lado fluido
    C_s = _to_scipy(p_t * dot(u_v, n_s) * ds_i)   # lado sólido
    labels = ("M_p (fluido)", "M_u (sólido)",
              "C_pu (lado fluido)", "C_up (lado sólido)")
    return (M_f, M_s, C_f, C_s), labels


def _matrices_displacement(wave):
    Vf, Vs = wave.fluid_function_space, wave.solid_function_space
    n_f, ds_i = wave.n_f, wave.ds_int(wave.interface_id)
    dim = wave.dimension
    uf_t, uf_v = TrialFunction(Vf), TestFunction(Vf)
    us_t, us_v = TrialFunction(Vs), TestFunction(Vs)

    M_f = _to_scipy(dot(uf_t, uf_v) * dx(**wave.quadrature_rule_fluid))
    M_s = _to_scipy(dot(us_t, us_v) * dx(**wave.quadrature_rule_solid))

    # Lado fluido: u_f.n = u_s.n imposto forte (cópia nodal da componente normal)
    rows = np.asarray(wave._iface_f_nodes) * dim + 1  # componente 1 = x (z,x)
    cols = np.asarray(wave._iface_s_nodes) * dim + 1
    C_f = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(Vf.dim(), Vs.dim())
    )
    # Lado sólido: tração kappa div(u_f) n
    C_s = _to_scipy(wave.K * div(uf_t) * dot(us_v, n_f) * ds_i)
    labels = ("M_uf (fluido)", "M_us (sólido)",
              "T_fs (lado fluido, u_f·n = u_s·n)", "C_su (lado sólido, κ∇·u_f)")
    return (M_f, M_s, C_f, C_s), labels


def plot_sparsity_diagnostic(wave, outdir="results"):
    """Esparsidade das matrizes de massa e acoplamento, para o caso em execução."""
    if wave.fluid_is_vector:
        mats, labels = _matrices_displacement(wave)
        tag, case = "displacement", "u_f – u_s"
    else:
        mats, labels = _matrices_pressure(wave)
        tag, case = "pressure", "P – u_s"

    print(f"=== Diagnóstico de esparsidade: {case} ===")
    print(f"Dofs fluido: {wave.fluid_function_space.dim()}")
    print(f"Dofs sólido: {wave.solid_function_space.dim()}")
    for A, lab in zip(mats, labels):
        pct = 100.0 * A.nnz / (A.shape[0] * A.shape[1])
        print(f"{lab}: {A.shape[0]}x{A.shape[1]}, nnz = {A.nnz} ({pct:.2e}%)")

    if PETSc.COMM_WORLD.rank != 0:
        return
    os.makedirs(outdir, exist_ok=True)

    # --- Figura 1: padrão de esparsidade ---
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    for ax, A, lab in zip(axes, mats, labels):
        ms = 4 if max(A.shape) < 5000 else 0.3
        ax.spy(A, markersize=ms)
        _style(ax, f"{lab}\n{A.shape[0]}x{A.shape[1]}")
    fig.suptitle(f"Esparsidade — formulação {case}", fontsize=22)
    plt.tight_layout()
    f1 = os.path.join(outdir, f"sparsity_{tag}.png")
    plt.savefig(f1, dpi=150)
    plt.close(fig)
    print(f"Salvo: {f1}")

    # --- Figura 2: magnitude dos blocos de acoplamento (só linhas/colunas não nulas) ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    for ax, A, lab in zip(axes2, mats[2:], labels[2:]):
        r = np.unique(A.nonzero()[0])
        c = np.unique(A.nonzero()[1])
        block = np.abs(A[r][:, c].toarray())
        im = ax.imshow(block, cmap="viridis", aspect="auto")
        _style(ax, f"|{lab}|\nbloco da interface {block.shape[0]}x{block.shape[1]}")
        cb = plt.colorbar(im, ax=ax)
        cb.set_label("valor absoluto", fontsize=14)
        cb.ax.tick_params(labelsize=12)
    fig2.suptitle(f"Acoplamento na interface — formulação {case}", fontsize=22)
    plt.tight_layout()
    f2 = os.path.join(outdir, f"coupling_magnitude_{tag}.png")
    plt.savefig(f2, dpi=150)
    plt.close(fig2)
    print(f"Salvo: {f2}")
