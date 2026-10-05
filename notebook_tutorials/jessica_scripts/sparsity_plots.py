"""Save the sparsity data of the mass and interface-coupling matrices.

The matrices are assembled as sparse (never densified) and saved to an .npz
file, so that all plotting can be done later in a single analysis script
(plot_all_results.py), without Firedrake objects.
"""
import numpy as np
import scipy.sparse as sp

import firedrake as fire
from firedrake import TrialFunction, TestFunction, dot, dx


def _to_scipy(form, tol=1e-14):
    """Assemble the form as a sparse matrix (never densified)."""
    mat = fire.assemble(form, mat_type="aij").petscmat
    indptr, indices, data = mat.getValuesCSR()
    A = sp.csr_matrix((data, indices, indptr), shape=mat.getSize())
    A.data[np.abs(A.data) < tol] = 0.0
    A.eliminate_zeros()
    return A


def _matrices_pressure(wave):
    """P-us: fluid/solid mass matrices and the two interface integrals."""
    Vf, Vs = wave.fluid_function_space, wave.solid_function_space
    n_s, ds_i = wave.n_s, wave.ds_int(wave.interface_id)
    p_t, p_v = TrialFunction(Vf), TestFunction(Vf)
    u_t, u_v = TrialFunction(Vs), TestFunction(Vs)

    mats = [
        _to_scipy(p_t * p_v * dx(**wave.quadrature_rule_fluid)),
        _to_scipy(dot(u_t, u_v) * dx(**wave.quadrature_rule_solid)),
        _to_scipy(dot(u_t, n_s) * p_v * ds_i),    # fluid side
        _to_scipy(p_t * dot(u_v, n_s) * ds_i),    # solid side
    ]
    labels = ["M_p (fluid)", "M_u (solid)", "C_pu (fluid side)", "C_up (solid side)"]
    rows = ["P dofs (fluid equation)", "u_s dofs (solid equation)",
            "P dofs (fluid equation)", "u_s dofs (solid equation)"]
    cols = ["P dofs (unknown)", "u_s dofs (unknown)",
            "u_s dofs (unknown)", "P dofs (unknown)"]
    return mats, labels, rows, cols


def _matrices_displacement(wave):
    """u_f-u_s: fluid/solid mass matrices and the shared normal dof map.
    There is no interface integral in this formulation: the coupling is the
    normal displacement dof shared by the paired interface nodes."""
    Vf, Vs = wave.fluid_function_space, wave.solid_function_space
    dim = wave.dimension
    uf_t, uf_v = TrialFunction(Vf), TestFunction(Vf)
    us_t, us_v = TrialFunction(Vs), TestFunction(Vs)

    c = wave._iface_normal_comp
    rows = np.asarray(wave._iface_f_nodes) * dim + c
    cols = np.asarray(wave._iface_s_nodes) * dim + c
    shared = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(Vf.dim(), Vs.dim())
    )
    mats = [
        _to_scipy(dot(uf_t, uf_v) * dx(**wave.quadrature_rule_fluid)),
        _to_scipy(dot(us_t, us_v) * dx(**wave.quadrature_rule_solid)),
        shared,
    ]
    labels = ["M_uf (fluid)", "M_us (solid)", "Shared normal dof (u_f·n = u_s·n)"]
    rows = ["u_f dofs (fluid equation)", "u_s dofs (solid equation)",
            "u_f dofs (fluid, interface)"]
    cols = ["u_f dofs (unknown)", "u_s dofs (unknown)",
            "u_s dofs (solid, interface)"]
    return mats, labels, rows, cols


def save_sparsity_matrices(wave, filename):
    """Assemble the matrices of the case being run and save them to filename.
    Row index = dof of the equation (test function);
    column index = dof of the unknown (trial function)."""
    split = -1  # block separator (monolithic only)
    if getattr(wave, "use_monolithic", False):
        # Monolithic: the single system matrix actually solved at each step
        mats = [_to_scipy(wave.lhs)]   # mesma matriz do diagnóstico antigo
        labels = ["Mass matrix"]
        rows = ["[ P | u_s ] dofs (equations)"]
        cols = ["[ P | u_s ] dofs (unknowns)"]
        case = "P – u_s (monolithic)"
        split = wave.fluid_function_space.dim()
    elif wave.fluid_is_vector:
        mats, labels, rows, cols = _matrices_displacement(wave)
        case = "u_f – u_s"
    else:
        mats, labels, rows, cols = _matrices_pressure(wave)
        case = "P – u_s"

    print(f"=== Sparsity data: {case} ===")
    print(f"Fluid dofs: {wave.fluid_function_space.dim()}")
    print(f"Solid dofs: {wave.solid_function_space.dim()}")
    payload = {"labels": np.array(labels), "case": np.array(case),
               "row_labels": np.array(rows), "col_labels": np.array(cols),
               "split": np.array(split)}
    for i, (A, lab) in enumerate(zip(mats, labels)):
        A = A.tocsr()
        pct = 100.0 * A.nnz / (A.shape[0] * A.shape[1])
        print(f"{lab}: {A.shape[0]}x{A.shape[1]}, nnz = {A.nnz} ({pct:.2e}%)")
        payload[f"m{i}_data"] = A.data
        payload[f"m{i}_indices"] = A.indices
        payload[f"m{i}_indptr"] = A.indptr
        payload[f"m{i}_shape"] = np.array(A.shape)
    np.savez(filename, **payload)
    print(f"[OK] Saved: {filename}")


def load_sparsity_matrices(filename):
    """Inverse of save_sparsity_matrices.
    Returns (case, labels, matrices, row_labels, col_labels, split)."""
    d = np.load(filename)
    labels = [str(x) for x in d["labels"]]
    rows = [str(x) for x in d["row_labels"]] if "row_labels" in d else ["Row index"] * len(labels)
    cols = [str(x) for x in d["col_labels"]] if "col_labels" in d else ["Column index"] * len(labels)
    split = int(d["split"]) if "split" in d else -1
    mats = [
        sp.csr_matrix(
            (d[f"m{i}_data"], d[f"m{i}_indices"], d[f"m{i}_indptr"]),
            shape=tuple(d[f"m{i}_shape"]),
        )
        for i in range(len(labels))
    ]
    return str(d["case"]), labels, mats, rows, cols, split