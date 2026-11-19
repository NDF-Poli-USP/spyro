"""Modal matrices of a mesh distributed over two ranks.

Run with ``mpiexec -n 2 pytest tests/parallel/test_modal_matrices_parallel.py``.
The degree-of-freedom numbering of a distributed mesh differs from the
serial one, so the parallel matrices are compared against a serial copy of
the same mesh through quantities that do not depend on the numbering.
"""
import numpy as np
import pytest
from firedrake import (
    COMM_SELF,
    COMM_WORLD,
    Function,
    FunctionSpace,
    RectangleMesh,
    SpatialCoordinate,
    assemble,
    dx,
    exp,
    grad,
    inner,
)

from spyro.solvers.modal.modal_forms_and_matrices import (
    assemble_sparse_matrices,
    maximum_diagonal_ratio,
    acoustic_weak_forms_c_in_stiffness,
)
from spyro.solvers.modal.modal_ana_sol import Modal_Analytical_Solver
from spyro.solvers.modal.modal_rq_matrices import (
    generate_eigenfunctions,
    matrices_rayleigh_quotient,
)
from spyro.solvers import wave as wave_module
from spyro.solvers.modal import modal_sol
from spyro.solvers.modal.modal_sol import Modal_Solver
from tests.on_one_core.test_modal_weak_forms import _elastic_wave

pytestmark = pytest.mark.modal


def _forms(comm) -> tuple:
    """Build the modal forms of a heterogeneous velocity on a small mesh.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Communicator the mesh is distributed over.

    Returns
    -------
    tuple
        Stiffness and mass forms ``(a, m)``.
    """
    mesh = RectangleMesh(8, 6, 1.0, 0.75, quadrilateral=True, comm=comm)
    V = FunctionSpace(mesh, "CG", 2)
    x, y = mesh.coordinates
    c = Function(V).interpolate(1.5 + x + 2.0 * y)
    return acoustic_weak_forms_c_in_stiffness(c, V)


def _invariants(matrix) -> tuple:
    """Return numbering-independent quantities of a sparse matrix.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Matrix to summarize.

    Returns
    -------
    tuple
        Shape, number of nonzeros, sorted diagonal, sorted row sums and
        Frobenius norm.
    """
    return (
        matrix.shape,
        matrix.nnz,
        np.sort(matrix.diagonal()),
        np.sort(np.asarray(matrix.sum(axis=1)).ravel()),
        np.sqrt((matrix.data**2).sum()),
    )


def _rayleigh_inputs(comm) -> tuple:
    """Build the inputs of the Rayleigh quotient matrices on a small mesh.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Communicator the mesh is distributed over.

    Returns
    -------
    tuple
        Velocity ``c`` and the eigenfunctions returned by
        `generate_eigenfunctions` with ``k = 2`` (16 eigenfunctions).
    """
    mesh = RectangleMesh(8, 6, 1.0, 0.75, quadrilateral=True, comm=comm)
    V = FunctionSpace(mesh, "CG", 2)
    coords = SpatialCoordinate(mesh)
    x, y = coords
    c = Function(V).interpolate(1.5 + x + 2.0 * y)
    limits = (np.array([0.0, 0.0]), np.array([1.0, 0.75]))
    eig_funcs = generate_eigenfunctions(coords, V, limits, k=2)
    return c, eig_funcs


def _fundamental_eigenvalue(method: str, comm) -> float:
    """Return the smallest nonzero eigenvalue of the modal forms.

    Parameters
    ----------
    method : str
        Eigensolver of `Modal_Solver`.
    comm : mpi4py.MPI.Comm
        Communicator the mesh is distributed over.

    Returns
    -------
    float
        Largest real part of the two eigenvalues closest to zero, the first
        of which is the rigid-body mode of the Neumann problem.
    """
    a, m = _forms(comm)
    V = a.arguments()[0].function_space()
    solver = Modal_Solver(2, method=method)
    eigenvalues = solver.solve_eigenproblem(None, V=V, shift=1e-8, forms=(a, m))
    return float(np.max(np.real(eigenvalues)))


@pytest.mark.parallel(2)
def test_rayleigh_matrices_match_serial_pairwise_integrals():
    """Rayleigh matrices of a distributed mesh equal the pairwise integrals.

    The entries are integrals, which do not depend on the numbering, so the
    matrices are compared entry by entry with the integrals of every pair of
    eigenfunctions assembled on a serial copy of the mesh. The reference
    builds the gradients of the eigenfunctions explicitly, as the pairwise
    implementation did, while `matrices_rayleigh_quotient` uses the
    stiffness matrix instead.
    """
    c, eig_funcs = _rayleigh_inputs(COMM_WORLD)
    Asp, Msp = matrices_rayleigh_quotient(c, eig_funcs)

    c_s, eig_s = _rayleigh_inputs(COMM_SELF)
    n_funcs = len(eig_s)
    grad_s = [grad(u_eig) for u_eig in eig_s]
    A_ref = np.array(
        [
            [
                assemble(c_s * c_s * inner(grad_s[i], grad_s[j]) * dx)
                for j in range(n_funcs)
            ]
            for i in range(n_funcs)
        ]
    )
    M_ref = np.array(
        [
            [assemble(inner(eig_s[i], eig_s[j]) * dx) for j in range(n_funcs)]
            for i in range(n_funcs)
        ]
    )
    for matrix, reference in ((Asp, A_ref), (Msp, M_ref)):
        scale = np.abs(reference).max()
        assert np.allclose(matrix.toarray(), reference, rtol=1e-10, atol=1e-12 * scale)

    # Every rank holds the same matrices
    gathered = COMM_WORLD.allgather(Asp.toarray())
    assert np.array_equal(gathered[0], gathered[1])


@pytest.mark.parallel(2)
def test_shift_invert_eigenvalue_matches_serial_scipy():
    """Distributed shift-and-invert SLEPc matches the serial SciPy solvers."""
    slepc = _fundamental_eigenvalue("KRYLOVSCH_CH", COMM_WORLD)
    assert slepc > 1e-3
    assert COMM_WORLD.allgather(slepc) == [slepc, slepc]
    for method, rtol in (("ARNOLDI", 1e-6), ("LANCZOS", 1e-6), ("LOBPCG", 1e-4)):
        scipy = _fundamental_eigenvalue(method, COMM_SELF)
        assert np.isclose(slepc, scipy, rtol=rtol), (method, slepc, scipy)


def _equivalent_velocity(comm) -> float:
    """Compute the energy-equivalent velocity of a heterogeneous model.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        Communicator the mesh is distributed over.

    Returns
    -------
    float
        Equivalent velocity for a load concentrated around (0.3, 0.4).
    """
    mesh = RectangleMesh(8, 6, 1.0, 0.75, quadrilateral=True, comm=comm)
    V = FunctionSpace(mesh, "CG", 2)
    x, y = SpatialCoordinate(mesh)
    c = Function(V).interpolate(1.5 + x + 2.0 * y)
    q = Function(V).interpolate(exp(-((x - 0.3) ** 2 + (y - 0.4) ** 2) / 0.02))
    return Modal_Analytical_Solver(dimension=2).c_equivalent(
        c, V, static_load_for_ceq=q
    )


@pytest.mark.parallel(2)
def test_equivalent_velocity_matches_serial():
    """The zero-mean static solve of c_equivalent matches a serial copy.

    The static problem is singular, with the constants as null space, and
    its load has its mean removed. Its solution is then unique up to a
    constant, so the equivalent velocity of a distributed mesh equals that
    of a serial copy, whatever the partition and the preconditioner.
    """
    c_eq = _equivalent_velocity(COMM_WORLD)
    c_eq_serial = _equivalent_velocity(COMM_SELF)
    assert 1.5 < c_eq < 4.0
    assert np.isclose(c_eq, c_eq_serial, rtol=1e-8)
    assert COMM_WORLD.allgather(c_eq) == [c_eq, c_eq]


@pytest.mark.parallel(2)
def test_modal_matrices_match_serial():
    """Gathered matrices and the diagonal ratio match a serial assembly."""
    assert COMM_WORLD.size == 2
    a, m = _forms(COMM_WORLD)
    # COMM_SELF holds only the calling rank, so each rank builds an
    # undistributed copy of the mesh as the serial reference
    a_serial, m_serial = _forms(COMM_SELF)

    Asp, Msp = assemble_sparse_matrices(a, m)
    Asp_serial, Msp_serial = assemble_sparse_matrices(a_serial, m_serial)
    for parallel, serial in ((Asp, Asp_serial), (Msp, Msp_serial)):
        shape, nnz, diagonal, row_sums, norm = _invariants(parallel)
        shape_s, nnz_s, diagonal_s, row_sums_s, norm_s = _invariants(serial)
        assert shape == shape_s
        assert nnz == nnz_s
        assert np.allclose(diagonal, diagonal_s)
        assert np.allclose(row_sums, row_sums_s)
        assert np.isclose(norm, norm_s)

    # Every rank holds the whole, identical matrix
    gathered = COMM_WORLD.allgather(Asp.diagonal())
    assert np.array_equal(gathered[0], gathered[1])

    _, Msp_inv = assemble_sparse_matrices(a, m, return_M_inv=True)
    expected = np.abs(Msp_inv.multiply(Asp).diagonal()).max()
    ratio = maximum_diagonal_ratio(a, m)
    assert np.isclose(ratio, expected)
    assert np.isclose(ratio, maximum_diagonal_ratio(a_serial, m_serial))
    assert COMM_WORLD.allgather(ratio) == [ratio, ratio]


@pytest.mark.parallel(2)
@pytest.mark.parametrize("method", ["ANALYTICAL", "ARNOLDI", "KRYLOVSCH_CH"])
def test_estimated_timestep_is_identical_on_every_rank(method: str):
    """Every rank gets the same time step, so all take the same steps."""
    a, m = _forms(COMM_WORLD)
    V = a.arguments()[0].function_space()
    solver = Modal_Solver(2, method=method, calc_max_dt=True)
    dt = solver.estimate_timestep(None, V, 1.0, forms=(a, m))
    assert dt > 0.0
    assert COMM_WORLD.allgather(dt) == [dt, dt]


@pytest.mark.parallel(2)
def test_slepc_timestep_matches_serial_scipy():
    """The distributed SLEPc estimate matches SciPy on a serial copy."""
    a, m = _forms(COMM_WORLD)
    a_serial, m_serial = _forms(COMM_SELF)
    V = a.arguments()[0].function_space()
    V_serial = a_serial.arguments()[0].function_space()

    slepc_solver = Modal_Solver(2, method="KRYLOVSCH_CH", calc_max_dt=True)
    scipy_solver = Modal_Solver(2, method="ARNOLDI", calc_max_dt=True)
    slepc_dt = slepc_solver.estimate_timestep(None, V, 1.0, forms=(a, m))
    scipy_dt = scipy_solver.estimate_timestep(
        None, V_serial, 1.0, forms=(a_serial, m_serial)
    )
    assert np.isclose(slepc_dt, scipy_dt, rtol=1e-6)


@pytest.mark.parallel(2)
def test_wave_uses_slepc_on_a_distributed_mesh(monkeypatch):
    """A wave on a distributed mesh estimates its time step with SLEPc."""
    methods = []

    def recording_modal_solver(*args, method=None, **kwargs):
        methods.append(method)
        return Modal_Solver(*args, method=method, **kwargs)

    monkeypatch.setattr(wave_module, "Modal_Solver", recording_modal_solver)
    wave = _elastic_wave(1.5, 1.0)
    dt = wave.get_and_set_maximum_dt(fraction=1.0)
    assert methods == ["KRYLOVSCH_CH"]
    assert COMM_WORLD.allgather(dt) == [dt, dt]

    # The distributed vector-valued problem matches SciPy's solution
    scipy_solver = Modal_Solver(2, method="ARNOLDI", calc_max_dt=True)
    scipy_dt = scipy_solver.estimate_timestep(
        None, wave.function_space, wave.final_time, fraction=1.0,
        forms=wave.modal_weak_forms(),
    )
    assert np.isclose(dt, scipy_dt, rtol=1e-6)


@pytest.mark.parallel(2)
def test_scipy_eigensolve_runs_on_one_rank(monkeypatch):
    """SciPy solves the gathered problem once and broadcasts the result."""
    calls = []
    original_eigs = modal_sol.eigs

    def counting_eigs(*args, **kwargs):
        calls.append(1)
        return original_eigs(*args, **kwargs)

    monkeypatch.setattr(modal_sol, "eigs", counting_eigs)
    # Building the wave sets up the ensemble the solve is restricted by
    wave = _elastic_wave(1.5, 1.0)
    solver = Modal_Solver(2, method="ARNOLDI", calc_max_dt=True)
    dt = solver.estimate_timestep(
        None, wave.function_space, 1.0, forms=wave.modal_weak_forms()
    )
    assert COMM_WORLD.allgather(len(calls)) == [1, 0]
    assert COMM_WORLD.allgather(dt) == [dt, dt]
