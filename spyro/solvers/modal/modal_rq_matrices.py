"""Generate the inputs for the Rayleigh Quotient method."""

from firedrake import (
    assemble,
    Constant,
    cos,
    dx as fire_dx,
    Function,
    grad,
    inner,
    pi,
    sin,
    TestFunction,
    TrialFunction,
)
from scipy.sparse import csr_matrix, lil_matrix
from ...utils.error_management import (validate_data_structure, validate_numeric,
                                       validate_parameter)


def generate_eigenfunctions(ufl_coordinates, V, mesh_limits,
                            k=2, bc="Neumann", dimension=2):
    """Generate eigenfunctions for the Rayleigh Quotient method.

    Parameters
    ----------
    ufl_coordinates : `ufl.geometry.SpatialCoordinate`
        Domain coordinates.
    V : `Firedrake.FunctionSpace`
        Function space for the modal problem.
    mesh_limits : `tuple`, optional
        Tuple containing the minimum and maximum coordinates of the mesh.
        Structure: (min_coordinates, max_coordinates):
        - min_coordinates : `array`
            Array containing the minimum coordinates in each dimension (z, x, y).
        - max_coordinates : `array`
            Array containing the maximum coordinates in each dimension (z, x, y).
    k : `int`, optional
        Number of eigenvalues to compute. It sets the number of
        eigenfunctions per direction, n = max(2 k, 2). Default is 2.
    bc : `str`, optional
        Boundary condition type: "Dirichlet" or "Neumann". Default is "Neumann".
    dimension : `int`, optional
       Model dimension (2D or 3D). Default is 2D.

    Returns
    -------
    eig_funcs : `list`
        The n**dimension eigenfunctions, interpolated into `V` as
        `Firedrake.Function`.

    Notes
    -----
    Only the eigenfunctions are returned. Their gradients are not needed,
    because `matrices_rayleigh_quotient` obtains the stiffness entries from
    the stiffness matrix of `V`.
    """

    # Check input parameters
    validate_numeric("k", k, float_num=False, integer_num=True, lower_bound=0)
    validate_parameter("dimension", dimension, [2, 3])
    validate_parameter("bc", bc, ["Dirichlet", "Neumann"])

    # Mesh coordinates
    z, x = ufl_coordinates[0], ufl_coordinates[1]

    # Check mesh limits
    min_coordinates = validate_data_structure("min_coordinates", mesh_limits[0], "array",
                                              expected_type_element=("float", "int"),
                                              expected_length=dimension)
    max_coordinates = validate_data_structure("max_coordinates", mesh_limits[1], "array",
                                              expected_type_element=("float", "int"),
                                              expected_length=dimension)

    # Minimum coordinates
    z_min, x_min = min_coordinates[:2]

    # Domain dimensions
    length_z, length_x = abs(max_coordinates[:2] - min_coordinates[:2])

    # Number of eigenfunctions to use
    n_eigfunc = max(2 * k, 2)

    # Mesh normalized coordinates w.r.t. the minimum coordinates of the mesh
    zn = (z - z_min) / length_z
    xn = (x - x_min) / length_x

    # Mode numbers are Constants, with one set per direction, so that every
    # eigenfunction has the same UFL structure and all the interpolations
    # share a single compiled kernel instead of compiling one per product
    modes_x, modes_z, modes_y = (
        [Constant(float(n)) for n in range(n_eigfunc)] for _ in range(3)
    )

    # Precompute cosine values for efficiency
    if bc == "Neumann":
        fi_lst = [cos(i * pi * xn) for i in modes_x]
        fj_lst = [cos(j * pi * zn) for j in modes_z]

    if bc == "Dirichlet":
        fi_lst = [sin(i * pi * xn) for i in modes_x]
        fj_lst = [sin(j * pi * zn) for j in modes_z]

    if dimension == 3:  # 3D
        y = ufl_coordinates[2]
        y_min = min_coordinates[2]
        length_y = abs(max_coordinates[2] - min_coordinates[2])
        yn = (y - y_min) / length_y
        if bc == "Neumann":
            fk_lst = [cos(k * pi * yn) for k in modes_y]
        if bc == "Dirichlet":
            fk_lst = [sin(k * pi * yn) for k in modes_y]

    # Create eigenfunctions
    if dimension == 2:  # 2D
        # Eigenfunction: cos/sin(iπx/Lx) * cos/sin(jπz/Lz)
        products = [fi * fj for fi in fi_lst for fj in fj_lst]

    if dimension == 3:  # 3D
        # Eigenfunction: cos/sin(iπx/Lx) * cos/sin(jπz/Lz) * cos/sin(kπy/Ly)
        products = [fi * fj * fk for fi in fi_lst for fj in fj_lst for fk in fk_lst]

    eig_funcs = [Function(V).interpolate(prod) for prod in products]

    return eig_funcs


def matrices_rayleigh_quotient(
    c: Function | float, eig_funcs: list[Function], quad_rule: dict | None = None
) -> tuple[csr_matrix, csr_matrix]:
    """Assemble the sparce matrices for the Rayleigh Quotient method.

    Parameters
    ----------
    c : `Firedrake.Function` or `float`
        Velocity model or isotropic velocity
    eig_funcs : `list`
        Eigenfunctions computed as `Firedrake.Function`.
    quad_rule : `dict`, optional
        Quadrature rule to use for the integration.
        Default is `None`, which uses the default quadrature rule.

    Returns
    -------
    Asp : `csr matrix`
        Sparse matrix representing the stiffness matrix.
    Msp : `csr matrix`
        Sparse matrix representing the mass matrix.

    Notes
    -----
    The entries are the integrals of ``c**2 * inner(grad(phi_i),
    grad(phi_j))`` and ``phi_i * phi_j``. Every eigenfunction belongs to
    the function space, ``phi_i = sum_a Phi[a, i] N_a`` with basis functions
    ``N_a``, so ``grad(phi_i) = sum_a Phi[a, i] grad(N_a)`` and the entries
    equal ``phi_i^T K phi_j`` and ``phi_i^T M phi_j``. ``K`` and ``M`` are the
    stiffness and mass matrices assembled once with the same integrands and
    quadrature, so the gradients of the eigenfunctions are never needed on
    their own. That takes two assemblies instead of two per pair of
    eigenfunctions.
    """

    # Initialize matrices for generalized eigenvalue problem
    n_funcs = len(eig_funcs)
    Asp = lil_matrix((n_funcs, n_funcs))  # Stiffness matrix
    Msp = lil_matrix((n_funcs, n_funcs))  # Mass matrix

    # Assemble stiffness and mass matrices of the function space
    dx = fire_dx(**quad_rule) if quad_rule else fire_dx
    V = eig_funcs[0].function_space()
    u, v = TrialFunction(V), TestFunction(V)
    K = assemble(c * c * inner(grad(u), grad(v)) * dx).petscmat
    M = assemble(inner(u, v) * dx).petscmat

    # Eigenfunctions as PETSc vectors
    phi = []
    for u_eig in eig_funcs:
        with u_eig.dat.vec_ro as vec:
            phi.append(vec.copy())

    k_phi, m_phi = K.createVecLeft(), M.createVecLeft()
    for j in range(n_funcs):
        K.mult(phi[j], k_phi)
        M.mult(phi[j], m_phi)
        for i in range(j + 1):  # Only upper triangle
            # Stiffness and mass matrix term (global reductions in parallel)
            A_term = phi[i].dot(k_phi)
            M_term = phi[i].dot(m_phi)

            # Set symmetric entries
            Asp[i, j] = A_term
            Asp[j, i] = A_term
            Msp[i, j] = M_term
            Msp[j, i] = M_term

    # Convert to CSR format for eigenvalue solver
    Asp = Asp.tocsr()
    Msp = Msp.tocsr()

    return Asp, Msp
