"""Generate the inputs for the Rayleigh Quotient method."""

from firedrake import assemble, cos, dx as fire_dx, Function, grad, inner, pi, sin
from scipy.sparse import lil_matrix
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
        Number of eigenvalues to compute. Default is 2.
    bc : `str`, optional
        Boundary condition type: "Dirichlet" or "Neumann". Default is "Neumann".
    dimension : `int`, optional
       Model dimension (2D or 3D). Default is 2D.

    Returns
    -------
    eig_funcs : `list`
        Eigenfunctions computed as `Firedrake.Function`
    grad_eig : `list`
        Eigenfunction gradients computed as `Firedrake.Function`
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

    # Precompute cosine values for efficiency
    if bc == "Neumann":
        fi_lst = [cos(i * pi * xn) for i in range(n_eigfunc)]
        fj_lst = [cos(j * pi * zn) for j in range(n_eigfunc)]

    if bc == "Dirichlet":
        fi_lst = [sin(i * pi * xn) for i in range(n_eigfunc)]
        fj_lst = [sin(j * pi * zn) for j in range(n_eigfunc)]

    if dimension == 3:  # 3D
        y = ufl_coordinates[2]
        y_min = min_coordinates[2]
        length_y = abs(max_coordinates[2] - min_coordinates[2])
        yn = (y - y_min) / length_y
        if bc == "Neumann":
            fk_lst = [cos(k * pi * yn) for k in range(n_eigfunc)]
        if bc == "Dirichlet":
            fk_lst = [sin(k * pi * yn) for k in range(n_eigfunc)]

    # Create eigenfunctions
    if dimension == 2:  # 2D
        # Eigenfunction: cos/sin(iπx/Lx) * cos/sin(jπz/Lz)
        products = [fi * fj for fi in fi_lst for fj in fj_lst]

    if dimension == 3:  # 3D
        # Eigenfunction: cos/sin(iπx/Lx) * cos/sin(jπz/Lz) * cos/sin(kπy/Ly)
        products = [fi * fj * fk for fi in fi_lst for fj in fj_lst for fk in fk_lst]

    eig_funcs = [Function(V).interpolate(prod) for prod in products]
    grad_eig = [grad(u_eig) for u_eig in eig_funcs]

    return eig_funcs, grad_eig


def matrices_rayleigh_quotient(c, eig_funcs, grad_eig, quad_rule=None):
    """Assemble the sparce matrices for the Rayleigh Quotient method.

    Parameters
    ----------
    c : `Firedrake.Function` or `float`
        Velocity model or isotropic velocity
    eig_funcs : `list`
        Eigenfunctions computed as `Firedrake.Function`.
    grad_eig : `list`
        Eigenfunction gradients computed as `Firedrake.Function`.
    quad_rule : `dict`, optional
        Quadrature rule to use for the integration.
        Default is `None`, which uses the default quadrature rule.

    Returns
    -------
    Asp : `csr matrix`
        Sparse matrix representing the stiffness matrix.
    Msp : `csr matrix`
        Sparse matrix representing the mass matrix.
    """

    # Initialize matrices for generalized eigenvalue problem
    n_funcs = len(eig_funcs)
    Asp = lil_matrix((n_funcs, n_funcs))  # Stiffness matrix
    Msp = lil_matrix((n_funcs, n_funcs))  # Mass matrix

    # Assemble stiffness and mass matrices
    dx = fire_dx(**quad_rule) if quad_rule else fire_dx

    for i in range(n_funcs):
        for j in range(i, n_funcs):  # Only upper triangle
            # Stiffness and mass matrix term
            A_term = assemble(c * c * inner(grad_eig[i], grad_eig[j]) * dx)
            M_term = assemble(inner(eig_funcs[i], eig_funcs[j]) * dx)

            # Set symmetric entries
            Asp[i, j] = A_term
            Asp[j, i] = A_term
            Msp[i, j] = M_term
            Msp[j, i] = M_term

    # Convert to CSR format for eigenvalue solver
    Asp = Asp.tocsr()
    Msp = Msp.tocsr()

    return Asp, Msp
