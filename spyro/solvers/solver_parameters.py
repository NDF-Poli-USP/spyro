"""Sets up solver parameters for PETSc.

Returns None if you are using a method that does not have
a diagonal mass matrix as a solver.

For diagonal mass matrice-based solvers we use two options:
- "ksp_type": "preonly"
- "pc_type": "jacobi"

With preonly ksp_type PETSc does not iterate, and PETSc does not
create a krylov space. The rpeconditioner becomes the solver.

By using a jacoby preconditioner we extract the diagonal of the matrix
and invert it elementwise with point division.

Since we use diagonal mass-matrices Jacobi isn't an approximation and
we get a direct solve at the computational cost of vector division O(n).
"""


def get_default_parameters_for_method(method):
    """Return the default PETSc solver parameters for a method.

    Parameters
    ----------
    method : str
        Name of the numerical method.

    Returns
    -------
    dict or None
        PETSc solver parameters for supported methods, or ``None`` when
        no default solver parameters are defined.
    """
    solver_parameters = None

    if method == "mass_lumped_triangle":
        solver_parameters = {
            "ksp_type": "preonly",
            "pc_type": "jacobi",
        }
    elif method == "spectral_quadrilateral":
        solver_parameters = {
            "ksp_type": "preonly",
            "pc_type": "jacobi",
        }
    else:
        solver_parameters = None

    return solver_parameters
