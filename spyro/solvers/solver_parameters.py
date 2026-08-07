def get_default_linear_solver_parameters(method):
    """Return default PETSc/KSP options for a finite-element method.

    Parameters
    ----------
    method : `str` or `None`
        The finite element method to use. Must be one of:
        'mass_lumped_triangle' or 'spectral_quadrilateral'.

    Returns
    -------
    solver_parameters : `dict` or `None`
        Solver options if the method is recognized, otherwise ``None``.
    """
    if method in {"mass_lumped_triangle", "spectral_quadrilateral"}:
        return {
            "ksp_type": "preonly",
            "pc_type": "jacobi",
        }

    return None
