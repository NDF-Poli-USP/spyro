def get_default_parameters_for_method(method):
    """Get the default solver parameters for a given method.

    Parameters
    ----------
    value : `str` or `None`
        The finite element method to use. Must be one of:
        'mass_lumped_triangle' or 'spectral_quadrilateral'.

    Returns
    -------
    solver_parameters : `dict`or `None`
        A dictionary of solver parameters if the method is recognized, otherwise `None`.
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
