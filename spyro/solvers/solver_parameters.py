def get_default_parameters_for_method(method):
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
