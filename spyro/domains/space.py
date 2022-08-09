from firedrake import *


def FE_method(mesh, method, degree):
    """Define the finite element space:
    """

    if method == 'mass_lumped_triangle':
        element = FiniteElement('KMV', mesh.ufl_cell(), degree=degree, variant="KMV")
    elif method == 'spectral_quadrilateral':
        element = FiniteElement('CG', mesh.ufl_cell(), degree=degree, variant="spectral")
    elif method == 'DG_triangle' or 'DG_quadrilateral' or 'DG':
        element = FiniteElement("DG", mesh.ufl_cell(), degree=degree)
    elif method == 'CG_triangle' or 'CG_quadrilateral' or 'CG':
        element = FiniteElement("CG", mesh.ufl_cell(), degree=degree)
    
    function_space = FunctionSpace(mesh, element)
    return function_space
