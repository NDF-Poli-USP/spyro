from firedrake import *


def FE_method(mesh, method, degree):
    """Define the finite element space:

    Parameters:
    -----------
    mesh: Firedrake Mesh
        Mesh to be used in the finite element space.
    method: str
        Method to be used for the finite element space.
    degree: int
        Degree of the finite element space.

    Returns:
    --------
    function_space: Firedrake FunctionSpace
        Function space.
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
