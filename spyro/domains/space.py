from firedrake import *


def FE_method(mesh, method, degree):
    """Define the finite element method:
    Space discretization - Continuous
    or Discontinuous Galerkin methods"""
    cell_geometry = mesh.ufl_cell()
    if method == "CG":
        # CG - Continuous Galerkin
        if cell_geometry == quadrilateral or cell_geometry == hexahedron:
            element = FiniteElement(
                method, mesh.ufl_cell(), degree=degree, variant="spectral"
            )
        else:
            element = FiniteElement(
                method, mesh.ufl_cell(), degree=degree, variant="equispaced"
            )
    elif method == "DG":
        if cell_geometry == quadrilateral or cell_geometry == hexahedron:
            element = FiniteElement(
                method, mesh.ufl_cell(), degree=degree, variant="spectral"
            )
        else:
            element = FiniteElement(
                method, mesh.ufl_cell(), degree=degree, variant="equispaced"
            )
    elif method == "KMV":
        # CG- with KMV elements
        element = FiniteElement(method, mesh.ufl_cell(), degree=degree, variant="KMV")
    return element
