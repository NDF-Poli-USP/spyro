from firedrake import *
import FIAT, finat
import numpy as np
import meshio
import SeismicMesh

def gauss_lobatto_legendre_line_rule(degree):
    fiat_make_rule = FIAT.quadrature.GaussLobattoLegendreQuadratureLineRule
    fiat_rule = fiat_make_rule(FIAT.ufc_simplex(1), degree + 1)
    finat_ps = finat.point_set.GaussLobattoLegendrePointSet
    finat_qr = finat.quadrature.QuadratureRule
    return finat_qr(finat_ps(fiat_rule.get_points()), fiat_rule.get_weights())

def gauss_lobatto_legendre_cube_rule(dimension, degree):
    make_tensor_rule = finat.quadrature.TensorProductQuadratureRule
    result = gauss_lobatto_legendre_line_rule(degree)
    for _ in range(1, dimension):
        line_rule = gauss_lobatto_legendre_line_rule(degree)
        result = make_tensor_rule([result, line_rule])
    return result

def old_alpha_func(method, degree):
    if method == 'KMV':
        if degree == 1:
            M = 1.0
        if degree == 2:
            M = 0.5
        if degree == 3:
            M = 0.2934695559090401
        if degree == 4:
            M = 0.21132486540518713
        if degree == 5:
            M = 0.20231237605867816

    if method == 'spectral':
        if degree == 1:
            M = 1.0
        if degree == 2:
            M = 0.5
        if degree == 3:
            M = 0.27639320225002106
        if degree == 4:
            M = 0.32732683535398854
        if degree == 5:
            M = 0.23991190372440996

    return M

def spectral_alpha_calc(degrees):
    element_type = 'spectral'

    if element_type == 'spectral':
        quadrilateral = True
        method = 'CG'
        variant = 'spectral'
    elif element_type == 'KMV':
        quadrilateral = False
        method = 'KMV'
        variant = 'KMV'
    n = 1000

    mesh =  UnitSquareMesh(n,n, quadrilateral = quadrilateral)
    degrees = [2]
    for degree in degrees:
    #degree = 5

        print('For a '+method+' element with degree = '+ str(degree))
        element = FiniteElement(method, mesh.ufl_cell(), degree=degree, variant=variant)
        V = FunctionSpace(mesh, element)
        x, y = SpatialCoordinate(mesh)
        u = Function(V)
        udat = u.dat.data[:]
        dof = len(udat)

        print(1./np.sqrt(dof/n**2))
        print(np.sqrt(dof/n**2))
    print('END')

def kmv_alpha_calc_with_seismicmesh():
    method = 'KMV'
    variant = 'KMV'

    bbox = (0.0, 1.0, 0.0, 1.0)
    rectangle =SeismicMesh.Rectangle(bbox)
    points, cells = SeismicMesh.generate_mesh(domain=rectangle, edge_length=0.005, verbose = 0, mesh_improvement=False )
    #points, cells = SeismicMesh.geometry.laplacian2(points, cells)
    meshio.write_points_cells("meshes/temp_dofcalc.msh",
        points,[("triangle", cells)],
        file_format="gmsh22", 
        binary = False
        )
    meshio.write_points_cells("meshes/temp_dofcalc.vtk",
            points,[("triangle", cells)],
            file_format="vtk"
            )

    mesh = Mesh(
        "meshes/temp_dofcalc.msh",
        distribution_parameters={
            "overlap_type": (DistributedMeshOverlapType.NONE, 0)
        },
    )
    ele = FiniteElement(method, mesh.ufl_cell(), degree=1, variant=variant)
    space = FunctionSpace(mesh, ele)

    fdrake_cell_node_map = space.cell_node_map()
    cell_node_map = fdrake_cell_node_map.values_with_halo
    (num_cells, nodes_per_cell) = cell_node_map.shape


    degrees = [1]
    for degree in degrees:
    #degree = 5

        print('For a '+method+' element with degree = '+ str(degree))
        element = FiniteElement(method, mesh.ufl_cell(), degree=degree, variant=variant)
        V = FunctionSpace(mesh, element)
        x, y = SpatialCoordinate(mesh)
        u = Function(V)
        udat = u.dat.data[:]
        dof = len(udat)

        print(1./np.sqrt(dof/num_cells))
        print(np.sqrt(dof/num_cells))
    print('END')

def old_to_new_g_converter(method, degree, Gold):
    if method == 'KMV':
        if degree == 1:
            G = 0.707813887967734*Gold
        if degree == 2:
            G = 0.8663141029672784*Gold
        if degree == 3:
            G = 0.7483761673104953*Gold
        if degree == 4:
            G = 0.7010127254535244*Gold
        if degree == 5:
            G = 0.9381929803311276*Gold

    return G


element_type = 'KMV'

if element_type == 'spectral':
    quadrilateral = True
    method = 'CG'
    variant = 'spectral'
elif element_type == 'KMV':
    quadrilateral = False
    method = 'KMV'
    variant = 'KMV'
n = 1000

mesh =  UnitSquareMesh(n,n, quadrilateral = quadrilateral)
degrees = [5]
for degree in degrees:
#degree = 5

    print('For a '+method+' element with degree = '+ str(degree))
    element = FiniteElement(method, mesh.ufl_cell(), degree=degree, variant=variant)
    V = FunctionSpace(mesh, element)
    x, y = SpatialCoordinate(mesh)
    u = Function(V)
    udat = u.dat.data[:]
    dof = len(udat)

    new_alpha = 1./np.sqrt(dof/ (2*n**2) )
    print(new_alpha)
    old_alpha = old_alpha_func(method, degree)
    converter = old_alpha/new_alpha
    print(converter)
print('END')

