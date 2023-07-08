import firedrake as fire
import FIAT
import finat
import numpy as np

def get_hexa_real_cell_node_map(V, mesh):
    weird_cnm_func = V.cell_node_map()
    weird_cnm = weird_cnm_func.values_with_halo
    cells_per_layer, nodes_per_cell = np.shape(weird_cnm)
    layers = mesh.cell_set.layers - 1
    ufl_element = V.ufl_element()
    _, p = ufl_element.degree()

    cell_node_map = np.zeros((layers*cells_per_layer, nodes_per_cell), dtype=int)

    for layer in range(layers):
        for cell in range(cells_per_layer):
            cnm_base = weird_cnm[cell]
            cell_id = layer + layers*cell
            cell_node_map[cell_id] = [item+layer*(p) for item in cnm_base]
            temp=0
    
    return cell_node_map


def gauss_lobatto_legendre_line_rule(degree):
    fiat_make_rule = FIAT.quadrature.GaussLobattoLegendreQuadratureLineRule
    fiat_rule = fiat_make_rule(FIAT.ufc_simplex(1), degree + 1)
    finat_ps = finat.point_set.GaussLobattoLegendrePointSet
    points = finat_ps(fiat_rule.get_points())
    weights = fiat_rule.get_weights()
    return finat.quadrature.QuadratureRule(points, weights)

def gauss_lobatto_legendre_cube_rule(dimension, degree):
    make_tensor_rule = finat.quadrature.TensorProductQuadratureRule
    result = gauss_lobatto_legendre_line_rule(degree)
    for _ in range(1, dimension):
        line_rule = gauss_lobatto_legendre_line_rule(degree)
        result = make_tensor_rule([result, line_rule])
    return result

quad_mesh = fire.RectangleMesh(2, 2, 1.0, 1.0, quadrilateral=True)
layer_height = 0.3
mesh = fire.ExtrudedMesh(quad_mesh, 2, layer_height=layer_height)
x_mesh, y_mesh, z_mesh = fire.SpatialCoordinate(mesh)
element = fire.FiniteElement("CG", mesh.ufl_cell(), degree=2, variant='spectral')
V = fire.FunctionSpace(mesh, element)
x_func = fire.Function(V)
y_func = fire.Function(V)
z_func = fire.Function(V)

x_func.interpolate(x_mesh)
y_func.interpolate(y_mesh)
z_func.interpolate(z_mesh)

x = x_func.dat.data[:]
y = y_func.dat.data[:]
z = z_func.dat.data[:]

fdrake_cm = V.cell_node_map()
cell_node_map = fdrake_cm.values_with_halo

new_cnm = get_hexa_real_cell_node_map(V, mesh)

points = np.zeros((len(x),3))
for j in range(len(x)):
    points[j] = [x[j],y[j],z[j]]


print("END")
