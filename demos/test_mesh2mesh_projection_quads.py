from firedrake import *
import weakref
import FIAT, finat

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




m1 = RectangleMesh(100, 100, 1, 1,quadrilateral=True)
m2 = RectangleMesh(250, 250, 1, 1,quadrilateral=True)

# to run project in parallel
m2._parallel_compatible = {weakref.ref(m1)}

# the projection works for P1, P3, P4, P5 (max degree of KMV elements) but fails for P2
P = 2
element = FiniteElement('CG', m1.ufl_cell(), degree=P) #variant='spectral')
V1 = FunctionSpace(m1, element)
V2 = FunctionSpace(m2, element)

x, y = SpatialCoordinate(m1)
f1 = Function(V1).interpolate(sin(5*x)*cos(5*y))
#f1 = Function(V1).interpolate(Constant(1))

f2 = Function(V2)
#f2 = Projector(f1, V2).project() #it doesn't work
#f2.project(f1) #it doesn't work
#f2.interpolate(f1) #it doesn't work

# it works
m = V2.ufl_domain() 
W = VectorFunctionSpace(m, V2.ufl_element()) 
X = interpolate(m.coordinates, W) 
f2.dat.data[:] = f1.at(X.dat.data_ro, tolerance=0.001)

print(len(X.dat.data_ro))

# vertex only mesh will not work if the vertex pocitions change

# f2 should be numerically equal to f1
File("f1.pvd").write(f1)
File("f2.pvd").write(f2)
