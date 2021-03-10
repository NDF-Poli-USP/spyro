from firedrake import *
import numpy as np

#element_geometrys = ['triangle', 'quadrilateral']
element_geometry = 'quadrilateral'

if element_geometry == 'triangle':
    mesh =  UnitTriangleMesh()
    degrees = [1, 2, 3, 4, 5] 
    space_types = ['CG','KMV']


    for space_type in space_types:
        for degree in degrees:
            # Getting node locations
            print('For a '+space_type+' element with degree = '+ str(degree))
            V = FunctionSpace(mesh, space_type, degree )
            x, y = SpatialCoordinate(mesh)
            ux = Function(V).interpolate(x)
            uy = Function(V).interpolate(y)
            datax = ux.dat.data[:]
            datay = uy.dat.data[:]
            node_locations = np.zeros((len(datax), 2))
            node_locations[:, 0] = datax
            node_locations[:, 1] = datay

            # Calculating distance (getting largest nearest neighbor distance)
            max_nearest_neigh = 0.0 #small value to begin

            for i in range(len(datax)):
                # Finding nearest neighbor fot i
                nearest_neigh = 5.0  #large value to begin
                for j in range(len(datax)):
                    if i != j:
                        distance = np.sqrt( (node_locations[i, 0]-node_locations[j, 0])**2 +  (node_locations[i, 1]-node_locations[j, 1])**2 )
                        if distance < nearest_neigh:
                            nearest_neigh = distance
                if max_nearest_neigh < nearest_neigh:
                    max_nearest_neigh = nearest_neigh

            print('The max neighrest neighboor is '+ str(max_nearest_neigh))

if element_geometry == 'quadrilateral':


    import FIAT, finat

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



    mesh =  UnitSquareMesh(1,1,quadrilateral = True)
    degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
    space_types = ['spectral']


    for space_type in space_types:
        for degree in degrees:
            # Getting node locations
            print('For a '+space_type+' element with degree = '+ str(degree))
            element = FiniteElement('CG', mesh.ufl_cell(), degree=degree, variant='spectral')
            V = FunctionSpace(mesh, element)
            x, y = SpatialCoordinate(mesh)
            ux = Function(V).interpolate(x)
            uy = Function(V).interpolate(y)
            datax = ux.dat.data[:]
            datay = uy.dat.data[:]
            node_locations = np.zeros((len(datax), 2))
            node_locations[:, 0] = datax
            node_locations[:, 1] = datay

            # Calculating distance (getting largest nearest neighbor distance)
            max_nearest_neigh = 0.0 #small value to begin

            for i in range(len(datax)):
                # Finding nearest neighbor fot i
                nearest_neigh = 5.0  #large value to begin
                for j in range(len(datax)):
                    if i != j:
                        distance = np.sqrt( (node_locations[i, 0]-node_locations[j, 0])**2 +  (node_locations[i, 1]-node_locations[j, 1])**2 )
                        if distance < nearest_neigh:
                            nearest_neigh = distance
                if max_nearest_neigh < nearest_neigh:
                    max_nearest_neigh = nearest_neigh

            print('The max neighrest neighboor is '+ str(max_nearest_neigh))