import firedrake
import numpy as np

def calculate_mesh_quality(mesh):

    # Only works for two-dimensional meshes
    dim = mesh.topological_dimension()
    if dim != 2:
        raise NotImplementedError(f"Dimension {dim} has not been considered yet")

    def distance_between_points(a, b):
        d = np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        return d

    def angle_between_vectors(a, b):
        cosine_angle = np.dot(a,b) / ( np.linalg.norm(a)*np.linalg.norm(b) )
        return np.arccos( cosine_angle) # in radian

    V = firedrake.FunctionSpace(mesh, "CG", 1)
    V0 = firedrake.FunctionSpace(mesh, "DG", 0)
    
    cell_node_map = V.cell_node_map().values_with_halo
    x_mesh, y_mesh = firedrake.SpatialCoordinate(mesh)
    
    ux = firedrake.Function(V)
    uy = firedrake.Function(V)
    cr_func = firedrake.Function(V0)
    ca_func = firedrake.Function(V0)
    
    ux.interpolate(x_mesh)
    uy.interpolate(y_mesh)
    x = ux.dat.data[:]
    y = uy.dat.data[:]
    
    mesh_quality = []
   
    if mesh.ufl_cell().is_simplex():

        cr_func.interpolate(firedrake.Circumradius(mesh))
        ca_func.interpolate(firedrake.CellVolume(mesh))
        cell_circumradius = cr_func.dat.data[:]
        cell_areas = ca_func.dat.data[:]

        for i in range(mesh.num_cells()):
            p0 = [x[cell_node_map[i][0]], y[cell_node_map[i][0]]]
            p1 = [x[cell_node_map[i][1]], y[cell_node_map[i][1]]]
            p2 = [x[cell_node_map[i][2]], y[cell_node_map[i][2]]]
            edge1 = distance_between_points(p0, p1)
            edge2 = distance_between_points(p0, p2)
            edge3 = distance_between_points(p2, p1)
            perimeter = edge1 + edge2 + edge3
            cell_inscribed = 2*cell_areas[i]/perimeter
            quality_metric = 2*cell_inscribed/cell_circumradius[i]
            mesh_quality.append(quality_metric)

    else: 
        #  quadrilateral
        # 1-------------3
        # |             |
        # |             |
        # |             |
        # |             |
        # |             | 
        # 0-------------2

        for i in range(mesh.num_cells()):
            p0 = np.array([x[cell_node_map[i][0]], y[cell_node_map[i][0]]])
            p1 = np.array([x[cell_node_map[i][1]], y[cell_node_map[i][1]]])
            p2 = np.array([x[cell_node_map[i][2]], y[cell_node_map[i][2]]])
            p3 = np.array([x[cell_node_map[i][3]], y[cell_node_map[i][3]]])
            
            theta = np.empty(4) 
            theta[0] = angle_between_vectors(p1-p0, p2-p0)  # theta 0  
            theta[1] = angle_between_vectors(p3-p1, p0-p1)  # theta 1  
            theta[2] = angle_between_vectors(p0-p2, p3-p2)  # theta 2
            theta[3] = angle_between_vectors(p2-p3, p1-p3)  # theta 3  

            mesh_quality.append( max(1-(2./np.pi)*np.max( np.abs( np.pi/2.-theta ) ), 0) )

    return mesh_quality
