import firedrake as fire
import netgen.meshing as ngm
import numpy as np
from math import sqrt, pi, cos, sin
from netgen.geom2d import SplineGeometry
from netgen.csg import Pnt, Vec, Ellipsoid, CSGeometry, Sphere, Plane, Pnt, Vec, ZRefinement, OrthoBrick
from netgen.meshing import Mesh, Element2D, Element3D, FaceDescriptor, MeshPoint, MeshingStep
from scipy.integrate import quad
from scipy.spatial import KDTree, Delaunay, ConvexHull
from sys import float_info
import ipdb


def hyp_full_perimeter(a, b, n):
    '''
    Compute perimeter of a hyperellipse.

    Parameters
    ----------
    a : `float`
        Hyperellipse semi-axis in direction 1
    b : `float`
        Hyperellipse semi-axis in direction 2
    n : `int`
        Degree of the hyperellipse

    Returns
    -------
    perim_hyp : `float`
        Perimeter of the hyperellipse
    '''

    def integrand(theta):
        '''
        Differential arc length element to compute the perimeter

        Parameters
        ----------
        theta : `float`
            Angle in radians

        Returns
        -------
        ds : `float`
            Differential arc length element
        '''
        dx = -(2 * a / n) * np.cos(theta)**(2 / n - 1) * np.sin(theta)
        dy = (2 * b / n) * np.sin(theta)**(2 / n - 1) * np.cos(theta)
        ds = np.sqrt(dx**2 + dy**2)

        return ds

    # Integrate from 0 to pi/2 and multiply by 4
    perim_hyp = 4 * quad(integrand, 0, np.pi/2)[0]

    return perim_hyp


def create_ellipsoid(a, b, c, n, max_edge_length):
    # Calculate appropriate divisions
    nr, nt = calculate_divisions(a, b, c, n, max_edge_length)
    nr += 1 if nr % 2 == 0 else 0  # Ensure nr is even
    nt = nt + 1 if nt % 2 == 0 else nt  # Ensure nt is even
    print(f"Using nr={nr}, nt={nt}")

    # Create mesh
    mesh = ngm.Mesh()
    mesh.Add(ngm.FaceDescriptor(surfnr=1, domin=1, domout=0, bc=1))

    # Generate points
    point_index = {}
    r_values = np.linspace(0, 2*np.pi, nr, endpoint=False)
    t_values = np.linspace(0, np.pi, nt)

    c_zero = [np.pi / 2., 3 * np.pi / 2.]
    s_zero = [0., np.pi, 2 * np.pi]

    cr = np.cos(r_values)
    sr = np.sin(r_values)
    cr = np.where(np.isin(r_values, c_zero), 0, cr)
    sr = np.where(np.isin(r_values, s_zero), 0, sr)

    ct = np.cos(t_values)
    st = np.sin(t_values)
    ct = np.where(np.isin(t_values, c_zero), 0, ct)
    st = np.where(np.isin(t_values, s_zero), 0, st)

    for i, _ in enumerate(r_values):
        vcr, vsr = cr[i], sr[i]
        for j, _ in enumerate(t_values):
            vct, vst = ct[j], st[j]
            x = a * np.abs(vcr * vst)**(2 / n) * np.sign(vcr * vst)
            y = b * np.abs(vsr * vst)**(2 / n) * np.sign(vsr * vst)
            z = c * np.abs(vct)**(2 / n) * np.sign(vct)
            pid = mesh.Add(ngm.MeshPoint(ngm.Pnt(x, y, z)))
            point_index[(i, j)] = pid

    # Add poles
    bottom_pid = mesh.Add(ngm.MeshPoint(ngm.Pnt(0, 0, -c)))
    top_pid = mesh.Add(ngm.MeshPoint(ngm.Pnt(0, 0, c)))

    # Create surface elements
    for i in range(nr):
        i_next = (i + 1) % nr

        # Top cap
        mesh.Add(ngm.Element2D(1, [point_index[(i, 0)],
                                   point_index[(i_next, 0)],
                                   top_pid]))

        # Bottom cap
        mesh.Add(ngm.Element2D(1, [point_index[(i, nt-1)],
                                   bottom_pid,
                                   point_index[(i_next, nt-1)]]))

        # Sides
        for j in range(nt-1):
            p1 = point_index[(i, j)]
            p2 = point_index[(i_next, j)]
            p3 = point_index[(i_next, j+1)]
            p4 = point_index[(i, j+1)]

            mesh.Add(ngm.Element2D(1, [p1, p2, p4]))
            mesh.Add(ngm.Element2D(1, [p2, p3, p4]))

    # Generate volume mesh
    # mesh.Refine()
    mesh.Compress()
    # mesh.GenerateVolumeMesh(maxh=max_edge_length)

    return mesh


def create_ellipsoid_solid(a, b, c, max_edge_length):
    # Option 1: Simple CSG approach (recommended)
    print("Using CSG approach for robust mesh generation")
    geo = CSGeometry()
    ellipsoid = Ellipsoid(Pnt(0, 0, 0), Vec(a, 0, 0),
                          Vec(0, b, 0), Vec(0, 0, c))
    geo.Add(ellipsoid)
    mesh = geo.GenerateMesh(maxh=max_edge_length)
    return mesh


def substitute_ellipsoid_caps(ellipsoid_mesh, cap_mesh_2d,
                              a, b, c, n, num_pts_t):
    """Fix mapping issues when substituting the caps of an ellipsoid mesh."""

    t = 10 * pi / (num_pts_t - 1)
    z0 = c * np.cos(t)**(2/n)

    merged_mesh = ngm.Mesh()
    merged_mesh.dim = 3

    # Identify non-cap points
    ellipsoid_points = list(ellipsoid_mesh.Points())
    ellipsoid_elements = list(ellipsoid_mesh.Elements2D())

    point_map = {}
    cap_points = set()

    for i, p in enumerate(ellipsoid_points, 1):
        if abs(p[2]) > z0:
            cap_points.add(i)

    for i, p in enumerate(ellipsoid_points, 1):
        if i not in cap_points:
            new_p = merged_mesh.Add(ngm.MeshPoint((p[0], p[1], p[2])))
            point_map[i] = new_p

    fd_hyp = merged_mesh.Add(ngm.FaceDescriptor(bc=1, domin=1, domout=0))

    for el in ellipsoid_elements:
        if any(pnr in cap_points for pnr in el.vertices):
            continue

        try:
            new_vertices = [point_map[pnr] for pnr in el.vertices]
            merged_mesh.Add(ngm.Element2D(fd_hyp, new_vertices))
        except KeyError:
            continue

    # Correct Projection for Top Cap Mesh
    cap_points_3d = []
    cap_points_2d = list(cap_mesh_2d.Points())
    for p in cap_points_2d:
        x2d, y2d = p[0], p[1]
        z3d = (1 - abs(x2d / a)**n - abs(y2d / b)**n)**(1/n)
        cap_points_3d.append((x2d, y2d, z3d))

    num_cap_points = len(cap_points_2d)
    cap_point_indices = []
    for idxpt in range(num_cap_points):
        pt = cap_points_3d[idxpt]
        pid = merged_mesh.Add(ngm.MeshPoint(ngm.Pnt(*pt)))
        cap_point_indices.append(pid)

    cap_elements = list(cap_mesh_2d.Elements2D())
    for el in cap_elements:
        if len(el.vertices) != 3:
            continue

        p1, p2, p3 = el.vertices[0].nr - 1, el.vertices[1].nr - 1, el.vertices[2].nr - 1

        merged_mesh.Add(ngm.Element2D(fd_hyp, [
            cap_point_indices[p1],
            cap_point_indices[p2],
            cap_point_indices[p3]
        ]))

    # Correct Projection for botton Cap Mesh
    cap_points_3d = []
    cap_points_2d = list(cap_mesh_2d.Points())
    for p in cap_points_2d:
        x2d, y2d = p[0], p[1]
        z3d = (1 - abs(x2d / a)**n - abs(y2d / b)**n)**(1/n)
        cap_points_3d.append((x2d, y2d, -z3d))

    num_cap_points = len(cap_points_2d)
    cap_point_indices = []
    for idxpt in range(num_cap_points):
        pt = cap_points_3d[idxpt]
        pid = merged_mesh.Add(ngm.MeshPoint(ngm.Pnt(*pt)))
        cap_point_indices.append(pid)

    cap_elements = list(cap_mesh_2d.Elements2D())
    for el in cap_elements:
        if len(el.vertices) != 3:
            continue

        p1, p2, p3 = el.vertices[0].nr - 1, el.vertices[1].nr - 1, el.vertices[2].nr - 1

        merged_mesh.Add(ngm.Element2D(fd_hyp, [
            cap_point_indices[p1],
            cap_point_indices[p2],
            cap_point_indices[p3]
        ]))

    merged_mesh.Compress()

    return merged_mesh


def create_hyperellipsoid_scaled(a, b, c, n, max_edge_length):
    """
    Create a mesh for a hyperellipsoid defined by |x/a|ⁿ + |y/b|ⁿ + |z/c|ⁿ = 1

    Parameters:
        a, b, c (float): Semi-axes lengths
        n (float): Degree of the hyperellipsoid (n=2 gives standard ellipsoid)
        max_edge_length (float): Maximum edge length for mesh generation
    """
    # Since CSG doesn't directly support hyperellipsoids, we'll approximate with a surface mesh

    # Step 1: Create a temporary mesh
    geo = CSGeometry()

    fact = max(a/b, c/b, b/a, c/a, a/c, b/c)
    sphere = Sphere(Pnt(0., 0., 0.), 1.0)
    geo.Add(sphere)
    mesh = geo.GenerateMesh(maxh=max_edge_length / fact,
                            perfstepsend=MeshingStep.MESHSURFACE)
    # mesh.Refine()

    # Transform the sphere points to create the hyperellipsoid
    for i, p in enumerate(mesh.Coordinates()):
        x, y, z = p

        # Compute the Lp norm
        r = (abs(x)**n + abs(y)**n + abs(z)**n)**(1.0/n)

        if r > 1e-16:  # Avoid division by zero
            # Scale each component according to the hyperellipsoid equation
            x_scaled = a * np.sign(x) * (abs(x))**(2/n)
            y_scaled = b * np.sign(y) * (abs(y))**(2/n)
            z_scaled = c * np.sign(z) * (abs(z))**(2/n)

            # Apply the transformation
            mesh.Coordinates()[i] = (x_scaled, y_scaled, z_scaled)

    # mesh.Curve(2)
    mesh.Compress()

    return mesh

# Function to define the hyperellipse boundary points


def parametric_hyperellipse(a, b, n, num_pts):
    r_ang = np.linspace(0, 2 * np.pi, num_pts)

    rc_zero = [np.pi / 2., 3 * np.pi / 2.]
    rs_zero = [0., np.pi, 2 * np.pi]

    cr = np.cos(r_ang)
    sr = np.sin(r_ang)
    cr = np.where(np.isin(r_ang, rc_zero), 0, cr)
    sr = np.where(np.isin(r_ang, rs_zero), 0, sr)

    x = a * np.sign(cr) * np.abs(cr)**(2/n)
    y = b * np.sign(sr) * np.abs(sr)**(2/n)
    return np.column_stack((x, y))


def create_ellipsoidal_cap_boundary(a, b, c, n, num_pts_r, num_pts_t):
    """
    Create boundary points of an ellipsoidal cap with given height.

    Parameters:
    a, b, c - semi-axes of the ellipsoid (a and b equatorial, c polar)
    h - height - height of the cap (distance from pole along c-axis)

    Returns:
    Array of (x,y) points representing the boundary of the cap
    """

    # Calculate the parameter t (angle from pole) where the cap ends
    # At height h from the pole (z = c - h)
    t = pi / 2  # 10 * pi / (num_pts_t - 1)

    # The boundary is an ellipse with semi-axes at height z0
    a_boundary = a * np.sin(t)**(2 / n)
    b_boundary = b * np.sin(t)**(2 / n)
    z0 = c * np.cos(t)**(2 / n)
    print('z0: {}'.format(z0))

    # Generate points on the cap boundary
    bnd_cap = parametric_hyperellipse(a_boundary, b_boundary, n, num_pts_r)

    return bnd_cap


def calculate_divisions(a, b, c, n, max_edge_length):

    # Calculate divisions in equatorial direction
    per_equ = hyp_full_perimeter(a, b, n)
    nr = max(10, int(np.ceil(per_equ / max_edge_length)))

    # Calculate divisions in meridional direction
    per_mer = hyp_full_perimeter(a, c, n) / 2
    nt = max(5, int(np.ceil(per_mer / max_edge_length)))

    return nr, nt


def create_hyp_cap_mesh(a, b, c, n, max_edge_length):
    """
    Creates a 2D mesh of an ellipsoidal cap boundary using Netgen.

    Args:
        a, b, c: Semi-axes of the ellipsoid (a, b equatorial, c polar).
        max_edge_length: Maximum allowed edge length in the mesh.

    Returns:
        Netgen mesh object.
    """

    # Calculate appropriate divisions
    nr, nt = calculate_divisions(a, b, c, n, max_edge_length)
    nr += 1 if nr % 2 == 0 else 0  # Ensure nr is even
    nt = nt + 1 if nt % 2 == 0 else nt  # Ensure nt is even
    print(f"Using nr={nr}, nt={nt}")

    # Generate points on the ellipse
    bnd_pts = create_ellipsoidal_cap_boundary(a, b, c, n, nr, nt)

    # Initialize geometry
    geo = SplineGeometry()

    # Append points to the geometry
    [geo.AppendPoint(*pnt) for pnt in bnd_pts]

    # Generate the boundary curves
    curves = []
    for idp in range(0, nr - 1, 2):
        p1 = geo.PointData()[2][idp]
        p2 = geo.PointData()[2][idp + 1]
        p3 = geo.PointData()[2][idp + 2]
        curves.append(["spline3", p1, p2, p3])
        # print(p1, p2, p3)
    [geo.Append(curv, bc="hyp", maxh=max_edge_length) for curv in curves]

    # Generate the mesh
    mesh = geo.GenerateMesh(maxh=max_edge_length, quad_dominated=False)
    mesh.Compress()

    return mesh


def projection_hyp2D_mesh(hyp2D_mesh_equ, hyp2D_mesh_mer, a, b, c, n, lmax):

    caps_mesh = ngm.Mesh()
    caps_mesh.dim = 3

    # Identify cap points
    equ_points = list(hyp2D_mesh_equ.Points())
    equ_elements = list(hyp2D_mesh_equ.Elements2D())
    mer_points = list(hyp2D_mesh_mer.Points())
    mer_elements = list(hyp2D_mesh_mer.Elements2D())

    point_map = {}
    edge_pnts = {}
    tol = 4 * lmax   # float_info.epsilon
    print('tol: {}'.format(tol))

    # Step 1: Add cap points
    for i, p in enumerate(equ_points, 1):
        x2d, y2d = p[0], p[1]
        z3d = 0. if 1 - abs(x2d / a)**n - abs(y2d / b)**n < tol \
            else c * (1 - abs(x2d / a)**n - abs(y2d / b)**n)**(1/n)

        if z3d > 0.:
            point_map[i] = caps_mesh.Add(MeshPoint((x2d, y2d, z3d)))
            point_map[len(equ_points) + i] = caps_mesh.Add(
                MeshPoint((x2d, y2d, -z3d)))
        else:
            edge_pnts[i] = caps_mesh.Add(MeshPoint((x2d, y2d, 0.0)))

    for i, p in enumerate(mer_points, 1):
        x2d, z2d = p[0], p[1]
        y3d = 0. if 1 - abs(x2d / a)**n - abs(z2d / c)**n < tol \
            else b * (1 - abs(x2d / a)**n - abs(z2d / c)**n)**(1/n)

        if y3d > 0.:
            point_map[i + 2 * len(equ_points)] = caps_mesh.Add(MeshPoint((x2d, y3d, z2d)))
            point_map[2 * len(equ_points) + len(mer_points) + i] = caps_mesh.Add(
                MeshPoint((x2d, -y3d, z2d)))
        else:
            edge_pnts[i + len(equ_points)] = caps_mesh.Add(MeshPoint((x2d, 0.0, z2d)))

    fd_hyp = caps_mesh.Add(ngm.FaceDescriptor(bc=1, domin=1, domout=0))

    # Step 2: Add the original cap elements
    for el in equ_elements:
        p1, p2, p3 = [el.vertices[i].nr for i in range(len(el.vertices))]

        if p1 in point_map and p2 in point_map and p3 in point_map:
            caps_mesh.Add(ngm.Element2D(
                fd_hyp, [point_map[p1], point_map[p2], point_map[p3]]))
            caps_mesh.Add(ngm.Element2D(
                fd_hyp, [point_map[p1 + len(equ_points)],
                         point_map[p2 + len(equ_points)],
                         point_map[p3 + len(equ_points)]]))

    for el in mer_elements:
        p1, p2, p3 = [el.vertices[i].nr + 2 * len(equ_points) for i in range(len(el.vertices))]

        if p1 in point_map and p2 in point_map and p3 in point_map:
            caps_mesh.Add(ngm.Element2D(
                fd_hyp, [point_map[p1], point_map[p2], point_map[p3]]))
            caps_mesh.Add(ngm.Element2D(
                fd_hyp, [point_map[p1 + len(mer_points)],
                         point_map[p2 + len(mer_points)],
                         point_map[p3 + len(mer_points)]]))

    caps_mesh.Curve(n)
    caps_mesh.Compress()

    return caps_mesh, point_map, edge_pnts


def create_edge_surface_mesh(edge_points, final_mesh, existing_points, edge_length):

    kdtree = KDTree(existing_points)  # Rebuild KDTree

    # Create a temporary mesh for the edge surface
    edge_mesh = Mesh()

    # Add face descriptor (bc=2 matches your existing fd_edg)
    fd_edg = edge_mesh.Add(FaceDescriptor(bc=2, domin=1, domout=0))

    # Add points to temporary mesh and create mapping
    edge_coords = []
    point_map = {}
    for i, p in enumerate(edge_points):
        # p should be a coordinate tuple (x,y,z)
        coords = (final_mesh[p][0], final_mesh[p][1], final_mesh[p][2])
        mp = edge_mesh.Add(MeshPoint(coords))
        point_map[i] = mp
        edge_coords.append(coords)

    # Generate surface triangles (simplified approach)
    # For production code, use proper surface reconstruction here
    if len(edge_points) == 3:
        # Simple triangle
        edge_mesh.Add(Element2D(fd_edg, [point_map[i] for i in range(3)]))
    else:
        # Convex hull approach for simple cases
        points_array = np.array(edge_coords)
        hull = ConvexHull(points_array)

        # Add triangles to mesh
        for simplex in hull.simplices:
            edge_mesh.Add(Element2D(fd_edg, [point_map[i] for i in simplex]))
    # fire.VTKFile("output/hyp_edge_mesh.pvd").write(fire.Mesh(edge_mesh))

    # Transfer to final mesh
    tol = 1e-10
    # Add surface elements to final mesh
    for el in edge_mesh.Elements2D():

        vert = [vi.nr for vi in el.vertices]
        coord_vert = [edge_mesh[vi] for vi in vert]

        nodes = []
        for p in coord_vert:
            coords = (p[0], p[1], p[2])
            dist, idx = kdtree.query(coords)

            # Check if point exists in final mesh
            if dist <= tol:
                # Reuse the existing point
                nodes.append(idx + 1)
            else:
                pnt = final_mesh.Add(MeshPoint(coords))
                nodes.append(pnt)

                # Update KDTree and cache
                existing_points = np.vstack([existing_points, coords])
                kdtree = KDTree(existing_points)  # Rebuild KDTree

        # Add the edge element to the final mesh
        final_mesh.Add(ngm.Element2D(fd_edg, nodes))

    return final_mesh, existing_points


def complete_hyp3D_mesh(a, b, c, n, edge_length):
    # Compute the superness
    s = 3**(-1 / n)

    d = a * s
    e = b * s
    f = c * s

    geo1 = CSGeometry()
    box = OrthoBrick(Pnt(-d, -e, -f), Pnt(d, e, f))
    geo1.Add(box)
    box_mesh = geo1.GenerateMesh(maxh=edge_length)

    # Create a new mesh for the final result
    final_mesh = ngm.Mesh()
    final_mesh.dim = 3

    box_pnts = list(box_mesh.Points())
    # box_elem = list(box_mesh.Elements3D())

    # point_map = {}
    # # Add vertices from original mesh
    # for i, p in enumerate(box_pnts, 1):
    #     x, y, z = p[0], p[1], p[2]
    #     point_map[i] = final_mesh.Add(MeshPoint((x, y, z)))

    # # Copy the original box mesh into our final mesh
    # for el in box_elem:
    #     verts = [point_map[v.nr] for v in el.vertices]
    #     final_mesh.Add(ngm.Element3D(1, verts))

    # Create face descriptors for the hyp3D mesh
    fd_hyp = final_mesh.Add(ngm.FaceDescriptor(bc=1, domin=1, domout=0))

    # Initialize KDTree for existing points (empty at first)
    existing_points = np.empty((0, 3))  # Start with no points
    kdtree = KDTree(existing_points)
    tolerance = 1e-10  # Adjust based on your precision needs

    # Dictionary to map coordinates to existing mesh points (for reuse)
    point_cache = {}

    # Get unique faces and edge points
    edge_pnt = [set() for _ in range(6)]
    # Create projected points for each element in the box mesh
    for el in box_mesh.Elements2D():
        boundary_faces = tuple(vi.nr for vi in el.vertices)

        # Get original points
        orig_points = [box_pnts[vi - 1] for vi in boundary_faces]

        # Project points and add to mesh
        proj_nodes = []
        for p in orig_points:
            xs, ys, zs = p[0], p[1], p[2]
            rel_x = abs(xs / d)
            rel_y = abs(ys / e)
            rel_z = abs(zs / f)

            ref = sum([rel_x, rel_y, rel_z])
            if ref == 3.:  # Corner point (all coordinates on boundary)
                x = xs
                y = ys
                z = zs
            else:

                if rel_x == 1. and (el.index == 4 or el.index == 2):  # X-face
                    y = ys
                    z = zs
                    x = a * np.sign(xs) * (1 - abs(y / b)**n - abs(z / c)**n)**(1/n)

                if rel_y == 1. and (el.index == 6 or el.index == 3):  # Y-face
                    x = xs
                    z = zs
                    y = b * np.sign(ys) * (1 - abs(x / a)**n - abs(z / c)**n)**(1/n)

                if rel_z == 1. and (el.index == 5 or el.index == 1):  # Z-face
                    x = xs
                    y = ys
                    z = c * np.sign(zs) * (1 - abs(x / a)**n - abs(y / b)**n)**(1/n)

            new_point = (x, y, z)

            # Check if point already exists using KDTree
            if existing_points.size:
                dist, idx = kdtree.query(new_point)
                if dist <= tolerance:
                    # Reuse existing point
                    cached_pnt = point_cache[tuple(existing_points[idx])]
                    proj_nodes.append(cached_pnt)
                    continue

            # If not found, create new point
            pnt = final_mesh.Add(MeshPoint(new_point))
            proj_nodes.append(pnt)

            # Update KDTree and cache
            existing_points = np.vstack([existing_points, new_point])
            kdtree = KDTree(existing_points)  # Rebuild KDTree
            point_cache[tuple(new_point)] = pnt

            # Get nodes from box mesh on the edges
            if (rel_x >= 1 and rel_y >= 1) or \
               (rel_x >= 1 and rel_z >= 1) or \
                    (rel_y >= 1 and rel_z >= 1):
                # Edge or corner point
                edge_pnt[el.index - 1].add(pnt)

        # Add the projected face
        final_mesh.Add(ngm.Element2D(fd_hyp, proj_nodes))

    edge = [set() for _ in range(12)]
    for pnts in edge_pnt:
        for p in pnts:
            rel_x = final_mesh[p][0] / d
            rel_y = final_mesh[p][1] / e
            rel_z = final_mesh[p][2] / f

            if (rel_x == 1. and rel_y >= 1.) or (rel_y == 1. and rel_x >= 1.):
                edge[0].add(p)

            if (rel_x == 1. and rel_y <= -1.) or (rel_y == -1. and rel_x >= 1.):
                edge[1].add(p)

            if (rel_x == 1. and rel_z >= 1.) or (rel_z == 1. and rel_x >= 1.):
                edge[2].add(p)

            if (rel_x == 1. and rel_z <= -1.) or (rel_z == -1. and rel_x >= 1.):
                edge[3].add(p)

            if (rel_x == -1. and rel_y >= 1.) or (rel_y == 1. and rel_x <= -1.):
                edge[4].add(p)

            if (rel_x == -1. and rel_y <= -1.) or (rel_y == -1. and rel_x <= -1.):
                edge[5].add(p)

            if (rel_x == -1. and rel_z >= 1.) or (rel_z == 1. and rel_x <= -1.):
                edge[6].add(p)

            if (rel_x == -1. and rel_z <= -1.) or (rel_z == -1. and rel_x <= -1.):
                edge[7].add(p)

            if (rel_y == 1. and rel_z >= 1.) or (rel_z == 1. and rel_y >= 1.):
                edge[8].add(p)

            if (rel_y == 1. and rel_z <= -1.) or (rel_z == -1. and rel_y >= 1.):
                edge[9].add(p)

            if (rel_y == -1. and rel_z >= 1.) or (rel_z == 1. and rel_y <= -1.):
                edge[10].add(p)

            if (rel_y == -1. and rel_z <= -1.) or (rel_z == -1. and rel_y <= -1.):
                edge[11].add(p)

    for edg in edge:
        create_edge_surface_mesh(edg, final_mesh, existing_points, edge_length)

    final_mesh.Compress()
    # final_mesh.GenerateVolumeMesh()

    return final_mesh, box_mesh


a, b, c, n = 2.0, 1.5, 1.0, 6
lmax = 0.1

# Generate and export mesh
hyp3D_mesh, box_mesh = complete_hyp3D_mesh(a, b, c, n, lmax)
fire.VTKFile("output/hyp3D_mesh.pvd").write(fire.Mesh(hyp3D_mesh))
fire.VTKFile("output/box_mesh.pvd").write(fire.Mesh(box_mesh))

# hyp2D_mesh_equ = create_hyp_cap_mesh(a, b, c, n, lmax)
# hyp2D_mesh_mer = create_hyp_cap_mesh(a, c, b, n, lmax)
# fire.VTKFile("output/hyp2D_mesh_equ.pvd").write(fire.Mesh(hyp2D_mesh_equ))
# fire.VTKFile("output/hyp2D_mesh_mer.pvd").write(fire.Mesh(hyp2D_mesh_mer))
# ellip3D_pol = create_ellipsoid(a, b, c, n, lmax)
# fire.VTKFile("output/ellip3D_pol.pvd").write(fire.Mesh(ellip3D_pol))
# ellip3D_merge = substitute_ellipsoid_caps(ellip3D_pol, ellip_cad2D, a, b, c, n, nt)
# fire.VTKFile("output/ellipsoid_test.pvd").write(fire.Mesh(ellip3D_merge))

# ellipsoid_solid = create_hyperellipsoid_scaled(a, b, c, n, lmax)
# # ellipsoid_solid = create_ellipsoid_solid(a, b, c, lmax)
# # Export results
# fire.VTKFile("output/ellipsoid_solid.pvd").write(fire.Mesh(ellipsoid_solid))

# for p in orig_points:
#     xs, ys, zs = p[0], p[1], p[2]

#     r_xy = sqrt(xs**2 + ys**2)
#     r_z = sqrt(r_xy**2 + zs**2)

#     if r_xy > float_info.epsilon:
#         cr = xs / r_xy
#         sr = ys / r_xy
#     else:
#         cr = 1.0
#         sr = 0.0

#     if r_z > float_info.epsilon:
#         ct = zs / r_z
#         st = r_xy / r_z
#     else:
#         ct = 1.0
#         st = 0.0

#     x = a * np.abs(cr * st)**(2 / n) * np.sign(xs)
#     y = b * np.abs(sr * st)**(2 / n) * np.sign(ys)
#     z = c * np.abs(ct)**(2 / n) * np.sign(zs)
