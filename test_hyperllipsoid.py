import firedrake as fire
import numpy as np
from math import sqrt, pi, cos, sin
from netgen.csg import CSGeometry, OrthoBrick, Pnt
from netgen.meshing import Mesh, Element2D, Element3D, FaceDescriptor, MeshPoint
from scipy.spatial import KDTree, Delaunay, ConvexHull
from sys import float_info
import ipdb


def create_edge_surface_mesh(edge_points, final_mesh,
                             existing_points, edge_length):

    kdtree = KDTree(existing_points)  # Rebuild KDTree

    # Create a temporary mesh for the edge surface
    edge_mesh = Mesh()
    edge_mesh.dim = 3

    # Add points to temporary mesh and create mapping
    edge_coords = []
    point_map = {}
    for i, p in enumerate(edge_points):
        # p should be a coordinate tuple (x,y,z)
        coords = (final_mesh[p][0], final_mesh[p][1], final_mesh[p][2])
        mp = edge_mesh.Add(MeshPoint(coords))
        point_map[i] = mp
        edge_coords.append(coords)

    # Add face descriptor (bc=2 matches your existing fd_edg)
    fd_edg = edge_mesh.Add(FaceDescriptor(bc=2, domin=1, domout=0))

    # Generate surface triangles (simplified approach)
    if len(edge_points) == 3:
        # Simple triangle
        edge_mesh.Add(Element2D(fd_edg, [point_map[i] for i in range(3)]))
    else:
        # Convex hull approach for simple cases
        points_array = np.array(edge_coords)
        hull = ConvexHull(points_array)
        valid_simplices = []
        for simplex in hull.simplices:
            # Get the 3 points of the simplex (triangle)
            tri_points = points_array[simplex]

            # Check all 3 edges of the triangle
            edges = [np.linalg.norm(tri_points[0] - tri_points[1]),
                     np.linalg.norm(tri_points[1] - tri_points[2]),
                     np.linalg.norm(tri_points[2] - tri_points[0])]

            # Keep simplex if all edges are "short"
            if all(edge <= edge_length for edge in edges):
                valid_simplices.append(simplex)

        # Add filtered triangles to mesh
        for simplex in valid_simplices:
            edge_mesh.Add(Element2D(fd_edg, [point_map[i] for i in simplex]))

    # edge_mesh.Refine(adaptive=True)
    fire.VTKFile("output/hyp_edge_mesh.pvd").write(fire.Mesh(edge_mesh))

    # Transfer to final mesh
    tol = 1e-10

    # Create a list to store valid elements
    valid_elements = []

    # Add surface elements to final mesh - with hyperellipsoid filtering
    for el in edge_mesh.Elements2D():
        vert = [vi.nr for vi in el.vertices]
        coord_vert = [edge_mesh[vi] for vi in vert]
        valid_elements.append(el)

    # Now clear and rebuild the edge mesh with only valid elements
    filtered_edge_mesh = Mesh()
    filtered_edge_mesh.dim = 3

    # Add points and create new mapping (automatically handles duplicates)
    point_map = {}  # maps old point indices to new ones
    for el in valid_elements:

        vert = [vi.nr for vi in el.vertices]
        coord_vert = [edge_mesh[vi] for vi in vert]

        for idv, p in enumerate(coord_vert):
            if vert[idv] not in point_map:
                coords = (p[0], p[1], p[2])
                new_p = filtered_edge_mesh.Add(MeshPoint(coords))
                point_map[vert[idv]] = new_p

    # Add face descriptor to new mesh
    fd_edg_filtered = filtered_edge_mesh.Add(
        FaceDescriptor(bc=2, domin=1, domout=0))

    # Add the valid elements to the new mesh
    for el in valid_elements:
        new_vertices = [point_map[vi.nr] for vi in el.vertices]
        filtered_edge_mesh.Add(Element2D(fd_edg_filtered, new_vertices))

    # Optional: Refine the filtered mesh
    fire.VTKFile("output/hyp_edge_mesh.pvd").write(fire.Mesh(filtered_edge_mesh))
    # filtered_edge_mesh.Refine(adaptive=True)

    # Add surface elements to final mesh
    for el in filtered_edge_mesh.Elements2D():

        vert = [vi.nr for vi in el.vertices]
        coord_vert = [filtered_edge_mesh[vi] for vi in vert]

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
        final_mesh.Add(Element2D(fd_edg, nodes))

    # ipdb.set_trace()
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
    final_mesh = Mesh()
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
    #     final_mesh.Add(Element3D(1, verts))

    # Create face descriptors for the hyp3D mesh
    fd_hyp = final_mesh.Add(FaceDescriptor(bc=1, domin=1, domout=0))

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
            rel_lim = 1.0
            if (rel_x >= rel_lim and rel_y >= rel_lim) or \
               (rel_x >= rel_lim and rel_z >= rel_lim) or \
                    (rel_y >= rel_lim and rel_z >= rel_lim):
                # Edge or corner point
                edge_pnt[el.index - 1].add(pnt)

        # Add the projected face
        final_mesh.Add(Element2D(fd_hyp, proj_nodes))

    edge = [set() for _ in range(12)]
    for pnts in edge_pnt:
        for p in pnts:
            rel_x = final_mesh[p][0] / d
            rel_y = final_mesh[p][1] / e
            rel_z = final_mesh[p][2] / f

            if (rel_x == 1. and rel_y >= 1.) or (rel_y == 1. and rel_x >= 1.):
                edge[0].add(p)
                edge_lim = (1 - s)*(a**2 + b**2)**0.5
            if (rel_x == 1. and rel_y <= -1.) or (rel_y == -1. and rel_x >= 1.):
                edge[1].add(p)
                edge_lim = (1 - s)*(a**2 + b**2)**0.5
            if (rel_x == 1. and rel_z >= 1.) or (rel_z == 1. and rel_x >= 1.):
                edge[2].add(p)
                edge_lim = (1 - s)*(a**2 + c**2)**0.5
            if (rel_x == 1. and rel_z <= -1.) or (rel_z == -1. and rel_x >= 1.):
                edge[3].add(p)
                edge_lim = (1 - s)*(a**2 + c**2)**0.5
            if (rel_x == -1. and rel_y >= 1.) or (rel_y == 1. and rel_x <= -1.):
                edge[4].add(p)
                edge_lim = (1 - s)*(a**2 + b**2)**0.5
            if (rel_x == -1. and rel_y <= -1.) or (rel_y == -1. and rel_x <= -1.):
                edge[5].add(p)
                edge_lim = (1 - s)*(a**2 + b**2)**0.5
            if (rel_x == -1. and rel_z >= 1.) or (rel_z == 1. and rel_x <= -1.):
                edge[6].add(p)
                edge_lim = (1 - s)*(a**2 + c**2)**0.5
            if (rel_x == -1. and rel_z <= -1.) or (rel_z == -1. and rel_x <= -1.):
                edge[7].add(p)
                edge_lim = (1 - s)*(a**2 + c**2)**0.5
            if (rel_y == 1. and rel_z >= 1.) or (rel_z == 1. and rel_y >= 1.):
                edge[8].add(p)
                edge_lim = (1 - s)*(b**2 + c**2)**0.5
            if (rel_y == 1. and rel_z <= -1.) or (rel_z == -1. and rel_y >= 1.):
                edge[9].add(p)
                edge_lim = (1 - s)*(b**2 + c**2)**0.5
            if (rel_y == -1. and rel_z >= 1.) or (rel_z == 1. and rel_y <= -1.):
                edge[10].add(p)
                edge_lim = (1 - s)*(b**2 + c**2)**0.5
            if (rel_y == -1. and rel_z <= -1.) or (rel_z == -1. and rel_y <= -1.):
                edge[11].add(p)
                edge_lim = (1 - s)*(b**2 + c**2)**0.5

    for edg in edge:
        create_edge_surface_mesh(edg, final_mesh, existing_points, edge_lim)

    final_mesh.Compress()
    # final_mesh.Refine(adaptive=True)
    # final_mesh.GenerateVolumeMesh()
    # ipdb.set_trace()

    return final_mesh, box_mesh


a, b, c, n = 2.0, 1.5, 1.0, 6
lmax = 0.1

# Generate and export mesh
hyp3D_mesh, box_mesh = complete_hyp3D_mesh(a, b, c, n, lmax)
fire.VTKFile("output/hyp3D_mesh.pvd").write(fire.Mesh(hyp3D_mesh))
fire.VTKFile("output/box_mesh.pvd").write(fire.Mesh(box_mesh))

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


# # Find closest point pairs between non-adjacent edges
# n = len(edge_points)
# closest_pairs = []
# for i in range(n):
#     for j in range(i+2, n):  # Skip adjacent points
#         if j == n-1 and i == 0:
#             continue  # Skip wrap-around pair
#         dist = np.linalg.norm(points_array[i] - points_array[j])
#         closest_pairs.append((i, j, dist))

# # Sort by distance and keep only the closest 25% to avoid too many midpoints
# closest_pairs.sort(key=lambda x: x[2])
# selected_pairs = closest_pairs[:n//2]

# # Create midpoints and add to points_array
# for p, (i, j, _) in enumerate(selected_pairs):
#     midpoint = (points_array[i] + points_array[j]) / 2
#     coords = (midpoint[0], midpoint[1], midpoint[2])
#     mp = edge_mesh.Add(MeshPoint(coords))
#     point_map[n + p] = mp
#     edge_coords.append(coords)
