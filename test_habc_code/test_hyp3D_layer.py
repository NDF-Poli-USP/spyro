import firedrake as fire
import numpy as np
from netgen.geom2d import SplineGeometry
from netgen.meshing import Mesh, Element2D, MeshPoint, FaceDescriptor
from scipy.special import gamma, factorial
from scipy.integrate import quad, dblquad
from scipy.spatial import KDTree
import ipdb
# https://docu.ngsolve.org/latest/netgen_tutorials/working_with_meshes.html


def hyperellipsoid_surface_area(a, b, c, n):
    """
    Compute the surface area of a hyperellipsoid using symmetry (first octant ×8).
    |x/a|^n + |y/b|^n + |z/c|^n = 1.

    Parameters:
        a, b, c (float): Semi-axes.
        n (float): Exponent (n=2 for standard ellipsoid).

    Returns:
        float: Total surface area.
    """
    def surface_element(r, t):
        # Parametric equations (first octant, no abs/sgn needed)
        x = a * (np.cos(r) * np.sin(t))**(2/n)
        y = b * (np.sin(r) * np.sin(t))**(2/n)
        z = c * np.cos(t)**(2/n)

        # Partial derivatives (simplified)
        dx_dr = a * (2/n) * (np.cos(r)*np.sin(t))**(2/n - 1) * (-np.sin(r)*np.sin(t))
        dy_dr = b * (2/n) * (np.sin(r)*np.sin(t))**(2/n - 1) * (np.cos(r)*np.sin(t))
        dz_dr = 0.0

        dx_dt = a * (2/n) * (np.cos(r)*np.sin(t))**(2/n - 1) * (np.cos(r)*np.cos(t))
        dy_dt = b * (2/n) * (np.sin(r)*np.sin(t))**(2/n - 1) * (np.sin(r)*np.cos(t))
        dz_dt = c * (2/n) * (np.cos(t))**(2/n - 1) * (-np.sin(t))

        # Cross product and magnitude
        F1 = dy_dr * dz_dt  # dz_dr = 0
        F2 = -dx_dr * dz_dt  # dz_dr = 0
        F3 = dx_dr * dy_dt - dy_dr * dx_dt
        dS = np.sqrt(F1**2 + F2**2 + F3**2)
        return dS

    # Integrate over first octant (r ∈ [0, π/2], t ∈ [0, π/2]) and multiply by 8
    area, _ = dblquad(lambda t, r: surface_element(r, t),
                      0, np.pi/2,  # r limits
                      lambda r: 0, lambda r: np.pi / 2)  # t limits

    return 8 * area


# Test cases
def test_surface_cases():

    # Example usage
    a, b, c = 1.0, 1.0, 1.0  # Unit sphere for n=2
    n = 2.0
    surface_area = hyperellipsoid_surface_area(a, b, c, n)
    print(f"Surface area for n={n}: {surface_area:.6f} (Exact: 12.566371)")

    n = 4.0
    surface_area = hyperellipsoid_surface_area(a, b, c, n)
    print(f"Surface area for n={n}: {surface_area:.6f}")

    n = 24.0
    surface_area = hyperellipsoid_surface_area(a, b, c, n)
    print(f"Surface area for n={n}: {surface_area:.6f} (Exact: 24.0)")


# Function to define the hyperellipsoid boundary points
def parametric_hyperellipsoid(a, b, c, n, num_pts):
    """
    Generate parametric coordinates for a hyperellipsoid:
    |x/a|^n + |y/b|^n + |z/c|^n = 1

    Parameters:
        a, b, c (float): Semi-axes
        n (float): Exponent
        num_pts(int): Angular resolution

    Returns:
        tuple: (x, y, z) coordinates
    """

    # Create wdge popints
    c_zero = [np.pi / 2., 3 * np.pi / 2.]
    s_zero = [0., np.pi, 2 * np.pi]
    theta = np.linspace(0, 2 * np.pi, num_pts)  # Azimuthal angle (0 to 2π)
    n_pts_phi = num_pts // 2 + 1 if num_pts // 2 % 2 == 0 else num_pts // 2 + 2
    phi = np.linspace(0, np.pi, n_pts_phi)  # Polar angle (0 to π)

    cr = np.cos(theta)
    sr = np.sin(theta)
    cr = np.where(np.isin(theta, c_zero), 0, cr)
    sr = np.where(np.isin(theta, s_zero), 0, sr)

    ct = np.cos(phi)
    st = np.sin(phi)
    ct = np.where(np.isin(phi, c_zero), 0, ct)
    st = np.where(np.isin(phi, s_zero), 0, st)

    aux0 = np.outer(cr, st)
    aux1 = np.outer(sr, st)
    aux2 = np.outer(np.ones(num_pts), ct)
    x = a * np.abs(aux0)**(2 / n) * np.sign(aux0)
    y = b * np.abs(aux1)**(2 / n) * np.sign(aux1)
    z = c * np.abs(aux2)**(2 / n) * np.sign(aux2)
    # https://github.com/oszn/rengongzhineng_zuoye/blob/459fea5bd0dcb34457d750f4b22abb750bb0e493/k/xixi/drx.py

    # Return as a list of arrays
    return [x, y, z]


def merge_meshes_without_match_bnds(mesh_rec, mesh_hyp):
    # Create final mesh that will contain both
    final_mesh = Mesh()
    final_mesh.dim = 2

    # First add all rectangular mesh vertices
    # Get Firedrake mesh coordinates
    coords = mesh_rec.coordinates.dat.data_ro

    # Add all vertices to Netgen mesh and create mapping
    rec_map = {}
    for i, coord in enumerate(coords):
        x, y = coord
        rec_map[i] = final_mesh.Add(MeshPoint((x, y, 0.)))  # z = 0 for 2D

    # Get Firedrake cells
    cells = mesh_rec.coordinates.cell_node_map().values

    # Face descriptor for the rectangular mesh
    fd_rec = final_mesh.Add(FaceDescriptor(bc=1, domin=1, domout=2))

    # Add all elements to Netgen mesh
    for cell in cells:
        netgen_points = [rec_map[cell[i]] for i in range(len(cell))]
        final_mesh.Add(Element2D(fd_rec, netgen_points))

    # Then add hyperellipse mesh
    hyp_map = {}
    for i, p in enumerate(mesh_hyp.Points()):
        x, y, z = p
        hyp_map[i + 1] = final_mesh.Add(MeshPoint((x, y, z)))

    # Face descriptor for the hyperelliptical mesh
    fd_hyp = final_mesh.Add(FaceDescriptor(bc=2, domin=2, domout=0))

    # Add all elements to Netgen mesh
    for el in mesh_hyp.Elements2D():
        netgen_points = [hyp_map[p] for p in el.points]
        final_mesh.Add(Element2D(fd_hyp, netgen_points))

    return final_mesh


def fire_mesh_to_netgen(mesh_fire):
    # Create empty Netgen mesh
    netgen_mesh = Mesh()
    netgen_mesh.dim = 2  # 2D mesh

    # Get Firedrake mesh coordinates
    coords = mesh_fire.coordinates.dat.data_ro

    # Add all vertices to Netgen mesh
    vertex_map = {}  # To map Firedrake vertex indices to Netgen point numbers
    for i, coord in enumerate(coords):
        x, y = coord
        vertex_map[i] = netgen_mesh.Add(MeshPoint((x, y, 0.)))  # z=0 for 2D

    # Get Firedrake cells (triangles in this case)
    cells = mesh_fire.coordinates.cell_node_map().values

    # Face descriptor for the rectangular mesh
    fd_rec = netgen_mesh.Add(FaceDescriptor(bc=1, domin=1, domout=2))

    # Add all elements to Netgen mesh
    for cell in cells:
        # Get Netgen point numbers for this triangle's vertices
        netgen_points = [vertex_map[cell[i]] for i in range(3)]
        netgen_mesh.Add(Element2D(fd_rec, netgen_points))

    return netgen_mesh


def merge_meshes(mesh_rec, mesh_hyp):
    # Create final mesh that will contain both
    final_mesh = Mesh()
    final_mesh.dim = 2

    # First add all rectangular mesh vertices
    coords = mesh_rec.coordinates.dat.data_ro

    # Find min/max x and y to detect boundary points
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    # Add all vertices to Netgen mesh and create mapping
    rec_map = {}
    boundary_points = set()  # Stores indices of boundary points in rec_map
    boundary_coords_list = []  # List of boundary coordinates for KDTree

    for i, coord in enumerate(coords):
        x, y = coord
        rec_map[i] = final_mesh.Add(MeshPoint((x, y, 0.)))  # z = 0 for 2D

        # Check if this point is on the boundary using np.isclose
        if (np.isclose(x, x_min) or np.isclose(x, x_max)
                or np.isclose(y, y_min) or np.isclose(y, y_max)):
            boundary_points.add(i)
            boundary_coords_list.append((x, y))

    # Create KDTree for efficient nearest neighbor search
    if boundary_coords_list:
        boundary_tree = KDTree(boundary_coords_list)
    else:
        boundary_tree = None

    # Get Firedrake cells
    cells = mesh_rec.coordinates.cell_node_map().values

    # Face descriptor for the rectangular mesh
    fd_rec = final_mesh.Add(FaceDescriptor(bc=1, domin=1, domout=2))

    # Add all elements to Netgen mesh
    for cell in cells:
        netgen_points = [rec_map[cell[i]] for i in range(len(cell))]
        final_mesh.Add(Element2D(fd_rec, netgen_points))

    # Then add hyperellipse mesh
    hyp_map = {}
    boundary_coords = {tuple(coords[idx]): rec_map[idx]
                       for idx in boundary_points}

    for i, p in enumerate(mesh_hyp.Points()):
        x, y, z = p
        # First check if point is near any boundary extent (quick check)
        if (np.isclose(x, x_min) or np.isclose(x, x_max)
                or np.isclose(y, y_min) or np.isclose(y, y_max)):

            # Check for exact matches first
            is_duplicate = False
            for (bx, by), bnd_point_id in boundary_coords.items():
                if np.isclose(x, bx) and np.isclose(y, by):
                    hyp_map[i + 1] = bnd_point_id  # Reuse the existing point
                    is_duplicate = True
                    break

            if not is_duplicate and boundary_tree:
                # Find the nearest boundary point
                dist, idx = boundary_tree.query((x, y))
                nearest_boundary_pnt = boundary_coords_list[idx]

                # Get the corresponding point ID from boundary_coords
                near_bnd_pnt_id = boundary_coords.get(nearest_boundary_pnt)
                if near_bnd_pnt_id is not None:
                    hyp_map[i + 1] = near_bnd_pnt_id
                else:
                    hyp_map[i + 1] = final_mesh.Add(MeshPoint((x, y, z)))

            elif not is_duplicate:
                hyp_map[i + 1] = final_mesh.Add(MeshPoint((x, y, z)))
        else:
            # Point is clearly not on boundary, add directly
            hyp_map[i + 1] = final_mesh.Add(MeshPoint((x, y, z)))

    # Face descriptor for the hyperelliptical mesh
    fd_hyp = final_mesh.Add(FaceDescriptor(bc=2, domin=2, domout=0))

    # Add all elements to Netgen mesh
    for el in mesh_hyp.Elements2D():
        netgen_points = [hyp_map[p] for p in el.points]
        final_mesh.Add(Element2D(fd_hyp, netgen_points))

    return final_mesh


def create_composite_mesh(Lx, Ly, Lz, pad, n, lmin, lmax):

    # Parameters for the hyperellipse
    a = 0.5 * Lx + pad  # Semi-axis 1
    b = 0.5 * Ly + pad  # Semi-axis 2
    c = 0.5 * Lz + pad  # Semi-axis 3
    surface = hyperellipsoid_surface_area(a, b, c, n)

    # Boundary points: min 16 or 24
    num_bnd_pts = int(max(np.ceil(np.sqrt((surface / lmax**2))) + 1, 16))
    print(f"Number of boundary points: {num_bnd_pts}")

    # 1. Create outer hyperellipsoidal boundary
    # Generate the hyperellipse boundary points
    num_bnd_pts += 1 if num_bnd_pts % 2 == 0 else 2  # to close the curve
    bnd_pts = parametric_hyperellipsoid(a, b, c, n, num_bnd_pts)
    # print(bnd_pts)

    # Initialize geometry
    geo = SplineGeometry()

    # Append points to the geometry
    [geo.AppendPoint(*pnt) for pnt in bnd_pts]

    # # Generate the boundary curves
    # curves = []
    # for idp in range(0, num_bnd_pts - 1, 2):
    #     ipdb.set_trace()
    #     p1 = geo.PointData()[2][idp]
    #     p2 = geo.PointData()[2][idp + 1]
    #     p3 = geo.PointData()[2][idp + 2]
    #     curves.append(["spline3", p1, p2, p3])
    #     # print(p1, p2, p3)
    # [geo.Append(c, bc="outer", maxh=lmin, leftdomain=1,
    #             rightdomain=0) for c in curves]

    # # 2. Create inner rectangular hole (counter-clockwise)
    # rect_vtx = [(-Lx/2, -Ly/2), (Lx/2, -Ly/2), (Lx/2, Ly/2), (-Lx/2, Ly/2)]
    # [geo.AppendPoint(*pnt) for pnt in rect_vtx]

    # # # Add rectangle edges
    # geo.Append(["line", geo.PointData()[2][-4], geo.PointData()[2][-3]],
    #            bc="inner", maxh=lmax, leftdomain=0, rightdomain=1)
    # geo.Append(["line", geo.PointData()[2][-3], geo.PointData()[2][-2]],
    #            bc="inner", maxh=lmax, leftdomain=0, rightdomain=1)
    # geo.Append(["line", geo.PointData()[2][-2], geo.PointData()[2][-1]],
    #            bc="inner", maxh=lmax, leftdomain=0, rightdomain=1)
    # geo.Append(["line", geo.PointData()[2][-1], geo.PointData()[2][-4]],
    #            bc="inner", maxh=lmax, leftdomain=0, rightdomain=1)

    # # Set domains
    # geo.SetMaterial(1, "outer")
    # geo.SetMaterial(2, "inner")

    return geo.GenerateMesh(maxh=lmax, quad_dominated=False)


# Parameters for the hyperellipsoid
Lx = 1.0           # Length of the parallelepiped
Ly = 1.0           # Width of the parallelepiped
Lz = 1.0           # Depth of the parallelepiped
pad = 0.45         # Pad length
n = 2              # Degree of the hyperellipsoid
lmax = 0.05        # Maximum edge length
lmin = lmax        # Minimum edge length

# Create rectangular mesh
nx, ny, nz = int(Lx / lmax), int(Ly / lmax), int(Lz / lmax)
mesh_rec = fire.BoxMesh(nx, ny, nz, Lx, Ly, Lz)
mesh_rec.coordinates.dat.data_with_halos[:, 0] -= Lx / 2
mesh_rec.coordinates.dat.data_with_halos[:, 1] -= Ly / 2
mesh_rec.coordinates.dat.data_with_halos[:, 2] -= Lz / 2

# Create hyperellipsoidal mesh
mesh_hyp = create_composite_mesh(Lx, Ly, Lz, pad, n, lmin, lmax)

# # Merge meshes
# msh = merge_meshes(mesh_rec, mesh_hyp)
# fire.VTKFile("output/hyp_test.pvd").write(fire.Mesh(msh))
# https://docu.ngsolve.org/latest/netgen_tutorials/define_2d_geometries.html
