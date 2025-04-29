import firedrake as fire
import netgen.meshing as ngm
import numpy as np
from math import sqrt, pi, cos, sin
from netgen.geom2d import SplineGeometry
from netgen.csg import Pnt, Vec, Ellipsoid, CSGeometry, Sphere, Plane, Pnt, Vec, ZRefinement
from netgen.meshing import Mesh, Element2D, Element3D, FaceDescriptor, MeshPoint, MeshingStep
from scipy.integrate import quad
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


def calculate_divisions(a, b, c, n, max_edge_length):
    # Calculate approximate circumference at equator (max circumference)
    per_equ = hyp_full_perimeter(a, b, n)
    nr = max(10, int(np.ceil(per_equ / max_edge_length)))

    # Calculate divisions along t (meridional direction)
    # Using half circumference (pi*c) as approximation
    per_mer = hyp_full_perimeter(a, c, n) / 2
    nt = max(5, int(np.ceil(per_mer / max_edge_length)))

    return nr, nt


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
    ellipsoid = Ellipsoid(Pnt(0, 0, 0), Vec(a, 0, 0), Vec(0, b, 0), Vec(0, 0, c))
    geo.Add(ellipsoid)
    mesh = geo.GenerateMesh(maxh=max_edge_length)
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
    t = 10 * pi / (num_pts_t - 1)

    # The boundary is an ellipse with semi-axes at height z0
    a_boundary = a * np.sin(t)**(2/n)
    b_boundary = b * np.sin(t)**(2/n)
    z0 = c * np.cos(t)**(2 / n)

    print(z0)

    # Generate points on the cap boundary
    bnd_cap = parametric_hyperellipse(a_boundary, b_boundary, n, num_pts_r)

    return bnd_cap


def create_ellipsoidal_cap_mesh(a, b, c, n, max_edge_length):
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
    [geo.Append(c, bc="hyp", maxh=max_edge_length) for c in curves]

    # Generate the mesh
    mesh = geo.GenerateMesh(maxh=max_edge_length, quad_dominated=False)
    mesh.Compress()

    return mesh, nt


def substitute_ellipsoid_caps(ellipsoid_mesh, cap_mesh_2d, a, b, c, n, num_pts_t):
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
    mesh = geo.GenerateMesh(maxh=max_edge_length / fact, perfstepsend=MeshingStep.MESHSURFACE)
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


a, b, c, n = 2.0, 1.5, 1.0, 4
lmax = 0.05

# Generate and export mesh
ellip_cad2D, nt = create_ellipsoidal_cap_mesh(a, b, c, n, lmax)
# fire.VTKFile("output/ellip_cad2D.pvd").write(fire.Mesh(ellip_cad2D))
ellip3D_pol = create_ellipsoid(a, b, c, n, lmax)
fire.VTKFile("output/ellip3D_pol.pvd").write(fire.Mesh(ellip3D_pol))
ellip3D_merge = substitute_ellipsoid_caps(ellip3D_pol, ellip_cad2D, a, b, c, n, nt)
fire.VTKFile("output/ellipsoid_test.pvd").write(fire.Mesh(ellip3D_merge))

ellipsoid_solid = create_hyperellipsoid_scaled(a, b, c, n, lmax)
# ellipsoid_solid = create_ellipsoid_solid(a, b, c, lmax)
# Export results
fire.VTKFile("output/ellipsoid_solid.pvd").write(fire.Mesh(ellipsoid_solid))
