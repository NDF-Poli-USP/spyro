import firedrake as fire
import numpy as np
from netgen.geom2d import SplineGeometry
from netgen.meshing import Mesh, Element2D, MeshPoint, FaceDescriptor
from scipy.special import gamma, factorial
from scipy.integrate import quad
from scipy.spatial import KDTree
import ipdb
# https://docu.ngsolve.org/latest/netgen_tutorials/working_with_meshes.html


def superellipse_perimeter_integral(a, b, n):
    """
    Compute perimeter using numerical integration of the parametric form.
    """
    def integrand(theta):
        dx = -(2*a/n) * np.cos(theta)**(2/n - 1) * np.sin(theta)
        dy = (2*b/n) * np.sin(theta)**(2/n - 1) * np.cos(theta)
        return np.sqrt(dx**2 + dy**2)

    # Integrate from 0 to pi/2 and multiply by 4
    perimeter, error = quad(integrand, 0, np.pi/2)

    return 4 * perimeter


def superellipse_perimeter(a, b, n, max_terms=10):
    """
    Compute the perimeter of a superellipse using series expansion.

    Parameters:
        a, b (float): Semi-axes of the superellipse
        n (float): Exponent parameter (n >= 1)
        max_terms (int): Number of terms to use in the series expansion

    Returns:
        float: Perimeter approximation
    """
    # Handle special cases directly
    if n == 1:  # Diamond (rhombus) case
        return 4 * np.sqrt(a**2 + b**2)
    elif n == 2:  # Ellipse case
        if np.isclose(a, b):  # Circle
            return 2 * np.pi * a
        else:
            # Use complete elliptic integral of the second kind
            from scipy.special import ellipe
            e = np.sqrt(1 - (min(a, b)/max(a, b))**2)
            return 4 * max(a, b) * ellipe(e)
    elif n > 100:  # Approximate rectangle case
        return 4 * (a + b)

    # General case using series expansion
    term0 = 4 * (a + b) * gamma(1 + 1/n)**2 / gamma(1 + 2/n)

    if np.isclose(a, b):
        return term0

    series_sum = 0.0
    ratio = (a - b)/(a + b)

    for k in range(1, max_terms + 1):
        double_fact_top = gamma(2*k + 1) / (2**k * gamma(k + 1))  # (2k-1)!!
        double_fact_bottom = 2**k * gamma(k + 1)  # (2k)!!
        gamma_top = gamma(1 + 2*k/n)
        gamma_bottom = gamma(1 + (2*k - 1)/n)**2
        term = (double_fact_top/double_fact_bottom) * \
            (gamma_top/gamma_bottom) * ratio**(2*k)
        series_sum += term

    return term0 - (4*(a - b)**2/(a + b)) * series_sum


# Test cases
def test_perimeter_methods():
    cases = [
        (1.0, 1.0, 1),   # Diamond
        (1.0, 1.0, 2),   # Circle
        (1.0, 1.0, 100),  # Near-rectangle
        (2.0, 1.0, 2),   # Ellipse
        (0.95, 0.95, 4)  # Your case
    ]

    print(f"{'Case':<15} {'Integral':<15} {'Series':<15} {'Expected':<15}")
    for a, b, n in cases:
        integral = superellipse_perimeter_integral(a, b, n)
        series = superellipse_perimeter_series(a, b, n)

        if n == 1:
            expected = 4 * np.sqrt(a**2 + b**2)
        elif n == 2:
            expected = 2 * np.pi * a if np.isclose(a, b) else None
        elif n > 100:
            expected = 4 * (a + b)
        else:
            expected = None

        aux0 = f"a={a}, b={b}, n={n}: {integral:.6f} {series:.6f}"
        aux1 = f"{expected if expected is not None else '':<15}"
        print(aux0, aux1)


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


def create_composite_mesh(Lx, Ly, pad, n, lmin, lmax):

    # Parameters for the hyperellipse
    a = 0.5 * Lx + pad  # Semi-major axis
    b = 0.5 * Ly + pad  # Semi-minor axis
    perimeter = superellipse_perimeter_integral(a, b, n)

    # Boundary points: min 16 or 24
    num_bnd_pts = int(max(np.ceil(perimeter / lmax), 16))
    print(f"Number of boundary points: {num_bnd_pts}")

    # Initialize geometry
    geo = SplineGeometry()

    # 1. Create outer hyperelliptical boundary
    # Generate the hyperellipse boundary points
    num_bnd_pts += 1 if num_bnd_pts % 2 == 0 else 2  # to close the curve
    bnd_pts = parametric_hyperellipse(a, b, n, num_bnd_pts)
    # print(bnd_pts)

    geo = SplineGeometry()
    # Append points to the geometry
    [geo.AppendPoint(*pnt) for pnt in bnd_pts]

    # Generate the boundary curves
    curves = []
    for idp in range(0, num_bnd_pts - 1, 2):
        p1 = geo.PointData()[2][idp]
        p2 = geo.PointData()[2][idp + 1]
        p3 = geo.PointData()[2][idp + 2]
        curves.append(["spline3", p1, p2, p3])
        # print(p1, p2, p3)
    [geo.Append(c, bc="outer", maxh=lmin, leftdomain=1,
                rightdomain=0) for c in curves]

    # 2. Create inner rectangular hole (counter-clockwise)
    rect_vtx = [(-Lx/2, -Ly/2), (Lx/2, -Ly/2), (Lx/2, Ly/2), (-Lx/2, Ly/2)]
    [geo.AppendPoint(*pnt) for pnt in rect_vtx]

    # # Add rectangle edges
    geo.Append(["line", geo.PointData()[2][-4], geo.PointData()[2][-3]],
               bc="inner", maxh=lmax, leftdomain=0, rightdomain=1)
    geo.Append(["line", geo.PointData()[2][-3], geo.PointData()[2][-2]],
               bc="inner", maxh=lmax, leftdomain=0, rightdomain=1)
    geo.Append(["line", geo.PointData()[2][-2], geo.PointData()[2][-1]],
               bc="inner", maxh=lmax, leftdomain=0, rightdomain=1)
    geo.Append(["line", geo.PointData()[2][-1], geo.PointData()[2][-4]],
               bc="inner", maxh=lmax, leftdomain=0, rightdomain=1)

    # Set domains
    geo.SetMaterial(1, "outer")
    geo.SetMaterial(2, "inner")

    return geo.GenerateMesh(maxh=lmax, quad_dominated=False)


# Parameters for the hyperellipse
Lx = 1.0           # Length of the rectangle
Ly = 1.0           # Width of the rectangle
pad = 0.45         # Pad length
n = 5              # Degree of the hyperellipse
lmax = 0.05        # Maximum edge length
lmin = lmax        # Minimum edge length

# Create rectangular mesh
nx, ny = int(Lx / lmax), int(Ly / lmax)
mesh_rec = fire.RectangleMesh(nx, ny, Lx, Ly)
mesh_rec.coordinates.dat.data_with_halos[:, 0] -= Lx / 2
mesh_rec.coordinates.dat.data_with_halos[:, 1] -= Ly / 2

# Create hyperelliptical mesh
mesh_hyp = create_composite_mesh(Lx, Ly, pad, n, lmin, lmax)

# Merge meshes
msh = merge_meshes(mesh_rec, mesh_hyp)
fire.VTKFile("output/hyp_test.pvd").write(fire.Mesh(msh))
