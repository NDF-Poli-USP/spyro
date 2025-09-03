import meshio
import SeismicMesh
from SeismicMesh import generation
from SeismicMesh import geometry
import numpy as np
import scipy.spatial as spatial
from scipy.spatial import Delaunay

# Problem Scale
scale = 1
# Element Size
h = 0.1*scale
# Cubic domain inside hyperellipsoid
domainX = 1.0*scale  # x size of cubic domain
domainY = 1.0*scale  # y size of cubic domain
domainZ = 1.0*scale  # z size of cubic domain
# HyperEllipsoid Padding
ellipseLx = 0.5*scale  # ellipseLx = Padding from cube domain x side
ellipseLy = 0.5*scale  # ellipseLy = Padding from cube domain y side
ellipseLz = 0.5*scale  # ellipseLz = Padding from cube domain z side
# HyperEllipsoid exponent
ellipse_n = 3.0
###################################################

###################################################
# Calculating dimensions for cube inside the hyperellipsoid
box_xmin = 0.0
box_xmax = domainX
box_ymin = 0.0
box_ymax = domainY
box_zmin = -domainZ
box_zmax = 0.0
cube = (box_xmin, box_xmax, box_ymin, box_ymax, box_zmin, box_zmax)
cube1 = SeismicMesh.geometry.Rectangle(cube)
# Calculating parameters for hyperellipsoid with center at center of cube
ellipse_a = domainX/2 + ellipseLx  # x semi-axis
ellipse_b = domainY/2 + ellipseLy  # y semi-axis
ellipse_c = domainZ/2 + ellipseLz  # z semi-axis
xc = domainX/2
yc = domainY/2
zc = -domainZ/2
z_cut = 0.0        # z-coordinate for surface cut
# Seismic mesh needs a box that envelopes all the domain
# the box is calculated here
domain_xmin = xc - ellipse_a
domain_xmax = xc + ellipse_a
domain_ymin = yc - ellipse_b
domain_ymax = yc + ellipse_b
domain_zmin = zc - ellipse_c
domain_zmax = zc + ellipse_c
bbox = (domain_xmin, domain_xmax, domain_ymin, domain_ymax, domain_zmin, domain_zmax)


def superellipsoide_sdf(p, a=ellipse_a, b=ellipse_b, c=ellipse_c, n=ellipse_n, xc=xc, yc=yc, zc=zc):
    # Shift coordinates to center at (xc, yc, zc)
    x = (p[:, 0] - xc) / a
    y = (p[:, 1] - yc) / b
    z = (p[:, 2] - zc) / c
    # Avoid zero to the power of small/large n by adding epsilon
    eps = 1e-8
    x_n = np.power(np.abs(x) + eps, n)
    y_n = np.power(np.abs(y) + eps, n)
    z_n = np.power(np.abs(z) + eps, n)
    # Compute radius in hyperellipsoid space
    r = np.power(x_n + y_n + z_n, 1.0 / n)
    # Signed distance
    sdf = r - 1.0
    # Rescale back to original space
    scale = np.minimum(np.minimum(a, b), c)
    return sdf * scale

# Rectangle for cutting the ellipse


def top_cut_sdf(p, z_cut=0.0):
    return (p[:, 2] - z_cut)


def u_shape_sdf3D(p):
    return np.maximum(superellipsoide_sdf(p), top_cut_sdf(p))

# Post processing function used by SeismicMesh


def _remove_triangles_outside(p, t, fd, geps):
    """Remove vertices outside the domain"""
    dim = p.shape[1]
    pmid = p[t].sum(1) / (dim + 1)  # Compute centroids
    return t[fd(pmid) < -geps]  # Keep interior triangles

# Function to remove points too close to the internal cube


def remove_points_near_cube(all_points, xmin, xmax, ymin, ymax, zmin, zmax,
                            distance_threshold, tolerance=1e-10):
    """
    Remove points that are outside the cube but within a specified distance from it.

    Parameters:
    - all_points: numpy array of shape (n, 3) containing all points
    - xmin, xmax, ymin, ymax, zmin, zmax: cube bounds
    - distance_threshold: distance from cube surface within which points are removed
    - tolerance: floating point tolerance for inside/outside determination

    Returns:
    - numpy array with points in buffer zone removed
    """

    if len(all_points) == 0:
        return all_points.copy()

    all_points = np.asarray(all_points)
    result_points = []

    for point in all_points:
        x, y, z = point

        # Check if point is inside cube
        inside = (xmin - tolerance <= x <= xmax + tolerance
                  and ymin - tolerance <= y <= ymax + tolerance
                  and zmin - tolerance <= z <= zmax + tolerance)

        if inside:
            # Point is inside cube → keep it
            result_points.append(point)
        else:
            # Point is outside cube → calculate distance to cube
            distance = calculate_distance_to_cube(point, xmin, xmax, ymin, ymax, zmin, zmax)

            if distance > distance_threshold:
                # Point is far enough from cube → keep it
                result_points.append(point)
            # If distance <= threshold, remove it (don't add to result)

    return np.array(result_points) if result_points else np.array([]).reshape(0, 3)


def calculate_distance_to_cube(point, xmin, xmax, ymin, ymax, zmin, zmax):
    """
    Calculate the minimum distance from a point to a cube.

    Returns 0 if point is inside the cube.
    """
    x, y, z = point

    # Calculate distance components for each axis
    dx = max(0, xmin - x, x - xmax)
    dy = max(0, ymin - y, y - ymax)
    dz = max(0, zmin - z, z - zmax)

    # Euclidean distance
    return np.sqrt(dx*dx + dy*dy + dz*dz)

# Function to create cube structured mesh points
def create_cube_filled_points(xmin, xmax, ymin, ymax, zmin, zmax,
                              x_spacing, y_spacing, z_spacing):
    """
    Create points that fill the entire cube volume.
    """

    # Calculate number of points
    n_x = int(np.ceil((xmax - xmin) / x_spacing)) + 1
    n_y = int(np.ceil((ymax - ymin) / y_spacing)) + 1
    n_z = int(np.ceil((zmax - zmin) / z_spacing)) + 1

    # Create coordinate arrays
    x_coords = np.linspace(xmin, xmax, n_x)
    y_coords = np.linspace(ymin, ymax, n_y)
    z_coords = np.linspace(zmin, zmax, n_z)

    # Create all combinations
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])

    return points


# Function to remove points that arent from the structured mesh
def remove_cube_interior_points(all_points, cube_points,
                                xmin=None, xmax=None, ymin=None, ymax=None,
                                zmin=None, zmax=None, tolerance=1e-10):
    """
    Remove points that are inside the cube but keep cube_points themselves.

    Parameters:
    - all_points: numpy array of shape (n, 3) containing all points
    - cube_points: numpy array of shape (m, 3) containing cube surface points to keep
    - xmin, xmax, ymin, ymax, zmin, zmax: cube bounds (calculated from cube_points if None)
    - tolerance: floating point tolerance for comparisons

    Returns:
    - numpy array with interior points removed (keeps exterior + cube surface points)
    """

    if len(all_points) == 0:
        return all_points.copy()

    # Convert to numpy arrays
    all_points = np.asarray(all_points)
    cube_points = np.asarray(cube_points)

    # Determine cube bounds if not provided
    if any(bound is None for bound in [xmin, xmax, ymin, ymax, zmin, zmax]):
        if len(cube_points) == 0:
            return all_points.copy()

        xmin = np.min(cube_points[:, 0])
        xmax = np.max(cube_points[:, 0])
        ymin = np.min(cube_points[:, 1])
        ymax = np.max(cube_points[:, 1])
        zmin = np.min(cube_points[:, 2])
        zmax = np.max(cube_points[:, 2])

    result_points = []

    for point in all_points:
        x, y, z = point

        # Check if point is clearly outside cube -> keep it
        outside = (x < xmin - tolerance or x > xmax + tolerance
                   or y < ymin - tolerance or y > ymax + tolerance
                   or z < zmin - tolerance or z > zmax + tolerance)

        if outside:
            result_points.append(point)
        else:
            # Point is inside or on boundary - check if it's a cube_point
            is_cube_point = False
            for cube_point in cube_points:
                if np.linalg.norm(point - cube_point) < tolerance:
                    is_cube_point = True
                    break

            if is_cube_point:
                result_points.append(point)
            # If inside/boundary and not cube_point, don't add (remove it)

    return np.array(result_points) if result_points else np.array([]).reshape(0, 3)


# Create structured mesh points
fixed_points = create_cube_filled_points(box_xmin, box_xmax, box_ymin, box_ymax, box_zmin, box_zmax, h, h, h)

# Generate 3D mesh
print("Generating mesh...")
points, cells = SeismicMesh.generation.generate_mesh(
    domain=u_shape_sdf3D,
    edge_length=h,
    bbox=bbox,
    max_iter=50,
    pfix=fixed_points,
    subdomains=[cube1]
)

# Remove points that are not from the structured mesh
points = remove_cube_interior_points(points, fixed_points, box_xmin, box_xmax, box_ymin, box_ymax, box_zmin, box_zmax)

# Remove points too close from cube
points = remove_points_near_cube(points, box_xmin, box_xmax, box_ymin, box_ymax, box_zmin, box_zmax, h/2, tolerance=1e-10)

# Create new Delaunay with structured mesh part
fixed_triangulation = Delaunay(points)
cells = fixed_triangulation.simplices

# SeismicMesh filter for boundary triangles
cells = _remove_triangles_outside(points, cells, u_shape_sdf3D, 0.1*h)  # in case of too much boundary filter, change this parameter

print(f"Mesh generated with {len(points)} points and {len(cells)} tetrahedra")

# Save mesh
meshio.write_points_cells(
    "u_shape.vtk",
    points,
    [("tetra", cells)],
    file_format="vtk"
)
print("Mesh saved to u_shape.vtk")
