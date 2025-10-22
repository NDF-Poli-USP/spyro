import os
import sys
import gmsh
import numpy as np

scale = 1.0
# Element Size
h = 0.15*scale  # Cube element size
h_hyperShape = h
# Cube domain inside hyperellipsoid
domainX = 1.0*scale  # x size of cubic domain
domainY = 1.0*scale  # y size of cubic domain
domainZ = 1.0*scale  # z size of cubic domain
# HyperEllipsoid Padding
ellipseLx = 0.5*scale  # ellipseLx = Padding from cube domain x side
ellipseLy = 0.5*scale  # ellipseLy = Padding from cube domain y side
ellipseLz = 0.5*scale  # ellipseLz = Padding from cube domain z side
# HyperEllipsoid exponent
ellipse_n = 3
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
# Calculating parameters for hyperellipsoid with center at center of cube
ellipse_a = domainX/2 + ellipseLx  # x semi-axis
ellipse_b = domainY/2 + ellipseLy  # y semi-axis
ellipse_c = domainZ/2 + ellipseLz  # z semi-axis
xc = domainX/2
yc = domainY/2
zc = -domainZ/2
# Seismic mesh needs a box that envelopes all the domain
# the box is calculated here
domain_xmin = xc - ellipse_a
domain_xmax = xc + ellipse_a
domain_ymin = yc - ellipse_b
domain_ymax = yc + ellipse_b
domain_zmin = zc - ellipse_c
domain_zmax = zc + ellipse_c
bbox = (domain_xmin, domain_xmax, domain_ymin,
        domain_ymax, domain_zmin, domain_zmax)


def create_super_ellipsoid_volume(a=1.0, b=1.0, c=1.0, n=2.0, xc=0.0, yc=0.0, zc=0.0):
    """
    Create a 3D super ellipsoid volume using OpenCASCADE B-spline surfaces.

    Parameters:
    -----------
    a, b, c : float
        Semi-axes lengths in x, y, z directions
    n : float
        Exponent parameter 
    xc, yc, zc : float
        Center coordinates of the ellipsoid

    Returns:
    --------
    volume_tag : int
        OpenCASCADE volume tag
    """

    # Ensure gmsh is properly initialized
    try:
        gmsh.finalize()
    except:
        pass

    gmsh.initialize()
    gmsh.clear()

    # Create a new model using OpenCASCADE kernel
    gmsh.model.add("super_ellipsoid_occ")

    def super_ellipsoid_point(u, v, a, b, c, n, xc, yc, zc):
        """
        Calculate a point on the super ellipsoid surface.
        u: longitude parameter [0, 2π]
        v: latitude parameter [-π/2, π/2]
        xc, yc, zc: center coordinates
        """
        # Sign function that preserves the sign
        def sign_power(x, p):
            if abs(x) < 1e-10:
                return 0.0
            return np.sign(x) * (np.abs(x) ** p)

        cos_v = np.cos(v)
        sin_v = np.sin(v)
        cos_u = np.cos(u)
        sin_u = np.sin(u)

        # Calculate point relative to origin
        x = a * sign_power(cos_v, 2.0/n) * sign_power(cos_u, 2.0/n)
        y = b * sign_power(cos_v, 2.0/n) * sign_power(sin_u, 2.0/n)
        z = c * sign_power(sin_v, 2.0/n)

        # Translate to center coordinates
        x += xc
        y += yc
        z += zc

        return x, y, z

    volume_tag = None

    try:
        volume_tag = create_closed_surface(
            a, b, c, n, xc, yc, zc, super_ellipsoid_point)
        if volume_tag:
            print("Successfully created volume using closed B-spline surface")
    except Exception as e:
        print(f"Closed B-spline failed: {e}")

    return volume_tag


def create_closed_surface(a, b, c, n, xc, yc, zc, point_func, u_res=92, v_res=93):
    """
    Create a closed B-spline surface for hypershape.
    """
    # Generate point grid
    point_tags = []

    # Create full parametric grid including closure point and poles
    for j in range(v_res):
        for i in range(u_res + 1):  # +1 to include closure point at u=2π
            # Handle u-direction closure: last point same as first
            if i == u_res:
                u = 0.0  # Close the loop
            else:
                u = 2 * np.pi * i / u_res

            # Handle v direction INCLUDING exact poles
            v = np.pi * (j / (v_res - 1) - 0.5)  # From -π/2 to π/2

            # At poles, all u values should give the same point
            if j == 0:  # South pole
                x, y, z = point_func(0, -np.pi/2, a, b, c, n, xc, yc, zc)
            elif j == v_res - 1:  # North pole
                x, y, z = point_func(0, np.pi/2, a, b, c, n, xc, yc, zc)
            else:
                x, y, z = point_func(u, v, a, b, c, n, xc, yc, zc)

            point_tag = gmsh.model.occ.addPoint(x, y, z)
            point_tags.append(point_tag)

    try:
        # Create B-spline surface
        surface_tag = gmsh.model.occ.addBSplineSurface(
            pointTags=point_tags,
            numPointsU=u_res + 1,  # Include closure point
            tag=-1,
            degreeU=min(3, u_res),
            degreeV=min(3, v_res-1)
        )

        gmsh.model.occ.synchronize()

        # Create volume
        surface_loop = gmsh.model.occ.addSurfaceLoop([surface_tag])
        volume_tag = gmsh.model.occ.addVolume([surface_loop])
        gmsh.model.occ.synchronize()
        z_cut = 0.0
        if volume_tag is not None and z_cut is not None:
            print(f"Applying z-cut at z = {z_cut}")

            # Create cutting box above z_cut
            domain_size = 2 * max(a, b, c)
            cutting_box = gmsh.model.occ.addBox(
                xc - domain_size, yc - domain_size, z_cut,
                2*domain_size, 2*domain_size, domain_size
            )

            gmsh.model.occ.synchronize()

            # Remove everything above z_cut
            result = gmsh.model.occ.cut(
                [(3, volume_tag)], [(3, cutting_box)],
                removeObject=True, removeTool=True
            )

            if result[0]:
                volume_tag = result[0][0][1]
                print(f"Z-cut applied successfully")
            else:
                print("Warning: Z-cut removed entire volume")
                volume_tag = None

        return volume_tag

    except Exception as e:
        print(f"B-spline surface creation failed: {e}")
        return None


def report_quality(dim=3, quality_type=2):
    """
    dim: 2 (surface) or 3 (volume)
    quality_type controls the metric Gmsh uses, e.g.:
      0=gamma (vol/sum_face/max_edge), 1=eta (vol^(2/3)/sum_edge^2), 2=rho (min_edge/max_edge)
    """
    gmsh.option.setNumber("Mesh.QualityType", quality_type)  # choose metric

    # Grab all elements of this dimension
    elem_types, elem_tags, node_tags = gmsh.model.mesh.getElements(
        dim)  # returns per-type lists

    # Flatten to a single list of element tags
    all_tags = []
    for tags in elem_tags:
        all_tags.extend(tags.tolist() if hasattr(
            tags, "tolist") else list(tags))

    if not all_tags:
        print(f"[quality] No elements found for dim={dim}")
        return

    # Compute qualities for elements
    q = gmsh.model.mesh.getElementQualities(all_tags)

    q = np.asarray(q, dtype=float)
    print(f"[quality] count={q.size}  min={q.min():.6g}  p1={np.percentile(q,1):.6g}  "
          f"p5={np.percentile(q,5):.6g}  median={np.median(q):.6g}  "
          f"p95={np.percentile(q,95):.6g}  max={q.max():.6g}  mean={q.mean():.6g}")


# Initialize GMSH
gmsh.initialize()
gmsh.model.add("superellipsoid_cube_fragment")

print(f"Creating superellipsoid with n={ellipse_n}")

# Create superellipsoid volume
ellipsoid_volume_tag = create_super_ellipsoid_volume(a=ellipse_a, b=ellipse_b, c=ellipse_c, n=ellipse_n,
                                                     xc=xc, yc=yc, zc=zc)

# Create cube volume
cube_volume_tag = gmsh.model.occ.addBox(
    box_xmin, box_ymin, box_zmin,  # x, y, z of corner
    box_xmax - box_xmin,           # width in x
    box_ymax - box_ymin,           # width in y
    box_zmax - box_zmin            # width in z
)

gmsh.model.occ.synchronize()

# Structured mesh for cube only
# Compute number of divisions along each axis based on edge size and h
nx = max(1, int(round((box_xmax - box_xmin) / h)))
ny = max(1, int(round((box_ymax - box_ymin) / h)))
nz = max(1, int(round((box_zmax - box_zmin) / h)))

print(f"Cube divisions: nx={nx}, ny={ny}, nz={nz}")

# Get cube surfaces
cube_surfaces = gmsh.model.getBoundary(
    [(3, cube_volume_tag)], oriented=False, recursive=False)
cube_surfaces = [s[1] for s in cube_surfaces if s[0] == 2]

# Apply transfinite meshing to cube edges and surfaces
for s in cube_surfaces:
    edges = gmsh.model.getBoundary([(2, s)], oriented=False, recursive=False)
    for e in edges:
        if e[0] == 1:  # line entity
            # Get curve bounding box to detect direction
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(
                1, e[1])
            dx, dy, dz = abs(xmax - xmin), abs(ymax - ymin), abs(zmax - zmin)

            if dx > dy and dx > dz:
                gmsh.model.mesh.setTransfiniteCurve(e[1], nx+1)
            elif dy > dx and dy > dz:
                gmsh.model.mesh.setTransfiniteCurve(e[1], ny+1)
            else:
                gmsh.model.mesh.setTransfiniteCurve(e[1], nz+1)

    gmsh.model.mesh.setTransfiniteSurface(s)

# Apply transfinite volume
gmsh.model.mesh.setTransfiniteVolume(cube_volume_tag)

gmsh.model.occ.synchronize()

# Fragment the geometries
if ellipsoid_volume_tag and cube_volume_tag:
    print("Fragmenting cube and superellipsoid...")

    # Fragment operation - create proper subdomain
    fragment_result = gmsh.model.occ.fragment(
        [(3, ellipsoid_volume_tag), (3, cube_volume_tag)],  # Object volumes
        [],  # (empty for self-fragmentation)
        removeObject=True,
        removeTool=False
    )

    gmsh.model.occ.synchronize()

    volumes = gmsh.model.getEntities(3)
    volume_tags = [tag for dim, tag in volumes if dim == 3]

    cube_physical = gmsh.model.addPhysicalGroup(
        3, [volume_tags[0]], name="Cube")
    ellipsoid_physical = gmsh.model.addPhysicalGroup(
        3, [volume_tags[1]], name="Ellipsoid")

    # Set mesh size in the hypershape
    gmsh.model.mesh.setSize(gmsh.model.getBoundary(
        [(3, volume_tags[1])], oriented=False, recursive=True), h_hyperShape)

    # Remove unused points
    gmsh.option.setNumber("Mesh.Algorithm", 1)
    gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.5)
    gmsh.option.setNumber("Mesh.Smoothing", 100)
    gmsh.option.setNumber("Mesh.SaveWithoutOrphans", 1)
    # built-in optimizer (tets)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    # Generate 3D mesh
    gmsh.model.mesh.generate(3)

    report_quality(dim=3, quality_type=2)   # rho (min_edge/max_edge) for tets

    # Get mesh info
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements(
        3)

    print(f"Generated mesh with {len(node_tags)} nodes")
    if element_tags:
        total_elements = sum(len(tags) for tags in element_tags)
        print(f"Generated {total_elements} volume elements")

    # Save mesh files
    gmsh.write("hypershape_mesh.msh")
    gmsh.clear()

    gmsh.open("hypershape_mesh.msh")
    gmsh.write("hypershape_mesh.vtk")
    print("Mesh files saved: hypershape_mesh.msh, hypershape_mesh.vtk")

else:
    print("Error: Could not create geometries")

gmsh.finalize()
print("Geometry creation and fragmentation completed!")
