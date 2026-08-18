import numpy as np
from scipy.interpolate import RegularGridInterpolator
from ..io.parallelism_wrappers import run_in_one_core
from .meshing_utils import check_gmsh, vp_to_sizing

try:
    import gmsh
except ImportError:
    gmsh = None


@run_in_one_core
@check_gmsh
def build_big_rect_with_inner_element_group(mesh_parameters):
    """
    Build a rectangular mesh with optional inner element group using GMSH.

    Parameters
    ----------
    mesh_parameters : object
        Object containing mesh parameters with the following attributes:
        - length_z : float
            Length of domain in z direction.
        - length_x : float
            Length of domain in x direction.
        - output_filename : str
            Path for output mesh file.
        - edge_length : float, optional
            Uniform edge length(if grid_velocity_data is None).
        - grid_velocity_data : dict, optional
            Dictionary with 'vp_values' key containing velocity field for
            adaptive meshing.
        - source_frequency : float, optional
            Source frequency for adaptive meshing.
        - cells_per_wavelength : float, optional
            Cells per wavelength for adaptive meshing.
        - gradient_mask : dict, optional
            Dictionary with keys 'z_min', 'z_max', 'x_min', 'x_max' defining
            a rectangular region to be marked as an inner element group.

    Returns
    -------
    None
        Writes mesh to file specified by mesh_parameters.output_filename.

    Notes
    -----
    This function creates a 2D rectangular mesh with:
    - Adaptive sizing based on velocity field(if provided)
    - Physical boundary groups for absorbing boundary conditions
      (1=Top, 2=Bottom, 3=Right, 4=Left)
    - Optional inner element group for gradient masking

    The function uses GMSH for mesh generation and supports velocity-based
    mesh refinement through interpolation of a regular grid velocity model.
    If a gradient_mask is provided, elements within the specified region
    are separated into an 'Inner' physical group, while elements outside
    are placed in an 'Outer' physical group.
    """
    if gmsh is None:
        raise ImportError("gmsh is not available. Please install it to use this function.")

    length_z = mesh_parameters.length_z
    length_x = mesh_parameters.length_x
    outfile = mesh_parameters.output_filename

    gmsh.initialize()
    gmsh.model.add("BigRect_InnerElements")

    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    # --- Geometry: on length_x the big rectangle ---
    surf_tag = gmsh.model.occ.addRectangle(-length_z, 0.0, 0.0, length_z, length_x)
    gmsh.model.occ.synchronize()

    # Get boundary edges for tagging
    boundary_entities = gmsh.model.getBoundary([(2, surf_tag)], oriented=False)
    edge_tags = [abs(entity[1]) for entity in boundary_entities if entity[0] == 1]

    # Identify boundary edges by their geometric center
    boundary_tag_map = {}
    for edge_tag in edge_tags:
        # Get center of mass of the edge
        com = gmsh.model.occ.getCenterOfMass(1, edge_tag)
        x_center, y_center = com[0], com[1]

        # Classify edges based on position
        # Top edge: z ≈ 0
        if abs(x_center - 0.0) < 1e-10:
            boundary_tag_map[edge_tag] = 1  # Top boundary
        # Bottom edge: z ≈ -length_z
        elif abs(x_center - (-length_z)) < 1e-10:
            boundary_tag_map[edge_tag] = 2  # Bottom boundary
        # Right edge: y ≈ length_x
        elif abs(y_center - length_x) < 1e-10:
            boundary_tag_map[edge_tag] = 3  # Right boundary
        # Left edge: y ≈ 0
        elif abs(y_center - 0.0) < 1e-10:
            boundary_tag_map[edge_tag] = 4  # Left boundary

    if mesh_parameters.grid_velocity_data is None:
        h_min = mesh_parameters.edge_length
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(h_min))

        def mesh_size_callback(dim, tag, x, y, z, lc):
            h = x + y
            return float(h)
    else:
        frequency = mesh_parameters.source_frequency
        cpw = mesh_parameters.cells_per_wavelength
        vp = mesh_parameters.grid_velocity_data["vp_values"]
        nz, nx = vp.shape
        z_grid = np.linspace(-length_z, 0.0, nz, dtype=np.float32)
        x_grid = np.linspace(0.0, length_x, nx, dtype=np.float32)
        cell_sizes = vp_to_sizing(vp, cpw, frequency)
        interpolator = RegularGridInterpolator(
            (z_grid, x_grid), cell_sizes, bounds_error=False
        )
        gmsh.model.occ.synchronize()

        def mesh_size_callback(dim, tag, x, y, z, lc):
            size = float(interpolator([[x, y]])[0])
            return size
    gmsh.model.mesh.setSizeCallback(mesh_size_callback)

    # --- Mesh the single surface ---
    gmsh.model.mesh.generate(2)
    # gmsh.fltk.run()

    # --- Collect elements & classify by centroid into inner/outer ---
    if mesh_parameters.gradient_mask is not None:
        # Get elements of the (onlength_x) geometric surface
        types, elemTags, nodeTags = gmsh.model.mesh.getElements(2, surf_tag)
        # Build a node coordinate map
        node_ids, node_xyz, _ = gmsh.model.mesh.getNodes()
        # node_ids may be unsorted; build dict id->(x,y,z)
        coords = np.array(node_xyz).reshape(-1, 3)
        id2idx = {int(nid): i for i, nid in enumerate(node_ids)}

        # Small rectangle bounds
        z_min = mesh_parameters.gradient_mask["z_min"]
        z_max = mesh_parameters.gradient_mask["z_max"]
        x_min = mesh_parameters.gradient_mask["x_min"]
        x_max = mesh_parameters.gradient_mask["x_max"]

        # Prepare per-type lists for inner/outer
        inner_elem_by_type = []
        inner_conn_by_type = []
        outer_elem_by_type = []
        outer_conn_by_type = []

        for t, tags, conn in zip(types, elemTags, nodeTags):
            tags = np.array(tags, dtype=np.int64)
            # number of nodes per element type t:
            nPer = gmsh.model.mesh.getElementProperties(t)[3]  # returns (name, dim, order, numNodes, ...)[3]
            conn = np.array(conn, dtype=np.int64).reshape(-1, nPer)

            # Compute centroids
            # (x_c, y_c) = average of node coords
            xyz = coords[[id2idx[int(n)] for n in conn.flatten()]].reshape(-1, nPer, 3)
            centroids = xyz.mean(axis=1)  # (nelem, 3)
            cz_e = centroids[:, 0]
            cx_e = centroids[:, 1]

            inside = (cz_e >= z_min) & (cz_e <= z_max) & (cx_e >= x_min) & (cx_e <= x_max)

            inner_elem_by_type.append(tags[inside])
            inner_conn_by_type.append(conn[inside])

            outer_elem_by_type.append(tags[~inside])
            outer_conn_by_type.append(conn[~inside])

        # --- Move inner elements to a DISCRETE surface entity ---
        # 1) Remove ALL elements from the original geometric surface
        gmsh.model.mesh.removeElements(2, surf_tag)

        # 2) Re-add the outer elements back to the geometric surface
        for t, tags, conn in zip(types, outer_elem_by_type, outer_conn_by_type):
            if tags.size == 0:
                continue
            gmsh.model.mesh.addElements(2, surf_tag, [t], [tags.tolist()], [conn.flatten().tolist()])

        # 3) Create a discrete surface and add the inner elements
        inner_surf_tag = gmsh.model.addDiscreteEntity(2)
        # Pure element set container.
        for t, tags, conn in zip(types, inner_elem_by_type, inner_conn_by_type):
            if tags.size == 0:
                continue
            gmsh.model.mesh.addElements(2, inner_surf_tag, [t], [tags.tolist()], [conn.flatten().tolist()])

        # --- Define Physical groups on entities ---
        pg_outer = gmsh.model.addPhysicalGroup(2, [surf_tag])
        gmsh.model.setPhysicalName(2, pg_outer, "Outer")
        pg_inner = gmsh.model.addPhysicalGroup(2, [inner_surf_tag])
        gmsh.model.setPhysicalName(2, pg_inner, "Inner")

    # Create physical groups for boundary edges (for absorbing boundary conditions)
    for edge_tag, boundary_id in boundary_tag_map.items():
        # Set physical group ID explicitly to match ds() tags (1=top, 2=bottom, 3=right, 4=left)
        pg_boundary = gmsh.model.addPhysicalGroup(1, [edge_tag], boundary_id)  # noqa: F841
        boundary_names = {1: "Top", 2: "Bottom", 3: "Right", 4: "Left"}
        gmsh.model.setPhysicalName(1, boundary_id, f"Boundary_{boundary_names.get(boundary_id, boundary_id)}")
        # This ensures ds(1), ds(2), ds(3), ds(4) work correctly

    # Save
    gmsh.write(outfile)

    print(f"Written mesh to: {outfile}")
    print(f"Boundary tags created: {len(boundary_tag_map)} edges")
    for edge_tag, boundary_id in boundary_tag_map.items():
        boundary_names = {1: "Top", 2: "Bottom", 3: "Right", 4: "Left"}
        print(f"  Edge {edge_tag} -> Boundary {boundary_id} ({boundary_names.get(boundary_id, 'Unknown')})")

    if mesh_parameters.gradient_mask is not None:
        print(f"Geometric surface tag      (Outer): {surf_tag}")
        print(f"Discrete surface tag       (Inner): {inner_surf_tag}")
        print(f"Physical group Outer tag: {pg_outer}")
        print(f"Physical group Inner tag: {pg_inner}")

    gmsh.finalize()
