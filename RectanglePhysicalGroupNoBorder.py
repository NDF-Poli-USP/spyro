import gmsh
# import firedrake as fire
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


def vp_to_sizing(vp, cpw, frequency):
    if cpw < 0.0 or cpw == 0.0:
        raise ValueError(f"Cells-per-wavelength value of {cpw} not supported.")
    if frequency < 0.0 or frequency == 0.0:
        raise ValueError(f"Frequency must be positive and non zero")

    return vp / (frequency * cpw)


def plot_interpolator_data(grid_data, length_z, length_x, cpw, frequency, save_plot=True):
    """
    Plot the velocity field and interpolator data for visualization
    
    Parameters:
    -----------
    grid_data : list
        Contains [vp_array, grid_spacing_z, grid_spacing_x]
    length_z : float
        Domain length in z direction
    length_x : float  
        Domain length in x direction
    cpw : float
        Cells per wavelength
    frequency : float
        Frequency in Hz
    save_plot : bool
        Whether to save the plot to file
        
    Returns:
    --------
    interpolator : RegularGridInterpolator
        The created interpolator object
    """
    vp = grid_data["vp_values"]
    nz, nx = vp.shape
    z_grid = np.linspace(-length_z, 0.0, nz, dtype=np.float32)
    x_grid = np.linspace(0.0, length_x, nx, dtype=np.float32)
    cell_sizes = vp_to_sizing(vp, cpw, frequency)
    interpolator = RegularGridInterpolator(
        (z_grid, x_grid), cell_sizes, bounds_error=False
    )
    
    # Create meshgrids for plotting
    Z_plot, X_plot = np.meshgrid(z_grid, x_grid, indexing='ij')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Interpolator Visualization', fontsize=16)

    # Plot 1: Original velocity field
    im1 = axes[0,0].contourf(X_plot, Z_plot, vp, levels=20, cmap='viridis')
    axes[0,0].set_title('Velocity Field (vp)')
    axes[0,0].set_xlabel('X (m)')
    axes[0,0].set_ylabel('Z (m)')
    axes[0,0].set_aspect('equal')
    cbar1 = plt.colorbar(im1, ax=axes[0,0])
    cbar1.set_label('Velocity (m/s)')

    # Plot 2: Cell sizes from vp_to_sizing
    im2 = axes[0,1].contourf(X_plot, Z_plot, cell_sizes, levels=20, cmap='plasma')
    axes[0,1].set_title('Cell Sizes (from vp_to_sizing)')
    axes[0,1].set_xlabel('X (m)') 
    axes[0,1].set_ylabel('Z (m)')
    axes[0,1].set_aspect('equal')
    cbar2 = plt.colorbar(im2, ax=axes[0,1])
    cbar2.set_label('Cell Size (m)')

    # Plot 3: Test interpolator on a finer grid
    z_test = np.linspace(-length_z, 0.0, 300)
    x_test = np.linspace(0.0, length_x, 300)
    Z_test, X_test = np.meshgrid(z_test, x_test, indexing='ij')
    
    # Flatten for interpolation
    points = np.column_stack([Z_test.ravel(), X_test.ravel()])
    interp_values = interpolator(points).reshape(Z_test.shape)
    
    im3 = axes[1,0].contourf(X_test, Z_test, interp_values, levels=20, cmap='plasma')
    axes[1,0].set_title('Interpolated Cell Sizes (Fine Grid)')
    axes[1,0].set_xlabel('X (m)')
    axes[1,0].set_ylabel('Z (m)')
    axes[1,0].set_aspect('equal')
    cbar3 = plt.colorbar(im3, ax=axes[1,0])
    cbar3.set_label('Cell Size (m)')

    # Plot 4: Cross-section comparison
    x_cross = length_x / 2.0  # Middle of domain
    x_idx = np.argmin(np.abs(x_grid - x_cross))
    x_idx_test = np.argmin(np.abs(x_test - x_cross))
    
    axes[1,1].plot(z_grid, vp[:, x_idx], 'o-', label=f'Original vp at x={x_cross:.1f}', linewidth=2, markersize=4)
    axes[1,1].plot(z_grid, cell_sizes[:, x_idx], 's-', label=f'Cell sizes at x={x_cross:.1f}', linewidth=2, markersize=4)
    axes[1,1].plot(z_test, interp_values[:, x_idx_test], '--', label=f'Interpolated at x={x_cross:.1f}', linewidth=2)
    
    axes[1,1].set_xlabel('Z (m)')
    axes[1,1].set_ylabel('Value')
    axes[1,1].set_title(f'Cross-section at X = {x_cross:.1f} m')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_plot:
        plt.savefig('interpolator_visualization.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'interpolator_visualization.png'")
    
    plt.show()

    # Print statistics
    print(f"Velocity field shape: {vp.shape}")
    print(f"Z range: {z_grid.min():.3f} to {z_grid.max():.3f}")
    print(f"X range: {x_grid.min():.3f} to {x_grid.max():.3f}")
    print(f"Velocity range: {vp.min():.3f} to {vp.max():.3f}")
    print(f"Cell size range: {cell_sizes.min():.6f} to {cell_sizes.max():.6f}")
    
    return interpolator


def build_big_rect_with_inner_element_group(mesh_parameters, plot_interpolator=True):

    length_z = mesh_parameters.length_z
    length_x = mesh_parameters.length_x
    outfile = mesh_parameters.output_filename

    gmsh.initialize()
    gmsh.model.add("BigRect_InnerElements")

    # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(h_min))
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    # --- Geometry: onlength_x the big rectangle ---
    surf_tag = gmsh.model.occ.addRectangle(-length_z, 0.0, 0.0, length_z, length_x)
    gmsh.model.occ.synchronize()

    if mesh_parameters.grid_velocity_data is None:
        h_min = mesh_parameters.edge_length
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(h_min))
        def mesh_size_callback(dim, tag, x, y, z, lc):
            # In case of interpolated function
            #coords = np.array([[z, x, y]])
            #element_size = interpolated_function(coords)[0] 
            #return float(element_size)
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

        # Plot interpolator if requested
        if plot_interpolator:
            plot_interpolator_data(mesh_parameters.grid_velocity_data, length_z, length_x, cpw, frequency)

        def mesh_size_callback(dim, tag, x, y, z, lc):
            size = float(interpolator([[x, y]])[0])
            # if size < h_min: size = h_min
            if size > 0.15:
                print(f"for point ({x}, {y}) we have size of {size}")
                print("DBUG")
            return size
    gmsh.model.mesh.setSizeCallback(mesh_size_callback)

    # --- Mesh the single surface ---
    gmsh.model.mesh.generate(2)
    gmsh.fltk.run()

    # --- Collect elements & classify by centroid into inner/outer ---
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
    pg_outer = gmsh.model.addPhysicalGroup(2, [surf_tag]);        gmsh.model.setPhysicalName(2, pg_outer, "Outer")
    pg_inner = gmsh.model.addPhysicalGroup(2, [inner_surf_tag]);   gmsh.model.setPhysicalName(2, pg_inner, "Inner")

    # Save
    gmsh.write(outfile)

    print(f"Written mesh to: {outfile}")
    print(f"Geometric surface tag      (Outer): {surf_tag}")
    print(f"Discrete surface tag       (Inner): {inner_surf_tag}")
    print(f"Physical group Outer tag: {pg_outer}")
    print(f"Physical group Inner tag: {pg_inner}")

    #gmsh.fltk.run()
    gmsh.finalize()


# if __name__ == "__main__":

#     input_mesh_parameters = {
#         "length_z": 2.0,
#         "length_x": 2.0,
#         "mesh_type": "spyro_mesh",
#         "output_filename": "two_rects_nogeom.vtk",
#         "edge_length": 0.05,
#     }

#     mesh_parameters = spyro.meshing.MeshingParameters(input_mesh_dictionary=input_mesh_parameters)
#     mask_boundaries = {
#         "z_min": -1.3,
#         "z_max": -0.7,
#         "x_min": 0.7,
#         "x_max": 1.3,
#     }
#     build_big_rect_with_inner_element_group(mesh_parameters, mask_boundaries)

    # mesh = fire.Mesh(
    #     outfile, 
    #     distribution_parameters={"overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)},
    #     )
    # V = fire.FunctionSpace(mesh, "CG", 1)
    # u = fire.Function(V)
    # output = fire.VTKFile("debug.pvd")
    # output.write(u)
