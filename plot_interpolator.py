import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import spyro
import firedrake as fire
from RectanglePhysicalGroupNoBorder import vp_to_sizing

def plot_interpolator():
    """
    Plot the velocity field and corresponding mesh sizing from the interpolator
    """
    # Recreate the same setup as in make_vp_into_numpy.py
    input_mesh_parameters = {
        "length_z": 2.0,
        "length_x": 2.0,
        "dimension": 2,
        "mesh_type": "firedrake_mesh",
        "output_filename": "trial.msh",
        "edge_length": 0.05,
    }

    mesh_parameters = spyro.meshing.MeshingParameters(input_mesh_dictionary=input_mesh_parameters)
    meshing_obj = spyro.meshing.AutomaticMesh(mesh_parameters)
    mesh_ho = meshing_obj.create_mesh()
    x, y = fire.SpatialCoordinate(mesh_ho)

    # Create the velocity field with circular anomaly
    V_ho = fire.FunctionSpace(mesh_ho, "KMV", 4)
    x_c = -1.0
    y_c = 1.0
    r_c = 0.5
    cond = fire.conditional((x-x_c)**2 + (y-y_c)**2 < r_c**2, 3.0, 1.5)
    u_ho = fire.Function(V_ho)
    u_ho.interpolate(cond)

    # Create regular grid
    grid_spacing = 0.01
    input_mesh_parameters_regular = {
        "length_z": 2.0,
        "length_x": 2.0,
        "dimension": 2,
        "mesh_type": "firedrake_mesh",
        "output_filename": "trial.vtk",
        "edge_length": grid_spacing,
    }

    mesh_parameters_regular = spyro.meshing.MeshingParameters(input_mesh_dictionary=input_mesh_parameters_regular)
    meshing_obj_regular = spyro.meshing.AutomaticMesh(mesh_parameters_regular)
    mesh = meshing_obj_regular.create_mesh()

    V = fire.FunctionSpace(mesh, "CG", 1)
    u = fire.Function(V).interpolate(u_ho, allow_missing_dofs=True)

    # Convert to numpy grid
    z = spyro.io.write_function_to_grid(u, V, grid_spacing, buffer=True)
    grid_data = [z, grid_spacing, grid_spacing]

    # Extract parameters for interpolator setup
    length_z = 2.0
    length_x = 2.0
    cpw = 2.7
    frequency = 5.0
    
    # Set up the interpolator exactly as in the original code
    vp = grid_data[0]
    nz, nx = vp.shape
    z_grid = np.linspace(-length_z, 0.0, nz, dtype=np.float32)
    x_grid = np.linspace(0.0, length_x, nx, dtype=np.float32)
    cell_sizes = vp_to_sizing(vp, cpw, frequency)
    interpolator = RegularGridInterpolator(
        (z_grid, x_grid), cell_sizes, bounds_error=False
    )

    print(f"Velocity field shape: {vp.shape}")
    print(f"Z range: {z_grid.min():.3f} to {z_grid.max():.3f}")
    print(f"X range: {x_grid.min():.3f} to {x_grid.max():.3f}")
    print(f"Velocity range: {vp.min():.3f} to {vp.max():.3f}")
    print(f"Cell size range: {cell_sizes.min():.6f} to {cell_sizes.max():.6f}")

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

    # Add circle outline to show the anomaly
    circle = plt.Circle((y_c, x_c), r_c, fill=False, color='red', linewidth=2, linestyle='--')
    axes[0,0].add_patch(circle)

    # Plot 2: Cell sizes from vp_to_sizing
    im2 = axes[0,1].contourf(X_plot, Z_plot, cell_sizes, levels=20, cmap='plasma')
    axes[0,1].set_title('Cell Sizes (from vp_to_sizing)')
    axes[0,1].set_xlabel('X (m)')
    axes[0,1].set_ylabel('Z (m)')
    axes[0,1].set_aspect('equal')
    cbar2 = plt.colorbar(im2, ax=axes[0,1])
    cbar2.set_label('Cell Size (m)')

    # Add circle outline
    circle2 = plt.Circle((y_c, x_c), r_c, fill=False, color='red', linewidth=2, linestyle='--')
    axes[0,1].add_patch(circle2)

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

    # Add circle outline
    circle3 = plt.Circle((y_c, x_c), r_c, fill=False, color='red', linewidth=2, linestyle='--')
    axes[1,0].add_patch(circle3)

    # Plot 4: Cross-section comparison
    # Take a cross-section at x = 1.0 (middle of domain)
    x_cross = 1.0
    x_idx = np.argmin(np.abs(x_grid - x_cross))
    x_idx_test = np.argmin(np.abs(x_test - x_cross))
    
    axes[1,1].plot(z_grid, vp[:, x_idx], 'o-', label=f'Original vp at x={x_cross}', linewidth=2, markersize=4)
    axes[1,1].plot(z_grid, cell_sizes[:, x_idx], 's-', label=f'Cell sizes at x={x_cross}', linewidth=2, markersize=4)
    axes[1,1].plot(z_test, interp_values[:, x_idx_test], '--', label=f'Interpolated at x={x_cross}', linewidth=2)
    
    axes[1,1].set_xlabel('Z (m)')
    axes[1,1].set_ylabel('Value')
    axes[1,1].set_title(f'Cross-section at X = {x_cross} m')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/workspaces/fwi_tutorial/interpolator_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print some statistics about the interpolator
    print("\n=== Interpolator Statistics ===")
    print(f"Grid spacing: {grid_spacing}")
    print(f"CPW (cells per wavelength): {cpw}")
    print(f"Frequency: {frequency} Hz")
    print(f"Formula: cell_size = vp / (frequency * cpw)")
    print(f"Expected cell size for vp=1.5: {1.5/(frequency*cpw):.6f}")
    print(f"Expected cell size for vp=3.0: {3.0/(frequency*cpw):.6f}")
    
    return interpolator, vp, cell_sizes, z_grid, x_grid

if __name__ == "__main__":
    interpolator, vp, cell_sizes, z_grid, x_grid = plot_interpolator()