import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from RectanglePhysicalGroupNoBorder import vp_to_sizing

def plot_interpolator_simple(grid_data, length_z=2.0, length_x=2.0, cpw=2.7, frequency=5.0):
    """
    Simple function to plot the interpolator from existing grid_data
    """
    # Extract velocity data
    vp = grid_data[0]
    nz, nx = vp.shape
    
    # Create grid coordinates (matching the original code)
    z_grid = np.linspace(-length_z, 0.0, nz, dtype=np.float32)
    x_grid = np.linspace(0.0, length_x, nx, dtype=np.float32)
    
    # Calculate cell sizes
    cell_sizes = vp_to_sizing(vp, cpw, frequency)
    
    # Create the interpolator
    interpolator = RegularGridInterpolator(
        (z_grid, x_grid), cell_sizes, bounds_error=False
    )
    
    # Create meshgrids for plotting
    Z_plot, X_plot = np.meshgrid(z_grid, x_grid, indexing='ij')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Interpolator Visualization from Grid Data', fontsize=16)

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

    # Plot 4: Cross-section comparison at x = 1.0
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
    plt.savefig('/workspaces/fwi_tutorial/interpolator_simple_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print some statistics
    print(f"Velocity field shape: {vp.shape}")
    print(f"Z range: {z_grid.min():.3f} to {z_grid.max():.3f}")
    print(f"X range: {x_grid.min():.3f} to {x_grid.max():.3f}")
    print(f"Velocity range: {vp.min():.3f} to {vp.max():.3f}")
    print(f"Cell size range: {cell_sizes.min():.6f} to {cell_sizes.max():.6f}")
    
    print("\n=== Interpolator Statistics ===")
    print(f"CPW (cells per wavelength): {cpw}")
    print(f"Frequency: {frequency} Hz")
    print(f"Formula: cell_size = vp / (frequency * cpw)")
    print(f"For vp=0.5: cell_size = {0.5/(frequency*cpw):.6f}")
    print(f"For vp=3.0: cell_size = {3.0/(frequency*cpw):.6f}")
    
    return interpolator, vp, cell_sizes, z_grid, x_grid

# Example of how to use this function
if __name__ == "__main__":
    # You would typically call this with actual grid_data
    # Example: plot_interpolator_simple(grid_data)
    print("This script is meant to be imported and used with actual grid_data")
    print("Example usage:")
    print("from plot_interpolator_simple import plot_interpolator_simple")
    print("interpolator, vp, cell_sizes, z_grid, x_grid = plot_interpolator_simple(grid_data)")