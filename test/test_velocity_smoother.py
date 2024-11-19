import firedrake as fire
import spyro
import segyio
import numpy as np
import matplotlib.pyplot as plt


def get_vp_from_2dsegy(filename):
    """
    Extracts velocity profile (vp) data from a 2D SEG-Y file.
    Parameters:
    filename (str): The path to the SEG-Y file.
    Returns:
    np.ndarray: A 2D numpy array containing the velocity profile data,
                with shape (nz, nx) where nz is the number of samples
                and nx is the number of traces.
    """

    with segyio.open(filename, ignore_geometry=True) as f:
        nz, nx = len(f.samples), len(f.trace)
        show_vp = np.zeros(shape=(nz, nx))
        for index, trace in enumerate(f.trace):
            show_vp[:, index] = trace

    return show_vp


def test_write_segy_and_smooth(show=False):
    vp_name = "velocity_models/test"
    segy_file = vp_name + ".segy"
    smoothed_file = "smoothed_test.segy"
    mesh = fire.UnitSquareMesh(50, 50)
    mesh.coordinates.dat.data[:, 0] *= -1

    V = fire.FunctionSpace(mesh, "CG", 3)
    x, y = fire.SpatialCoordinate(mesh)
    r = 0.2
    xc = -0.5
    yc = 0.5

    vp = fire.Function(V)

    c = fire.conditional((x - xc) ** 2 + (y - yc) ** 2 < r**2, 3.0, 1.5)

    vp.interpolate(c)

    xi, yi, zi = spyro.io.write_function_to_grid(vp, V, 10.0 / 1000.0)
    spyro.io.create_segy(zi, segy_file)
    original_vp = get_vp_from_2dsegy(segy_file)

    if show is True:
        fig, ax = plt.subplots()
        plt.pcolormesh(original_vp, shading="auto")
        plt.title("Non smoothed model model")
        plt.colorbar(label="P-wave velocity (km/s)")
        plt.xlabel("x-direction (m)")
        plt.ylabel("z-direction (m)")
        ax.axis("equal")
        plt.show()
        plt.savefig("nonsmoothedtest.png")

    spyro.tools.smooth_velocity_field_file(segy_file, smoothed_file, 5, show=show)

    smoothed_vp = get_vp_from_2dsegy(smoothed_file)
    check_boundary = np.isclose(original_vp[0, 0], smoothed_vp[0, 0])
    check_centre = np.isclose(original_vp[48, 48], smoothed_vp[48, 48], rtol=1e-3)
    check_halfway = original_vp[0, 0]*1.1 < smoothed_vp[24, 48] < original_vp[48, 48]*0.9

    assert all([check_boundary, check_halfway, check_centre])


if __name__ == "__main__":
    test_write_segy_and_smooth(show=True)
