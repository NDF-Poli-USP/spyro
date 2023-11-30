import numpy as np
import h5py
import firedrake as fire
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import RegularGridInterpolator

plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 12


class Velocity_figure:
    def __init__(self, velocity_filename, mesh_filename):
        self.velocity_filename = velocity_filename
        self.mesh = fire.Mesh(mesh_filename)

        # Getting mesh boundaries
        self.minimum_z = None
        self.maximum_z = None
        self.minimum_x = None
        self.maximum_x = None
        self.get_mesh_boundaries()

        # Setting function space
        self.function_space = self.get_function_space()

        # Calculating dof coordinates
        self.dofs_z, self.dofs_x = self.get_dof_coordinates()

        # Interpolating velocity model
        self.velocity = self.interpolate_velocity_model()
    
    def get_mesh_boundaries(self):
        mesh = self.mesh
        self.minimum_z = np.amin(mesh.coordinates.dat.data[:, 0])
        self.maximum_z = np.amax(mesh.coordinates.dat.data[:, 0])
        self.minimum_x = np.amin(mesh.coordinates.dat.data[:, 1])
        self.maximum_x = np.amax(mesh.coordinates.dat.data[:, 1])

    def get_function_space(self):
        return fire.FunctionSpace(self.mesh, "KMV", 5)

    def get_dof_coordinates(self):
        V = self.function_space
        x_mesh, y_mesh = fire.SpatialCoordinate(self.mesh)
        ux = fire.Function(V)
        uy = fire.Function(V)
        ux.interpolate(x_mesh)
        uy.interpolate(y_mesh)
        dofs_z_nonfiltered = ux.dat.data[:]
        dofs_x_nonfiltered = uy.dat.data[:]

        # make sure no out-of-bounds
        minimum_z = self.minimum_z
        maximum_z = self.maximum_z
        minimum_x = self.minimum_x
        maximum_x = self.maximum_x
        dofs_z = [
            minimum_z if z < minimum_z else maximum_z if z > maximum_z else z for z in dofs_z_nonfiltered
        ]
        dofs_x = [
            minimum_x if x < minimum_x else maximum_x if x > maximum_x else x for x in dofs_x_nonfiltered
        ]

        return dofs_z, dofs_x

    def interpolate_velocity_model(self):
        V = self.function_space
        dofs_z = self.dofs_z
        dofs_x = self.dofs_x
        minimum_z = self.minimum_z
        maximum_z = self.maximum_z
        minimum_x = self.minimum_x
        maximum_x = self.maximum_x
        velocity_filename = self.velocity_filename

        with h5py.File(velocity_filename, "r") as f:
            Z = np.asarray(f.get("velocity_model")[()])

            nrow, ncol = Z.shape
            z = np.linspace(minimum_z, maximum_z, nrow)
            x = np.linspace(minimum_x, maximum_x, ncol)

            interpolant = RegularGridInterpolator((z, x), Z)
            tmp = interpolant((dofs_z, dofs_x))
        c = fire.Function(V)
        c.dat.data[:] = tmp
        if min(c.dat.data[:]) > 100.0:
            # data is in m/s but must be in km/s
            if fire.COMM_WORLD.rank == 0:
                print("INFO: converting from m/s to km/s", flush=True)
            c.assign(c / 1000.0)  # meters to kilometers

        return c

    def plot(self, output_filename=None, show=False):
        coordinates = copy.deepcopy(self.mesh.coordinates.dat.data)
        self.mesh.coordinates.dat.data[:, 0] = coordinates[:, 1]
        self.mesh.coordinates.dat.data[:, 1] = coordinates[:, 0]

        fig, axes = plt.subplots()

        im = fire.tripcolor(self.velocity, axes=axes, cmap='coolwarm')
        axes.set_aspect("equal", "box")
        plt.title("Velocity field")

        cbar = fig.colorbar(im, orientation="horizontal")
        cbar.ax.set_xlabel("velocity (km/s)")
        axes.set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.3])

        fig.set_size_inches(13, 10)
        plt.margins(x=0, y=0)

        if output_filename is not None:
            plt.savefig(output_filename)

        if show:
            plt.show()


if __name__ == "__main__":
    velocity_filename = "/media/olender/T7 Shield/common_files/velocity_models/vp_marmousi-ii.hdf5"
    mesh_filename = "automatic_mesh.msh"
    Vel_obj = Velocity_figure(velocity_filename, mesh_filename)
    Vel_obj.plot(output_filename="velocity_model_marmousi.png", show=True)
