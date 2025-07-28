# from scipy.io import savemat
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import firedrake
import copy
from ..io import ensemble_save

__all__ = ["plot_shots"]


@ensemble_save
def plot_shots(
    Wave_object,
    show=False,
    file_name="plot_of_shot",
    shot_ids=[0],
    vmin=-1e-5,
    vmax=1e-5,
    contour_lines=700,
    file_format="pdf",
    start_index=0,
    end_index=0,
    out_index=None,
):
    """Plot a shot record and save the image to disk. Note that
    this automatically will rename shots when ensmeble paralleism is
    activated.
    Parameters
    ----------
    model: `dictionary`
        Contains model parameters and options.
    comm:A Firedrake commmunicator
        The communicator you get from calling spyro.utils.mpi_init()
    arr: array-like
        An array in which rows are intervals in time and columns are receivers
    show: `boolean`, optional
        Should the images appear on screen?
    file_name: string, optional
        The name of the saved image
    vmin: float, optional
        The minimum value to plot on the colorscale
    vmax: float, optional
        The maximum value to plot on the colorscale
    file_format: string, optional
        File format, pdf or png
    start_index: integer, optional
        The index of the first receiver to plot
    end_index: integer, optional
        The index of the last receiver to plot
    Returns
    -------
    None
    """
    file_name = file_name + str(shot_ids) + "." + file_format
    num_recvs = Wave_object.number_of_receivers

    dt = Wave_object.dt
    tf = Wave_object.final_time

    if out_index is None:
        arr = Wave_object.receivers_output
    else:
        arr = Wave_object.receivers_output[:, :, out_index]

    nt = int(tf / dt) + 1  # number of timesteps

    if end_index == 0:
        end_index = num_recvs

    x_rec = np.linspace(start_index, end_index, num_recvs)
    t_rec = np.linspace(0.0, tf, nt)
    X, Y = np.meshgrid(x_rec, t_rec)

    cmap = plt.get_cmap("gray")
    plt.contourf(X, Y, arr, contour_lines, cmap=cmap, vmin=vmin, vmax=vmax)
    # savemat("test.mat", {"mydata": arr})
    plt.xlabel("receiver number", fontsize=18)
    plt.ylabel("time (s)", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(start_index, end_index)
    plt.ylim(tf, 0)
    plt.subplots_adjust(left=0.18, right=0.95, bottom=0.14, top=0.95)
    plt.savefig(file_name, format=file_format)
    # plt.axis("image")
    if show:
        plt.show()
    # plt.close()
    return None


def plot_mesh_sizes(
    mesh_filename=None,
    firedrake_mesh=None,
    title_str=None,
    output_filename=None,
    show=False,
    show_size_contour=True,
):
    # plt.rcParams['font.family'] = "Times New Roman"
    plt.rcParams['font.size'] = 12

    if mesh_filename is not None:
        mesh = firedrake.Mesh(mesh_filename)
    elif firedrake_mesh is not None:
        mesh = firedrake_mesh
    else:
        raise ValueError("Please specify mesh")

    coordinates = copy.deepcopy(mesh.coordinates.dat.data)

    mesh.coordinates.dat.data[:, 0] = coordinates[:, 1]
    mesh.coordinates.dat.data[:, 1] = coordinates[:, 0]

    DG0 = firedrake.FunctionSpace(mesh, "DG", 0)
    f = firedrake.interpolate(firedrake.CellSize(mesh), DG0)

    fig, axes = plt.subplots()
    if show_size_contour:
        im = firedrake.tricontourf(f, axes=axes)
    else:
        im = firedrake.triplot(mesh, axes=axes)

    axes.set_aspect("equal", "box")
    plt.xlabel("X (km)")
    plt.ylabel("Z (km)")
    plt.title(title_str)

    if show_size_contour:
        cbar = fig.colorbar(im, orientation="horizontal")
        cbar.ax.set_xlabel("circumcircle radius (km)")
    fig.set_size_inches(13, 10)
    if show:
        plt.show()
    if output_filename is not None:
        plt.savefig(output_filename)

    # Flip back mesh coordinates so it does not change outside of method
    coordinates = copy.deepcopy(mesh.coordinates.dat.data)

    mesh.coordinates.dat.data[:, 0] = coordinates[:, 1]
    mesh.coordinates.dat.data[:, 1] = coordinates[:, 0]


def plot_model(Wave_object, filename="model.png", abc_points=None, show=False, flip_axis=True):
    """
    Plot the model with source and receiver locations.

    Parameters
    -----------
    Wave_object:
        The Wave object containing the model and locations.
    filename (optional):
        The filename to save the plot (default: "model.png").
    abc_points (optional):
        List of points to plot an ABC line (default: None).
    """
    plt.close()
    fig = plt.figure(figsize=(9, 9))
    axes = fig.add_subplot(111)
    fig.set_figwidth = 9.0
    fig.set_figheight = 9.0
    vp_object = Wave_object.initial_velocity_model
    vp_image = firedrake.tripcolor(vp_object, axes=axes)
    for source in Wave_object.source_locations:
        z, x = source
        plt.scatter(z, x, c="green")
    for receiver in Wave_object.receiver_locations:
        z, x = receiver
        plt.scatter(z, x, c="red")

    if flip_axis:
        axes.invert_yaxis()

    axes.set_xlabel("Z (km)")

    if flip_axis:
        axes.set_ylabel("X (km)", rotation=-90, labelpad=20)
        plt.setp(axes.get_xticklabels(), rotation=-90, va="top", ha="center")
        plt.setp(axes.get_yticklabels(), rotation=-90, va="center", ha="left")
    else:
        axes.set_ylabel("X (km)")

    cbar = plt.colorbar(vp_image, orientation="horizontal")
    cbar.set_label("Velocity (km/s)")
    if flip_axis:
        cbar.ax.tick_params(rotation=-90)
    axes.tick_params(axis='y', pad=20)
    axes.axis('equal')

    if abc_points is not None:
        zs = []
        xs = []

        first = True
        for point in abc_points:
            z, x = point
            zs.append(z)
            xs.append(x)
            if first:
                z_first = z
                x_first = x
            first = False
        zs.append(z_first)
        xs.append(x_first)
        plt.plot(zs, xs, "--")
    print(f"File name {filename}", flush=True)
    plt.savefig(filename)

    if flip_axis:
        img = Image.open(filename)
        img_rotated = img.rotate(90)

        # Save the rotated image
        img_rotated.save(filename)
    if show:
        plt.show()
    else:
        plt.close()


def plot_function(function):
    plt.close()
    fig = plt.figure(figsize=(9, 9))
    axes = fig.add_subplot(111)
    fig.set_figwidth = 9.0
    fig.set_figheight = 9.0
    firedrake.tricontourf(function, axes=axes)
    axes.axis('equal')


def debug_plot(function, filename="debug.png"):
    plot_function(function)
    plt.savefig(filename)


def debug_pvd(function, filename="debug.pvd"):
    out = firedrake.VTKFile(filename)
    out.write(function)


def plot_model_in_p1(Wave_object, dx=0.01, filename="model.png", abc_points=None, show=False, flip_axis=True):
    """
    Plots the velocity model of a given wave_object projected into a P1 (linear) finite element discretization.
    This function creates a deep copy of the input Wave_object's configuration, modifies it to use
    a P1 (degree 1) continuous Galerkin, and then generates a plot of the resulting velocity model.
    The plot can be saved to a file and optionally displayed.

    Parameters
    -----------
    Wave_object:
        An instance of a wave simulation object containing the velocity model and configuration.
    dx (float, optional):
        The mesh spacing to use for the new model. Defaults to 0.01.
    filename (str, optional):
        The filename to save the plot image. Defaults to "model.png".
    abc_points (list or None, optional):
        Points for absorbing boundary conditions to be marked on the plot. Defaults to None.
    show (bool, optional):
        Whether to display the plot interactively. Defaults to False.
    flip_axis (bool, optional):
        Whether to flip the plot axes for visualization. Defaults to True.

    Returns
    -------
    The result of the plot_model function.
    """

    # Local import to avoid circular import
    from ..solvers import AcousticWave
    p1_obj_dict = copy.deepcopy(Wave_object.input_dictionary)
    p1_obj_dict["options"]["method"] = "CG"
    p1_obj_dict["options"]["variant"] = "equispaced"
    p1_obj_dict["options"]["degree"] = 1

    new_wave_obj = AcousticWave(dictionary=p1_obj_dict)
    new_wave_obj.set_mesh(input_mesh_parameters={"edge_length": dx})
    new_wave_obj.set_initial_velocity_model(conditional=Wave_object.initial_velocity_model)

    return plot_model(new_wave_obj, filename=filename, abc_points=abc_points, show=show, flip_axis=flip_axis)
