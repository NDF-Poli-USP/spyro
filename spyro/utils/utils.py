import copy
from firedrake import *  # noqa: F403
import numpy as np
from mpi4py import MPI
from scipy.signal import butter, filtfilt
import warnings


def butter_lowpass_filter(shot, cutoff, fs, order=2):
    """Low-pass filter the shot record with sampling-rate fs Hz
    and cutoff freq. Hz

    Parameters
    ----------
    shot : numpy array
        Shot record
    cutoff : float
        Cutoff frequency in Hertz
    fs : float
        Sampling rate in Hertz
    order : int
        Order of the filter

    Returns
    -------
    filtered_shot : numpy array
        Filtered shot record
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    nr, nc = np.shape(shot)
    filtered_shot = np.zeros((nr, nc))
    for rec, ts in enumerate(shot.T):
        filtered_shot[:, rec] = filtfilt(b, a, ts)
    return filtered_shot


def compute_functional(Wave_object, residual):
    """Compute the functional to be optimized.
    Accepts the velocity optionally and uses
    it if regularization is enabled
    """
    num_receivers = Wave_object.number_of_receivers
    dt = Wave_object.dt
    comm = Wave_object.comm

    J = 0
    for rn in range(num_receivers):
        J += np.trapz(residual[:, rn] ** 2, dx=dt)

    J *= 0.5

    J_total = np.zeros((1))
    J_total[0] += J
    J_total = COMM_WORLD.allreduce(J_total, op=MPI.SUM)
    J_total[0] /= comm.comm.size
    return J_total[0]


def evaluate_misfit(model, guess, exact):
    """Compute the difference between the guess and exact
    at the receiver locations given downsampling"""

    if "skip" in model["timeaxis"]:
        skip = model["timeaxis"]["skip"]
    else:
        skip = 1

    ll = int(exact.shape[0] / skip)
    ds_exact = exact[::skip]
    return ds_exact[:ll] - guess


def myrank(COMM=COMM_SELF):  # noqa: F405
    return COMM.Get_rank()


def mysize(COMM=COMM_SELF):  # noqa: F405
    return COMM.Get_size()


def mpi_init(model):
    """Initialize computing environment"""
    # rank = myrank()
    # size = mysize()
    available_cores = COMM_WORLD.size  # noqa: F405
    print(f"Parallelism type: {model.parallelism_type}", flush=True)
    if model.parallelism_type == "automatic":
        num_cores_per_shot = available_cores / model.number_of_sources
        if available_cores % model.number_of_sources != 0:
            raise ValueError(
                "Available cores cannot be divided between sources equally."
            )
    elif model.parallelism_type == "spatial":
        num_cores_per_shot = available_cores
    elif model.parallelism_type == "serial":
        return Ensemble(COMM_WORLD, 1)
    elif model.parallelism_type == "custom":
        raise ValueError("Custom parallelism not yet implemented")

    comm_ens = Ensemble(COMM_WORLD, num_cores_per_shot)  # noqa: F405
    return comm_ens


def mpi_init_simple(number_of_sources):
    """Initialize computing environment"""
    rank = myrank()  # noqa: F841
    size = mysize()  # noqa: F841
    available_cores = COMM_WORLD.size  # noqa: F405

    num_cores_per_shot = available_cores / number_of_sources
    if available_cores % number_of_sources != 0:
        raise ValueError(
            "Available cores cannot be divided between sources equally."
        )

    comm_ens = Ensemble(COMM_WORLD, num_cores_per_shot)  # noqa: F405
    return comm_ens


def communicate(array, my_ensemble):
    """Communicate shot record to all processors

    Parameters
    ----------
    array: array-like
        Array of data to all-reduce across both ensemble
        and spatial communicators.
    comm: Firedrake.comm
        A Firedrake ensemble communicator

    Returns
    -------
    array_reduced: array-like
        Array of data max all-reduced
        amongst the ensemble communicator

    """
    array_reduced = copy.copy(array)

    if my_ensemble.comm.size > 1:
        if my_ensemble.comm.rank == 0 and my_ensemble.ensemble_comm.rank == 0:
            print("Spatial parallelism, reducing to comm 0", flush=True)
        my_ensemble.comm.Allreduce(array, array_reduced, op=MPI.MAX)
    # print(array_reduced,array)
    return array_reduced


class Mask():
    """
    A class representing a mask for a wave object.

    Parameters:
    - boundaries (dict): A dictionary containing the boundaries to be applied.
    - Wave_obj (object): An optional wave object.
    - dg (bool): Flag indicating whether to use DG space for the mask. Default is False.
    - inverse_mask (bool): Flag indicating whether to invert the mask. Default is False.

    Attributes:
    - active_boundaries (list): A list of active boundaries.
    - z (array): The z coordinates of the wave object's mesh.
    - x (array): The x coordinates of the wave object's mesh.
    - y (array): The y coordinates of the wave object's mesh (if applicable).
    - mask_dofs (array): Contains the indices of the mask degrees of freedom.

    Methods:
    - _calculate_mask_dofs(Wave_obj): Calculates the mask degrees of freedom.
    - apply_mask(dJ): Applies the mask to the given Firedrake function.

    """

    def __init__(self, boundaries, Wave_obj, dg=False, inverse_mask=False):
        possible_boundaries = [
            "z_min",
            "z_max",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
        ]
        active_boundaries = []

        for possible_boundary in possible_boundaries:
            if possible_boundary in boundaries:
                setattr(self, possible_boundary, boundaries[possible_boundary])
                active_boundaries.append(possible_boundary)

        self.active_boundaries = active_boundaries
        self._calculate_mask_conditional(Wave_obj, inverse_mask)
        self.in_dg = dg
        if dg is False:
            self._calculate_mask_dofs(Wave_obj)
        elif dg is True:
            self._calculate_dg_mask(Wave_obj)

    def _calculate_dg_mask(self, Wave_obj):
        """
        Calculates the DG mask.

        Parameters:
        - Wave_obj (object): The wave object containing the necessary data.

        """
        V_dg = FunctionSpace(Wave_obj.mesh, "DG", 0)
        dg_mask = Function(V_dg)
        dg_mask.interpolate(self.cond)
        self.dg_mask = dg_mask

    def _calculate_mask_conditional(self, Wave_obj, inverted=False):
        """
        Calculates the mask degrees of freedom based on the active boundaries.

        Parameters:
        - Wave_obj (object): The wave object containing the necessary data.
        - inverted (bool, optional): If True gives nonzero value inside the boundaries

        """
        # Getting necessary data from wave object
        active_boundaries = self.active_boundaries
        self.z = Wave_obj.mesh_z
        self.x = Wave_obj.mesh_x
        if ("y_min" in active_boundaries) or ("y_max" in active_boundaries):
            self.y = Wave_obj.mesh_y

        # Getting mask conditional
        if inverted:
            cond = [1]
            true_value = [0]
            false_value = cond
        else:
            cond = [0]
            true_value = [1]
            false_value = cond

        for boundary in active_boundaries:
            axis = boundary[0]
            if boundary[-3:] == "min":
                cond[0] = conditional(getattr(self, axis) < getattr(self, boundary), true_value[0], false_value[0])
            elif boundary[-3:] == "max":
                cond[0] = conditional(getattr(self, axis) > getattr(self, boundary), true_value[0], false_value[0])
            else:
                raise ValueError(f"Boundary of {boundary} not possible")

        self.cond = cond[0]

    def _calculate_mask_dofs(self, Wave_obj):
        """
        Calculates the mask degrees of freedom.

        Parameters:
        - Wave_obj (object): The wave object containing the necessary data.

        """
        if self.in_dg:
            raise ValueError("DG space can have different DoFs than the functional space")
        warnings.warn("When applying a mask in a continuous space, expect some error in the element adjacent to the mask")
        mask = Function(Wave_obj.function_space)
        mask.interpolate(self.cond)
        # Saving mask dofs
        self.mask_dofs = np.where(mask.dat.data[:] > 0.3)

    def apply_mask(self, dJ):
        """
        Applies the mask to the given data.

        Parameters:
        - dJ (object): Firedrake function.

        Returns:
        - object: The masked data Firedrake.

        """
        dJ.dat.data[self.mask_dofs] = 0.0
        return dJ


class Gradient_mask_for_pml(Mask):
    """
    A class representing a gradient mask for the Perfectly Matched Layer (PML).

    Args:
        Wave_obj (optional): An object representing a wave. Defaults to None.

    Attributes:
        boundaries (dict): A dictionary containing the boundaries of the mask.

    """

    def __init__(self, Wave_obj):
        if Wave_obj.abc_active is False:
            raise ValueError("No PML present in wave object")

        # building firedrake function for mask
        z_min = -(Wave_obj.length_z)
        x_min = 0.0
        x_max = Wave_obj.length_x
        boundaries = {
            "z_min": z_min,
            "x_min": x_min,
            "x_max": x_max,
        }
        super().__init__(boundaries, Wave_obj)


# def analytical_solution_for_pressure_based_on_MMS(model, mesh, time):
#     degree = model["opts"]["degree"]
#     V = FunctionSpace(mesh, "CG", degree)  # noqa: F405
#     z, x = SpatialCoordinate(mesh)  # noqa: F405
#     p = Function(V).interpolate(  # noqa: F405
#         (time**2) * sin(pi * z) * sin(pi * x)  # noqa: F405
#     )
#     return p
