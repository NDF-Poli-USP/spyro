import copy
from firedrake import *  # noqa: F403
import numpy as np
from mpi4py import MPI
import os
from scipy.signal import butter, filtfilt
import warnings
from ..io import ensemble_functional
from ..io import parallel_print
try:
    from SeismicMesh import write_velocity_model
    SEISMIC_MESH_AVAILABLE = True
except ImportError:
    SEISMIC_MESH_AVAILABLE = False


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


@ensemble_functional
def compute_functional(Wave_object, residual, step, nsteps):
    """Compute the functional contribution at one time step.

    Parameters
    ----------
    Wave_object: Spyro wave object
        The wave object containing the necessary data and parameters.
    residual: Firedrake function or numpy array
        The residual at the current time step.
    step: int
        The current time step index.
    nsteps: int
        The total number of time steps.

    Returns
    -------
    float
        The contribution to the functional from the current time step.
    """
    dt = Wave_object.dt
    weight = 0.5 if step == 0 or step == nsteps - 1 else 1.0

    if Wave_object.use_vertex_only_mesh:
        return assemble(0.5 * dt * weight * inner(residual, residual) * dx)

    else:
        residual_array = np.asarray(residual)
        if residual_array.ndim != 1:
            raise ValueError(
                "Expected one residual vector with shape "
                "(num_receivers,) for the current time step."
            )
        g_n = np.sum(residual_array**2)
        return g_n * (0.5 * dt * weight)


def evaluate_misfit(model, guess, exact):
    """Compute the difference between synthetic and observed data.

    Calculates the residual (misfit) between the exact (observed) and
    guess (synthetic) data at receiver locations, accounting for temporal
    downsampling specified in the model configuration.

    Parameters
    ----------
    model : dict
        Model configuration dictionary. If model['timeaxis']['skip'] exists,
        it specifies the downsampling factor for the exact data.
    guess : numpy.ndarray
        Synthetic data array of shape (n_time_steps, n_receivers).
    exact : numpy.ndarray
        Observed data array of shape (n_time_steps, n_receivers) before
        downsampling.

    Returns
    -------
    numpy.ndarray
        Residual array of shape (n_downsampled_time_steps, n_receivers)
        computed as downsampled_exact - guess.

    Notes
    -----
    The exact data is downsampled by taking every `skip`-th sample,
    while the guess data is assumed to already be at the correct sampling rate.
    """

    if "skip" in model["timeaxis"]:
        skip = model["timeaxis"]["skip"]
    else:
        skip = 1

    ll = int(exact.shape[0] / skip)
    ds_exact = exact[::skip]
    return ds_exact[:ll] - guess


def myrank(COMM=COMM_SELF):  # noqa: F405
    """Get the rank of the current MPI process.

    Parameters
    ----------
    COMM : MPI communicator, optional
        MPI communicator object. Default is COMM_SELF.

    Returns
    -------
    int
        Rank of the current process in the communicator.
    """
    return COMM.Get_rank()


def mysize(COMM=COMM_SELF):  # noqa: F405
    """Get the total number of MPI processes.

    Parameters
    ----------
    COMM : MPI communicator, optional
        MPI communicator object. Default is COMM_SELF.

    Returns
    -------
    int
        Total number of processes in the communicator.
    """
    return COMM.Get_size()


def mpi_init(model):
    """Initialize MPI computing environment for wave propagation simulations.

    Sets up parallel computing environment based on the parallelism type
    specified in the model. Creates appropriate ensemble communicators
    for distributing sources across available cores.

    Parameters
    ----------
    model : object
        Wave object containing parallelism configuration. Must have attributes:
        - parallelism_type : str
            Type of parallelism: 'automatic', 'spatial', or 'custom'.
        - number_of_sources : int
            Number of sources (required for 'automatic' mode).
        - shot_ids_per_propagation : list of lists
            Shot IDs for each propagation (required for 'custom' mode).

    Returns
    -------
    firedrake.ensemble.Ensemble
        Ensemble communicator configured for the specified parallelism type.

    Raises
    ------
    ValueError
        If available cores cannot be divided equally among sources in
        'automatic' mode.

    Notes
    -----
    Parallelism types:
    - 'automatic': Divides cores equally among sources.
    - 'spatial': Uses all cores for each source (sequential source processing).
    - 'custom': Uses user-defined shot distribution, allowing multiple shots in
        individual propagations. These propagations can be parallelized using
        ensemble paralelism with internal spatial paralelism.
    """
    # rank = myrank()
    # size = mysize()
    available_cores = COMM_WORLD.size  # noqa: F405
    if model.parallelism_type == "automatic":
        num_cores_per_propagation = available_cores / model.number_of_sources
        if available_cores % model.number_of_sources != 0:
            raise ValueError(
                f"Available cores cannot be divided between sources equally {available_cores}/{model.number_of_sources}."
            )
    elif model.parallelism_type == "spatial":
        num_cores_per_propagation = available_cores
    elif model.parallelism_type == "custom":
        shot_ids_per_propagation = model.shot_ids_per_propagation
        num_propagations = len(shot_ids_per_propagation)
        num_cores_per_propagation = available_cores / num_propagations

    comm_ens = Ensemble(COMM_WORLD, num_cores_per_propagation)  # noqa: F405
    parallel_print(f"Parallelism type: {model.parallelism_type}", comm=comm_ens)
    return comm_ens


def communicate(array, my_ensemble):
    """Communicate shot record to all processors.

    Performs an all-reduce operation with MAX reduction across spatial
    communicators within an ensemble. This is used to gather shot records
    from distributed computations in parallel wave simulations.

    Parameters
    ----------
    array : array-like
        Array of data to all-reduce across both ensemble and spatial
        communicators.
    my_ensemble : firedrake.ensemble.Ensemble
        A Firedrake ensemble communicator containing both ensemble and
        spatial communicators.

    Returns
    -------
    array_reduced : array-like
        Array of data with MAX all-reduce applied amongst the spatial
        communicator. Has the same shape and dtype as input array.

    Notes
    -----
    If spatial parallelism is used (my_ensemble.comm.size > 1), the
    function reduces data to rank 0 of the spatial communicator.
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
    DEPRECATED: Spatial mask for selective gradient updates in wave simulations.

    Creates a spatial mask to selectively zero out degrees of freedom in
    specified regions of the computational domain. Useful for gradient
    masking in full waveform inversion and restricting updates to regions
    of interest.

    Parameters
    ----------
    boundaries : dict
        Dictionary containing spatial boundaries for the mask. Valid keys are
        'z_min', 'z_max', 'x_min', 'x_max', 'y_min', 'y_max', with float
        values specifying the boundary locations.
    Wave_obj : object
        Wave object containing mesh information. Must have attributes:
        - mesh : firedrake.mesh.MeshGeometry
            Computational mesh.
        - mesh_z, mesh_x, mesh_y : firedrake.SpatialCoordinate
            Coordinate functions for the mesh.
        - function_space : firedrake.functionspace.FunctionSpace
            Function space for wave simulation (for continuous masks).
    dg : bool, optional
        If True, creates a DG (discontinuous Galerkin) mask; if False,
        creates a continuous mask. Default is False.
    inverse_mask : bool, optional
        If True, inverts the mask (nonzero inside boundaries, zero outside);
        if False, uses standard mask (zero inside boundaries, nonzero outside).
        Default is False.

    Attributes
    ----------
    active_boundaries : list of str
        List of active boundary keys from the boundaries dictionary.
    z : firedrake.SpatialCoordinate
        Z-coordinate function from the mesh.
    x : firedrake.SpatialCoordinate
        X-coordinate function from the mesh.
    y : firedrake.SpatialCoordinate
        Y-coordinate function from the mesh (3D only).
    mask_dofs : numpy.ndarray
        Indices of degrees of freedom in the masked region (continuous masks).
    dg_mask : firedrake.Function
        DG function representing the mask (DG masks only).
    cond : firedrake.Conditional
        UFL conditional expression defining the mask region.
    in_dg : bool
        Flag indicating whether the mask is in DG space.

    Warnings
    --------
    When using continuous masks, there may be small errors in elements
    adjacent to the mask boundary due to the need to interpolate the mask
    to discrete function space.

    This class is deprecated. We are switching to use either mesh related
    tags (already implemented) and submesh related tags (to be implemented)

    Examples
    --------
    Create a mask to zero gradients below z=-5.0 and outside x=[0, 10]:

    >>> boundaries = {'z_min': -5.0, 'x_min': 0.0, 'x_max': 10.0}
    >>> mask = Mask(boundaries, wave_obj)
    >>> masked_gradient = mask.apply_mask(gradient)
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
        """Calculate the discontinuous Galerkin (DG) mask.

        Creates a DG0 function space mask by interpolating the mask
        conditional onto piecewise constant elements.

        Parameters
        ----------
        Wave_obj : object
            Wave object containing the mesh on which to create the DG mask.
            Must have attribute:
            - mesh : firedrake.mesh.MeshGeometry
                Computational mesh.

        Notes
        -----
        Sets the `dg_mask` attribute to a DG0 function containing the
        interpolated mask values.
        """
        V_dg = FunctionSpace(Wave_obj.mesh, "DG", 0)
        dg_mask = Function(V_dg)
        dg_mask.interpolate(self.cond)
        self.dg_mask = dg_mask

    def _calculate_mask_conditional(self, Wave_obj, inverted=False):
        """Calculate the UFL conditional expression for the mask.

        Constructs a UFL conditional that evaluates to 1 (or 0 if inverted)
        inside the masked region and 0 (or 1 if inverted) outside.

        Parameters
        ----------
        Wave_obj : object
            Wave object containing mesh coordinate functions. Must have:
            - mesh_z : firedrake.SpatialCoordinate
                Z-coordinate function.
            - mesh_x : firedrake.SpatialCoordinate
                X-coordinate function.
            - mesh_y : firedrake.SpatialCoordinate
                Y-coordinate function (if y boundaries are active).
        inverted : bool, optional
            If True, gives nonzero value inside the boundaries and zero outside;
            if False, gives zero inside boundaries and nonzero outside.
            Default is False.

        Raises
        ------
        ValueError
            If a boundary name does not end with 'min' or 'max'.

        Notes
        -----
        Sets the `cond` attribute to a UFL conditional expression and
        `z`, `x`, and optionally `y` attributes to the coordinate functions.
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
        """Calculate degrees of freedom indices for the mask application.

        Interpolates the mask conditional onto the wave function space and
        identifies which degrees of freedom lie within the masked region.

        Parameters
        ----------
        Wave_obj : object
            Wave object containing the function space for mask interpolation.
            Must have attribute:
            - function_space : firedrake.functionspace.FunctionSpace
                Function space for the wave simulation.

        Raises
        ------
        ValueError
            If the mask is in DG space, as DG spaces may have different
            degrees of freedom than the functional space.

        Warnings
        --------
        Warns that there may be errors in elements adjacent to the mask
        boundary when using continuous function spaces.

        Notes
        -----
        Sets the `mask_dofs` attribute to a tuple containing the indices
        of degrees of freedom where the mask value exceeds 0.3.
        """
        if self.in_dg:
            raise ValueError("DG space can have different DoFs than the functional space")
        warnings.warn("When applying a mask in a continuous space, expect some error in the element adjacent to the mask")
        mask = Function(Wave_obj.function_space)
        mask.interpolate(self.cond)
        # Saving mask dofs
        self.mask_dofs = np.where(mask.dat.data[:] > 0.3)

    def apply_mask(self, dJ):
        """Apply the mask to a Firedrake function.

        Zeros out the degrees of freedom in the masked region of the
        provided function. Typically used to mask gradients in inversions.

        Parameters
        ----------
        dJ : firedrake.Function
            Firedrake function to which the mask will be applied.
            Modified in-place.

        Returns
        -------
        firedrake.Function
            The masked function (same object as input, modified in-place).

        Notes
        -----
        This method sets dJ.dat.data[mask_dofs] = 0.0, effectively zeroing
        all degrees of freedom identified during mask initialization.
        """
        dJ.dat.data[self.mask_dofs] = 0.0
        return dJ


class Gradient_mask_for_pml(Mask):
    """Gradient mask for Perfectly Matched Layer (PML) regions.

    Automatically creates a mask that zeros gradients in the PML absorbing
    boundary layer regions. Prevents unphysical updates to velocity models
    in the PML during inversion.

    Parameters
    ----------
    Wave_obj : object
        Wave object with active PML boundary conditions. Must have:
        - abc_active : bool
            Must be True; indicates PML is active.
        - mesh_parameters : object with attributes:
            - length_z : float
                Total length of the domain in z-direction.
            - length_x : float
                Total length of the domain in x-direction.

    Raises
    ------
    ValueError
        If Wave_obj.abc_active is False (no PML present).

    Notes
    -----
    The mask automatically sets boundaries to exclude the PML regions:
    - z_min: Bottom of the domain (-length_z)
    - x_min: Left edge of the domain (0.0)
    - x_max: Right edge of the domain (length_x)

    These boundaries are passed to the parent Mask class to create the
    appropriate masking region.

    Examples
    --------
    >>> gradient_mask = Gradient_mask_for_pml(wave_obj)
    >>> masked_gradient = gradient_mask.apply_mask(gradient)
    """

    def __init__(self, Wave_obj):
        if Wave_obj.abc_active is False:
            raise ValueError("No PML present in wave object")

        # building firedrake function for mask
        z_min = -(Wave_obj.mesh_parameters.length_z)
        x_min = 0.0
        x_max = Wave_obj.mesh_parameters.length_x
        boundaries = {
            "z_min": z_min,
            "x_min": x_min,
            "x_max": x_max,
        }
        super().__init__(boundaries, Wave_obj)


def run_in_one_core(func):
    """Decorator to execute function only on rank 0.

    Ensures the decorated function runs only on the root process (rank 0)
    of the communicator. Other processes skip execution. Useful for I/O
    operations and serial tasks in parallel environments.

    Parameters
    ----------
    func : callable
        Function to decorate. The first argument of func must be an object
        with a `comm` attribute containing an MPI communicator.

    Returns
    -------
    callable
        Wrapped function that executes only on rank 0.

    Notes
    -----
    The function checks for two types of communicators:
    - Ensemble communicator: Runs only if both `ensemble_comm.rank` == 0
      and comm.rank == 0.
    - Regular communicator: Runs only if comm.rank == 0.
    - If comm is None, the function runs normally without restrictions.

    The function does not broadcast results to other processes.

    See Also
    --------
    run_in_one_core_and_broadcast : Similar decorator that also broadcasts results.

    Examples
    --------
    >>> @run_in_one_core
    ... def save_file(obj, filename):
    ...     # Only rank 0 writes the file
    ...     with open(filename, 'w') as f:
    ...         f.write(str(obj.data))
    """

    def wrapper(*args, **kwargs):
        comm = args[0].comm
        if comm is None:
            return func(*args, **kwargs)
        else:
            if getattr(comm, "ensemble_comm", None) is not None:
                if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
                    return func(*args, **kwargs)
            elif getattr(comm, "rank", None) is not None:
                if comm.rank == 0:
                    return func(*args, **kwargs)

    return wrapper


def run_in_one_core_and_broadcast(func):
    """Decorator to execute function on rank 0 and broadcast result.

    Ensures the decorated function runs only on the root process (rank 0)
    and broadcasts the return value to all other processes. Useful for
    file reading and other operations that should be performed once but
    shared across all processes.

    Parameters
    ----------
    func : callable
        Function to decorate. The first argument of func must be an object
        with a 'comm' attribute containing an MPI communicator.

    Returns
    -------
    callable
        Wrapped function that executes on rank 0 and broadcasts the result
        to all processes.

    Notes
    -----
    The function handles two types of communicators:
    - Ensemble communicator: Executes on rank (0,0), broadcasts within
      ensemble, then within spatial communicator.
    - Regular communicator: Executes on rank 0, broadcasts to all.
    - If comm is None, the function runs normally without MPI operations.

    All processes receive the same return value from the broadcast.

    See Also
    --------
    run_in_one_core : Similar decorator without broadcasting.

    Examples
    --------
    >>> @run_in_one_core_and_broadcast
    ... def load_config(obj, filename):
    ...     # Only rank 0 reads the file, result shared with all
    ...     with open(filename, 'r') as f:
    ...         return json.load(f)
    """

    def wrapper(*args, **kwargs):
        comm = args[0].comm
        if comm is None:
            return func(*args, **kwargs)
        else:
            result = None
            if getattr(comm, "ensemble_comm", None) is not None:
                # Handle ensemble communicator
                if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
                    result = func(*args, **kwargs)
                # Broadcast within ensemble
                result = comm.ensemble_comm.bcast(result, root=0)
                # Broadcast within spatial communicator
                result = comm.comm.bcast(result, root=0)
            elif getattr(comm, "rank", None) is not None:
                # Handle regular communicator
                if comm.rank == 0:
                    result = func(*args, **kwargs)
                result = comm.bcast(result, root=0)
            return result

    return wrapper


@run_in_one_core_and_broadcast
def write_hdf5_velocity_model(obj_with_comm, segy_filename):
    """Convert SEG-Y velocity model to HDF5 format.

    Converts a SEG-Y format velocity model file to HDF5 format using
    SeismicMesh. The conversion is performed on rank 0 and the output
    filename is broadcast to all processes.

    Parameters
    ----------
    obj_with_comm : object
        Object with a 'comm' attribute containing an MPI communicator.
        Used by the decorator to determine which process performs the
        conversion.
    segy_filename : str
        Path to the input SEG-Y velocity model file.

    Returns
    -------
    str
        Path to the output HDF5 file (input basename with .hdf5 extension).

    Raises
    ------
    ValueError
        If SeismicMesh is not installed.

    Notes
    -----
    This is just a wrapper for the equivelant SeismicMesh method. We
    need to substitute this with our own method, since it is not related
    to mesh generation.

    This function requires the SeismicMesh package to be installed.
    The output filename is constructed by replacing the input file
    extension with '.hdf5'.

    Due to the @run_in_one_core_and_broadcast decorator, only rank 0
    performs the actual file conversion, but all processes receive
    the output filename.

    Examples
    --------
    >>> output_file = write_hdf5_velocity_model(wave_obj, 'velocity.segy')
    >>> print(output_file)
    'velocity.hdf5'
    """
    if SEISMIC_MESH_AVAILABLE is False:
        raise ValueError("Segy to HDF5 not yet implemented natively. Please install SeismicMesh")
    vp_filename, vp_filetype = os.path.splitext(
        segy_filename
    )
    write_velocity_model(
        segy_filename, ofname=vp_filename
    )
    output_filename = vp_filename + ".hdf5"
    return output_filename


# def analytical_solution_for_pressure_based_on_MMS(model, mesh, time):
#     degree = model["opts"]["degree"]
#     V = FunctionSpace(mesh, "CG", degree)  # noqa: F405
#     z, x = SpatialCoordinate(mesh)  # noqa: F405
#     p = Function(V).interpolate(  # noqa: F405
#         (time**2) * sin(pi * z) * sin(pi * x)  # noqa: F405
#     )
#     return p
