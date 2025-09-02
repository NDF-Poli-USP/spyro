import numpy as np
import uuid
from mpi4py import MPI  # noqa:F401
from firedrake import COMM_WORLD  # noqa:
import warnings
from ..io.dictionaryio import Read_options, Read_outputs
from ..io.boundary_layer_io import Read_boundary_layer
from ..io.time_io import Read_time_axis
from .. import io
from .. import utils
from .. import meshing


class Model_parameters(Read_options, Read_boundary_layer, Read_time_axis, Read_outputs):
    """
    Class that reads and sanitizes input parameters.

    Attributes
    ----------
    input_dictionary: dictionary
        Contains all input parameters already organized based on examples
        from github.
    cell_type: str
        Type of cell used in meshing. Can be "T" for triangles or "Q" for
        quadrilaterals.
    method: str
        Method used in meshing. Can be "MLT" for mass lumped triangles,
        "spectral_quadrilateral" for spectral quadrilaterals, "DG_triangle"
        for discontinuous Galerkin triangles, or "DG_quadrilateral" for
        discontinuous Galerkin quadrilaterals.
    variant: str
        Variant used in meshing. Can be "lumped" for lumped mass matrices,
        "equispaced" for equispaced nodes, or "DG" for discontinuous Galerkin
        nodes.
    degree: int
        Degree of the basis functions used in the FEM.
    dimension: int
        Dimension of the mesh.
    final_time: float
        Final time of the simulation.
    dt: float
        Time step of the simulation.
    initial_time: float
        Initial time of the simulation.
    output_frequency: int
        Frequency of outputting the solution to pvd files.
    gradient_sampling_frequency: int
        Frequency of saving the solution to RAM.
    number_of_sources: int
        Number of sources used in the simulation.
    source_locations: list
        List of source locations.
    frequency: float
        Frequency of the source.
    amplitude: float
        Amplitude of the source.
    delay: float
        Delay of the source.
    number_of_receivers: int
        Number of receivers used in the simulation.
    receiver_locations: list
        List of receiver locations.
    parallelism_type: str
        Type of parallelism used in the simulation. Can be "automatic" for
        automatic parallelism or "spatial" for spatial parallelism.
    mesh_file: str
        Path to the mesh file.
    length_z: float
        Length of the domain in the z-direction.
    length_x: float
        Length of the domain in the x-direction.
    length_y: float
        Length of the domain in the y-direction.
    user_mesh: spyro.Mesh
        User defined mesh.
    firedrake_mesh: firedrake.Mesh
        Firedrake mesh.
    abc_active: bool
        Whether or not the absorbing boundary conditions are used.
    abc_exponent: int
        Exponent of the absorbing boundary conditions.
    abc_cmax: float
        Maximum acoustic wave velocity in the absorbing boundary conditions.
    abc_R: float
        Theoretical reflection coefficient of the absorbing boundary
        conditions.
    abc_pad_length: float
        Thickness of the absorbing boundary conditions.
    abc_boundary_layer_type : `str`
        Type of the boundary layer. Option 'hybrid' is based on paper
        of Salas et al. (2022). doi: https://doi.org/10.1016/j.apm.2022.09.014
    abc_boundary_layer_shape : str
        Shape type of pad layer. Options: 'rectangular' or 'hypershape'
    abc_deg_layer : `int`
        Hypershape degree
    abc_reference_freq : `str`
        Reference frequency for sizing the hybrid absorbing layer.
        Options: 'source' or 'boundary'
    abc_deg_eikonal : `int`
        Finite element order for the Eikonal analysis
    abc_get_ref_model : `bool`
        If True, the infinite model is created
    source_type: str
        Type of source used in the simulation. Can be "ricker" for a Ricker
        wavelet or "MMS" for a manufactured solution.
    running_fwi: bool
        Whether or not the simulation is a FWI.
    initial_velocity_model_file: str
        Path to the initial velocity model file.
    fwi_output_folder: str
        Path to the FWI output folder.
    control_output_file: str
        Path to the control output file.
    gradient_output_file: str
        Path to the gradient output file.
    optimization_parameters: dict
        Dictionary of the optimization parameters.
    automatic_adjoint: bool
        Whether or not the adjoint is calculated automatically.
    forward_output: bool
        Whether or not the forward output is saved.
    fwi_velocity_model_output: bool
        Whether or not the FWI velocity model output is saved.
    gradient_output: bool
        Whether or not the gradient output is saved.
    adjoint_output: bool
        Whether or not the adjoint output is saved.
    forward_output_file: str
        Path to the forward output file.
    fwi_velocity_model_output_file: str
        Path to the FWI velocity model output file.
    gradient_output_file: str
        Path to the gradient output file.
    adjoint_output_file: str
        Path to the adjoint output file.
    comm: MPI communicator
        MPI communicator.
    velocity_model_type: str
        Type of velocity model used in the simulation. Can be "file" for a
        file, "conditional" for a conditional, or None for no velocity model.
    velocity_conditional: str
        Conditional used for the velocity model.
    equation_type: str
        Type of equation used in the simulation. Can be "second_order_in_pressure".
    time_integrator: str
        Type of time integrator used in the simulation. Can be "central_difference".

    Methods
    -------
    get_wavelet()
        Returns a wavelet based on the source type.
    set_mesh()
        Sets the mesh.
    get_mesh()
        Reads in a mesh and scatters it between cores.
    """

    def __init__(self, dictionary=None, comm=None):
        """Initializes class that reads and sanitizes input parameters.
        A dictionary can be used.

        Parameters
        ----------
        dictionary: 'dictionary' (optional)
            Contains all input parameters already organized based on examples
            from github.
        comm: MPI communicator (optional)
            MPI comunicator. If None is given model_parameters creates one.

        Returns
        -------
        model_parameters: :class: 'model_parameters' object
        """
        # Converts old dictionary to new one. Deprecated feature
        if "opts" in dictionary:
            warnings.warn("Old deprecated dictionary style in usage.")
            dictionary = io.Dictionary_conversion(dictionary).new_dictionary
        # Saves inout_dictionary internally
        self.input_dictionary = dictionary

        # some default parameters we might use in the future
        self.input_dictionary.setdefault("equation_type", "second_order_in_pressure")
        self.equation_type = self.input_dictionary["equation_type"]

        # Get options
        Read_options.__init__(self, dictionary=self.input_dictionary)

        self.sources = None

        # Checks time inputs
        Read_time_axis.__init__(self)

        # Checks outputs
        Read_outputs.__init__(self)

        # Checking source and receiver inputs
        self.input_dictionary["acquisition"].setdefault("source_type", "ricker")
        self.source_type = self.input_dictionary["acquisition"]["source_type"]

        if self.source_type == "ricker":
            self.frequency = self.input_dictionary["acquisition"]["frequency"]
    
        self.input_dictionary["acquisition"].setdefault("amplitude", 1.0)
        self.amplitude = self.input_dictionary["acquisition"]["amplitude"]
    
        self.input_dictionary["acquisition"].setdefault("delay", 1.5)
        self.delay = self.input_dictionary["acquisition"]["delay"]

        self.input_dictionary["acquisition"].setdefault("delay_type", "multiples_of_minimun")
        self.delay_type = self.input_dictionary["acquisition"]["delay_type"]

        self.input_dictionary["acquisition"].setdefault("source_locations", None)
        self.source_locations = self.input_dictionary["acquisition"]["source_locations"]

        self.input_dictionary["acquisition"].setdefault("receiver_locations", None)
        self.receiver_locations = self.input_dictionary["acquisition"]["receiver_locations"]

        # Setting up MPI communicator and checking parallelism:
        self.input_dictionary.setdefault("parallelism", {})
        self.input_dictionary["parallelism"].setdefault("type", "automatic")
        self.parallelism_type = self.input_dictionary["parallelism"]["type"]

        # Checking absorving boundary condition parameters
        Read_boundary_layer.__init__(self, dictionary=self.input_dictionary)

        # Checking mesh_parameters
        if self.cell_type == "quadrilateral" or self.method == "spectral_quadrilateral":
            quadrilateral = True
        else:
            quadrilateral = False
        self.input_dictionary["mesh"].setdefault("user_mesh", None)
        self.input_dictionary["mesh"].setdefault("negative_z", True)
        self.user_mesh = self.input_dictionary["mesh"]["user_mesh"]
        self.mesh_parameters = meshing.MeshingParameters(
            input_mesh_dictionary=self.input_dictionary["mesh"],
            dimension=self.dimension,
            source_frequency=self.input_dictionary["acquisition"]["frequency"],
            comm=self.comm,
            quadrilateral=quadrilateral,
            method=self.method,
            degree=self.degree,
            abc_pad_length=self.input_dictionary["absorving_boundary_conditions"]["pad_length"],
            negative_z=self.input_dictionary["mesh"]["negative_z"]
        )

        # Check automatic adjoint
        self.input_dictionary["time_axis"].setdefault("output_frequency", 99999)
        self.gradient_sampling_frequency = self.input_dictionary["time_axis"]["gradient_sampling_frequency"]
        self.output_frequency = self.input_dictionary["time_axis"]["output_frequency"]
        self._sanitize_automatic_adjoint()

        # add random string for temp files
        self.random_id_string = str(uuid.uuid4())[:10]

    @property
    def source_locations(self):
        return self._source_locations
    
    @source_locations.setter
    def source_locations(self, value):
        if self.dimension == 2:
            mesh_lengths = [self.mesh_parameters.length_z, self.mesh_parameters.length_x]
        elif self.dimension == 3:
            mesh_lengths = [self.mesh_parameters.length_z, self.mesh_parameters.length_x, self.mesh_parameters.length_y]
        if value is not None:
            for source in value:
                source_points = list(source)
                _check_point_in_domain(source_points, mesh_lengths, self.mesh_parameters.negative_z)
            self.number_of_sources = len(value)
        else:
            self.number_of_sources = 1

        self._source_locations = value
    
    @property
    def receiver_locations(self):
        return self._receiver_locations
    
    @receiver_locations.setter
    def receiver_locations(self, value):
        if self.dimension == 2:
            mesh_lengths = [self.mesh_parameters.length_z, self.mesh_parameters.length_x]
        elif self.dimension == 3:
            mesh_lengths = [self.mesh_parameters.length_z, self.mesh_parameters.length_x, self.mesh_parameters.length_y]
        if value is not None:
            for receiver in value:
                receiver_points = list(receiver)
                _check_point_in_domain(receiver_points, mesh_lengths, self.mesh_parameters.negative_z)
            self.number_of_receivers = len(value)

        self._receiver_locations = value

    @property
    def delay_type(self):
        return self._delay_type
    
    @delay_type.setter
    def delay_type(self, value):
        accepted_values = ["multiples_of_minimun", "time"]
        _validate_enum(value, accepted_values, 'delay_type')
        self._delay_type = value

    @property
    def source_type(self):
        return self._source_type
    
    @source_type.setter
    def source_type(self, value):
        accepted_values = ["ricker", "MMS"]
        _validate_enum(value, accepted_values, 'source_type')
        self._source_type = value

    @property
    def source_locations(self):
        return self._source_locations
    
    @source_locations.setter
    def source_locations(self, value):
        if value is not None:
            self.number_of_sources = len(value)
        else:
            self.number_of_sources = 1
        self._source_locations = value

    @property
    def frequency(self):
        return self._frequency
    
    @frequency.setter
    def frequency(self, value):
        if value is not None:
            if value < 1.0:
                warnings.warn("Frequency of {value} too low for realistic FWI.")
            elif value > 50:
                warnings.warn("Frequency of {value} too high for eficient FWI.")
        self._frequency = value

    @property
    def receiver_locations(self):
        return self._receiver_locations
    
    @receiver_locations.setter
    def receiver_locations(self, value):
        if value is not None:
            self.number_of_receivers = len(value)
        else:
            self.number_of_receivers = 1
        self._receiver_locations = value
    
    @property
    def equation_type(self):
        return self._equation_type
    
    @equation_type.setter
    def equation_type(self, value):
        if value != "second_order_in_pressure":
            raise ValueError(
                "The equation type specified is not implemented yet"
            )
        self._equation_type = value

    @property
    def parallelism_type(self):
        return self._parallelism_type
    
    @parallelism_type.setter
    def parallelism_type(self, value):
        accepted_values = ["custom", "automatic", "spatial", ]
        _validate_enum(value, accepted_values, 'parallelism_type')

        if value == "custom":
            self.shot_ids_per_propagation = self.input_dictionary["parallelism"]["shot_ids_per_propagation"]
        elif value == "automatic":
            self.shot_ids_per_propagation = [[i] for i in range(0, self.number_of_sources)]
        elif value == "spatial":
            self.shot_ids_per_propagation = [[i] for i in range(0, self.number_of_sources)]

        self._parallelism_type = value
        self.comm = utils.mpi_init(self)
        self.comm.comm.barrier()

    def _sanitize_automatic_adjoint(self):
        dictionary = self.input_dictionary
        if "automatic_adjoint" in dictionary:
            self.automatic_adjoint = True
        else:
            self.automatic_adjoint = False

    def _sanitize_time_inputs(self):
        self.__check_time()

    def __check_time(self):
        if self.final_time < 0.0:
            raise ValueError(f"Negative time of {self.final_time} not valid.")
        if self.dt > 1.0:
            warnings.warn(f"Time step of {self.dt} too big.")
        if self.dt is None:
            warnings.warn(
                "Timestep not given. Will calculate internally when user \
                    attemps to propagate wave."
            )

    def set_mesh(
        self,
        user_mesh=None,
        input_mesh_parameters={},
    ):
        """
        Set the mesh for the model.

        Parameters
        ----------
        user_mesh : spyro.Mesh, optional
            The desired mesh. The default is None.
        mesh_parameters : dict, optional
            Additional parameters for setting up the mesh. The default is an empty dictionary.

        Returns
        -------
        None
        """
        if user_mesh is not None:
            self.user_mesh = user_mesh

        pad_length = None
        if self.abc_active:
            pad_length = self.abc_pad_length
        self.mesh_parameters.set_mesh(user_mesh=user_mesh, input_mesh_parameters=input_mesh_parameters, abc_pad_length=pad_length)

        if self.mesh_parameters.automatic_mesh:
            autoMeshing = meshing.AutomaticMesh(
                mesh_parameters=self.mesh_parameters,
            )

            self.user_mesh = autoMeshing.create_mesh()

    def _set_mesh_length(
        self,
        length_z=None,
        length_x=None,
        length_y=None,
    ):
        if length_z is not None:
            self.mesh_parameters.length_z = length_z
        if length_x is not None:
            self.mesh_parameters.length_x = length_x
        if length_y is not None:
            self.mesh_parameters.length_y = length_y

    def get_mesh(self):
        """Reads in an external mesh and scatters it between cores.

        Returns
        -------
        mesh: Firedrake.Mesh object
            The distributed mesh across `ens_comm`
        """
        if self.mesh_parameters.user_mesh is not None:
            self.user_mesh = self.mesh_parameters.user_mesh
            return self.user_mesh
        elif self.mesh_parameters.mesh_file is not None:
            return io.read_mesh(self.mesh_parameters)
        else:
            return self.user_mesh


def _validate_enum(value, accepted_values, name):
    if value not in accepted_values:
        raise ValueError(f"{name} of {value} not one of {accepted_values}.")
    return value


def _check_point_in_domain(point_coordinates, mesh_lengths, negative_z):
    """
    Checks if a point is within the mesh domain.

    Parameters
    ----------
    point_coordinates : list
        Coordinates of the point to check.
    mesh_lengths : list
        Lengths of the mesh in each dimension (always positive).
    negative_z : bool
        If True, the first dimension (z) is negative.

    Raises
    ------
    ValueError
        If the point is outside the mesh domain.
    """
    if negative_z:
        mesh_lengths[0] = -mesh_lengths[0]
    
    for i, (coord, length) in enumerate(zip(point_coordinates, mesh_lengths)):
        if negative_z and i == 0:
            # For negative_z, domain is [length, 0] (length is negative)
            if not (length <= coord <= 0):
                raise ValueError(
                    f"Coordinate {coord} in dimension {i} is outside the domain [{length}, 0]."
                )
        else:
            # For other dimensions, domain is [0, length]
            if not (0 <= coord <= length):
                raise ValueError(
                    f"Coordinate {coord} in dimension {i} is outside the domain [0, {length}]."
                )
