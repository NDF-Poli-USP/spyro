import warnings
import os


def cells_per_wavelength(method, degree, dimension):
    cell_per_wavelength_dictionary = {
        'mass_lumped_triangle2dim2': 7.02,
        'mass_lumped_triangle3dim2': 3.70,
        'mass_lumped_triangle4dim2': 2.67,
        'mass_lumped_triangle5dim2': 2.03,
        'mass_lumped_triangle2dim3': 6.12,
        'mass_lumped_triangle3dim3': 3.72,
        'spectral_quadrilateral2dim2': None,
        'spectral_quadrilateral4dim2': None,
        'spectral_quadrilateral6dim2': None,
        'spectral_quadrilateral8dim2': None,
        'spectral_quadrilateral2dim3': None,
        'spectral_quadrilateral4dim3': None,
        'spectral_quadrilateral6dim3': None,
        'spectral_quadrilateral8dim3': None,
    }

    key = f"{method}{degree}dim{dimension}"

    return cell_per_wavelength_dictionary.get(key)


class MeshingParameters():
    """
    Class that handles mesh parameter logic and mesh type/length/file handling.
    """

    def __init__(self, input_mesh_dictionary={}, dimension=None, source_frequency=None, comm=None, quadrilateral=False, method=None, degree=None, velocity_model=None, abc_pad_length=None, negative_z=True, use_defaults=True):
        """
        Initializes the MeshingParamaters class.

        Parameters
        ----------
        mesh_dictionary : dict, optional
            Dictionary containing mesh parameters.
        dimension : int, optional
            Dimension of the mesh.
        comm : MPI communicator, optional
            MPI communicator.
        """
        self.input_mesh_dictionary = input_mesh_dictionary or {}
        self.dimension = dimension
        self.comm = comm

        # Set mesh parameters from dictionary or defaults
        self.quadrilateral = quadrilateral
        self.mesh_type = self.input_mesh_dictionary.get("mesh_type", None)
        self.method = method
        self.periodic = False
        self.mesh_file = self.input_mesh_dictionary.get("mesh_file", None)
        self.length_z = self.input_mesh_dictionary.get("length_z", None)
        self.length_x = self.input_mesh_dictionary.get("length_x", None)
        self.length_y = self.input_mesh_dictionary.get("length_y", None)
        self.user_mesh = self.input_mesh_dictionary.get("user_mesh", None)
        self.output_filename = self.input_mesh_dictionary.get("output_filename",
                                                              "automatic_mesh.msh")
        self.source_frequency = source_frequency
        self.abc_pad_length = abc_pad_length
        self.degree = degree
        self.minimum_velocity = None
        self.velocity_model = velocity_model
        self.automatic_mesh = self.mesh_type in {"firedrake_mesh", "SeismicMesh", "spyro_mesh"}
        self._edge_length = None
        self._cells_per_wavelength = None
        self.edge_length = None
        self.cells_per_wavelength = None
        self.grid_velocity_data = None
        self.gradient_mask = None
        self.negative_z = negative_z
        if use_defaults:
            self.set_mesh(input_mesh_parameters=input_mesh_dictionary)

    def _set_length_with_unit_check(self, attr_name, value):
        """
        Checks if all dimensions are in the same unit (meters or km)
        """
        if value is not None:
            if value > 100:
                new_unit = "meters"
            else:
                new_unit = "km"
        else:
            new_unit = None
        if not hasattr(self, "_unit") or self._unit is None:
            self._unit = new_unit
        elif new_unit != self._unit and value is not None:
            warnings.warn(
                f"{attr_name} value ({value}) appears to be in {new_unit}, "
                f"but the current unit is {self._unit}. Please check for consistency."
            )
        if value is not None:
            if value < 0.0:
                raise ValueError(f"Please do not use negative value for {attr_name}")
        setattr(self, attr_name, value)

    @property
    def grid_velocity_data(self):
        return self._grid_velocity_data

    @grid_velocity_data.setter
    def grid_velocity_data(self, value):
        if value is not None:
            necessary_keys = ["vp_values", "grid_spacing"]
            for necessary_key in necessary_keys:
                if necessary_key not in value:
                    raise ValueError(f"Grid velocity data needs {necessary_key} key.")
        self._grid_velocity_data = value

    @property
    def output_filename(self):
        return self._output_filename

    @output_filename.setter
    def output_filename(self, value):
        if value is not None:
            if isinstance(value, str) and value.endswith('.vtk'):
                warnings.warn("VTK meshes for visualization only, will not run a simulation.")
            elif not (isinstance(value, str) and value.endswith('.msh')):
                raise ValueError(f"mesh_file '{value}' must be a .msh file")
        self._output_filename = value

    @property
    def edge_length(self):
        return self._edge_length

    @edge_length.setter
    def edge_length(self, value):
        if self.cells_per_wavelength is not None:
            warnings.warn(
                "Mutual exclusion: Both 'edge_length' and 'cells_per_wavelength' control mesh size, "
                "but only one can be set at a time. Setting 'edge_length' will override and remove the "
                "previously set 'cells_per_wavelength'. If you wish to use 'cells_per_wavelength' instead, "
                "set it after setting 'edge_length'."
            )
            self.cells_per_wavelength = None
        self._edge_length = value

    @property
    def cells_per_wavelength(self):
        return self._cells_per_wavelength

    @cells_per_wavelength.setter
    def cells_per_wavelength(self, value):
        if self.edge_length is not None:
            warnings.warn("Setting cells_per_wavelength removes edge_length parameter")
            self.edge_length = None
        self._cells_per_wavelength = value

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        allowed_types = {
            "mass_lumped_triangle",
            "DG_triangle",
            "spectral_quadrilateral",
            "DG_quadrilateral",
            "CG",
        }
        if value is not None and value not in allowed_types:
            raise ValueError(
                f"method must be one of {allowed_types}, got '{value}'"
            )
        self._method = value

    @property
    def mesh_file(self):
        return self._mesh_file

    @mesh_file.setter
    def mesh_file(self, value):
        if value is not None:
            if not (isinstance(value, str) and value.endswith('.msh')):
                raise ValueError(f"mesh_file '{value}' must be a .msh file")
            if not os.path.exists(value):
                raise FileNotFoundError(f"mesh_file '{value}' does not exist")
        self._mesh_file = value

    @property
    def mesh_type(self):
        return self._mesh_type

    @mesh_type.setter
    def mesh_type(self, value):
        allowed_types = {"firedrake_mesh", "user_mesh", "SeismicMesh", "file", "spyro_mesh"}
        if value is not None and value not in allowed_types:
            raise ValueError(
                f"mesh_type must be one of {allowed_types}, got '{value}'"
            )
        if value == "SeismicMesh" and self.quadrilateral:
            raise ValueError("SeismicMesh does not work with quads.")
        self._mesh_type = value

    @property
    def source_frequency(self):
        return self._source_frequency

    @source_frequency.setter
    def source_frequency(self, value):
        if value is None:
            self._source_frequency = value
        elif not isinstance(value, (int, float)):
            raise TypeError(f"Source frequency must be a number, got {type(value).__name__}")
        else:
            if value < 1.5:
                warnings.warn(f"Source frequency of {value} too low for realistic FWI case")
            elif value > 50:
                warnings.warn(f"Source frequency of {value} too high for realistic FWI case, please low-pass filter")
            self._source_frequency = value

    @property
    def abc_pad_length(self):
        return self._abc_pad_length

    @abc_pad_length.setter
    def abc_pad_length(self, value):
        self._set_length_with_unit_check("_abc_pad_length", value)

    @property
    def length_z(self):
        return self._length_z

    @length_z.setter
    def length_z(self, value):
        self._set_length_with_unit_check("_length_z", value)

    @property
    def length_x(self):
        return self._length_x

    @length_x.setter
    def length_x(self, value):
        self._set_length_with_unit_check("_length_x", value)

    @property
    def length_y(self):
        return self._length_y

    @length_y.setter
    def length_y(self, value):
        self._set_length_with_unit_check("_length_y", value)

    @property
    def user_mesh(self):
        return self._user_mesh

    @user_mesh.setter
    def user_mesh(self, value):
        if value is not None:
            self.mesh_type = "user_mesh"
        self._user_mesh = value

    @property
    def periodic(self):
        return self._periodic

    @periodic.setter
    def periodic(self, value):
        if self.mesh_type != "firedrake_mesh" and value is True:
            raise ValueError("Periodic meshes are only supported with Firedrake meshes for now.")
        self._periodic = value

    def set_mesh(
        self,
        user_mesh=None,
        input_mesh_parameters={},
        abc_pad_length=None,
    ):
        """
        Set the mesh for the model.

        Parameters
        ----------
        user_mesh : spyro.Mesh, optional
            The desired mesh. The default is None.
        input_mesh_parameters : dict, optional
            Additional parameters for setting up the mesh. The default is an empty dictionary.

        Returns
        -------
        None
        """
        if abc_pad_length is not None:
            self.abc_pad_length = abc_pad_length

        # Setting default mesh parameters
        input_mesh_parameters.setdefault("periodic", self.periodic)
        input_mesh_parameters.setdefault("minimum_velocity", self.minimum_velocity)
        input_mesh_parameters.setdefault("length_z", self.length_z)
        input_mesh_parameters.setdefault("length_x", self.length_x)
        input_mesh_parameters.setdefault("length_y", self.length_y)
        input_mesh_parameters.setdefault("abc_pad_length", self.abc_pad_length)
        input_mesh_parameters.setdefault("mesh_file", self.mesh_file)
        input_mesh_parameters.setdefault("dimension", self.dimension)
        input_mesh_parameters.setdefault("mesh_type", self.mesh_type)
        input_mesh_parameters.setdefault("source_frequency", self.source_frequency)
        input_mesh_parameters.setdefault("method", self.method)
        input_mesh_parameters.setdefault("degree", self.degree)
        input_mesh_parameters.setdefault("quadrilateral", self.quadrilateral)
        input_mesh_parameters.setdefault("velocity_model", self.velocity_model)

        # Mesh length based parameters
        input_mesh_parameters.setdefault("cells_per_wavelength", None)
        input_mesh_parameters.setdefault("edge_length", None)

        # Set all parameters that are not None
        for key, value in input_mesh_parameters.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)

        self.automatic_mesh = self.mesh_type in {"firedrake_mesh", "SeismicMesh", "spyro_mesh"}
