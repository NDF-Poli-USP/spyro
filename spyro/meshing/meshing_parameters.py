import firedrake as fire
import warnings
import os


def cells_per_wavelength(method, degree, dimension):
    cell_per_wavelength_dictionary = {
        'mlt2tri': 7.02,
        'mlt3tri': 3.70,
        'mlt4tri': 2.67,
        'mlt5tri': 2.03,
        'mlt2tet': 6.12,
        'mlt3tet': 3.72,
        'sem2quad': None,
        'sem4quad': None,
        'sem6quad': None,
        'sem8quad': None,
    }

    if dimension == 2 and (method == 'mass_lumped_triangle' or method == "MLT"):
        cell_type = 'tri'
    if dimension == 3 and (method == 'mass_lumped_triangle' or method == "MLT"):
        cell_type = 'tet'
    if dimension == 2 and method == 'spectral_quadrilateral':
        cell_type = 'quad'
    if dimension == 3 and method == 'spectral_quadrilateral':
        cell_type = 'quad'

    key = method.lower()+str(degree)+cell_type

    return cell_per_wavelength_dictionary.get(key)


class MeshingParameters():
    """
    Class that handles mesh parameter logic and mesh type/length/file handling.
    """

    def __init__(self, input_mesh_dictionary=None, dimension=None, source_frequency=None, comm=None):
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
        self.mesh_file = self.input_mesh_dictionary.get("mesh_file", None)
        self.mesh_type = self.input_mesh_dictionary.get("mesh_type", None)
        self.length_z = self.input_mesh_dictionary.get("Lz", None)
        self.length_x = self.input_mesh_dictionary.get("Lx", None)
        self.length_y = self.input_mesh_dictionary.get("Ly", None)
        self.user_mesh = self.input_mesh_dictionary.get("user_mesh", None)
        self.firedrake_mesh = self.input_mesh_dictionary.get("firedrake_mesh", None)
        self.source_frequency = source_frequency
        self.abc_pad_length = None
    
    def _set_length_with_unit_check(self, attr_name, value):
        """
        Checks if all dimensions are in the same unit (meters or km)
        """
        new_unit = "meters" if value is not None and value > 100 else "km"
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
        allowed_types = {"firedrake_mesh", "user_mesh", "SeismicMesh", "file"}
        if value is not None and value not in allowed_types:
            raise ValueError(
                f"mesh_type must be one of {allowed_types}, got '{value}'"
            )
        self._mesh_type = value

    @property
    def source_frequency(self):
        return self._source_frequency

    @source_frequency.setter
    def source_frequency(self, value):
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

    def set_mesh(
        self,
        user_mesh=None,
        mesh_parameters={},
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

        # Setting default mesh parameters
        mesh_parameters.setdefault("periodic", False)
        mesh_parameters.setdefault("minimum_velocity", 1.5)
        mesh_parameters.setdefault("edge_length", None)
        mesh_parameters.setdefault("dx", None)
        mesh_parameters.setdefault("length_z", self._length_z)
        mesh_parameters.setdefault("length_x", self._length_x)
        mesh_parameters.setdefault("length_y", self._length_y)
        mesh_parameters.setdefault("abc_pad_length", self._abc_pad_length)
        mesh_parameters.setdefault("mesh_file", self.mesh_file)
        mesh_parameters.setdefault("dimension", self.dimension)
        mesh_parameters.setdefault("mesh_type", self.mesh_type)
        mesh_parameters.setdefault("source_frequency", self.source_frequency)
        mesh_parameters.setdefault("method", None)
        mesh_parameters.setdefault("degree", None)
        mesh_parameters.setdefault("velocity_model_file", None)
        mesh_parameters.setdefault("cell_type", None)
        mesh_parameters.setdefault("cells_per_wavelength", None)

        # Ensure all AutomaticMesh-required parameters are present
        required_keys = [
            "cell_type",
            "mesh_type",
            "abc_pad_length",
            "dx",
            "periodic",
            "edge_length",
            "cells_per_wavelength",
            "source_frequency",
            "velocity_model_file",
        ]
        for key in required_keys:
            if key not in mesh_parameters:
                mesh_parameters[key] = None

        self.length_z = mesh_parameters["length_z"]
        self.length_x = mesh_parameters["length_x"]
        self.length_y = mesh_parameters["length_y"]

        self.set_mesh_type(new_mesh_type=mesh_parameters.get("mesh_type"))

        automatic_mesh = self.mesh_type in {"firedrake_mesh", "SeismicMesh"}

        if user_mesh is not None:
            self.user_mesh = user_mesh
            self.mesh_type = "user_mesh"
        elif mesh_parameters["mesh_file"] is not None:
            self.mesh_file = mesh_parameters["mesh_file"]
            self.mesh_type = "file"
        elif automatic_mesh:
            self.user_mesh = self._creating_automatic_mesh(
                mesh_parameters=mesh_parameters
            )

        if (
            mesh_parameters["length_z"] is None
            or mesh_parameters["length_x"] is None
            or (mesh_parameters["length_y"] is None and self.dimension == 2)
        ) and self.mesh_type != "firedrake_mesh":
            raise ValueError("Mesh lengths must be specified for non-firedrake meshes.")

    def set_mesh_type(self, new_mesh_type=None):
        if new_mesh_type is not None:
            self.mesh_type = new_mesh_type

    def _set_mesh_length(
        self,
        length_z=None,
        length_x=None,
        length_y=None,
    ):
        if length_z is not None:
            self.length_z = length_z
        if length_x is not None:
            self.length_x = length_x
        if length_y is not None:
            self.length_y = length_y
