"""Meshing parameters module for Spyro.

This module provides classes and functions for managing mesh parameters,
including mesh type selection, dimension handling, and automatic mesh
generation based on wavelength constraints.
"""

import warnings
import os
from ..utils.error_management import value_parameter_error


def cells_per_wavelength(method, degree, dimension):
    """Retrieve the number of cells per wavelength for a given method configuration.

    Parameters
    ----------
    method : str
        The finite element method to use. Options include:
        'mass_lumped_triangle' or 'spectral_quadrilateral'.
    degree : int
        The polynomial degree of the finite element basis functions.
        Valid values are 2, 3, 4, 5, 6, or 8 depending on the method.
    dimension : int
        The spatial dimension of the problem (2 or 3).

    Returns
    -------
    float or None
        The number of cells per wavelength for the specified configuration.
        Returns None if the configuration is not defined in the dictionary.

    Notes
    -----
    The returned value represents the minimum number of mesh cells required
    per wavelength to maintain numerical accuracy for the specified method
    and degree combination.

    Examples
    --------
    >>> cells_per_wavelength('mass_lumped_triangle', 2, 2)
    7.02
    >>> cells_per_wavelength('mass_lumped_triangle', 3, 3)
    3.72
    """
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
    """Manage mesh parameters and configuration for seismic wave simulations.

    This class handles all aspects of mesh configuration including mesh type
    selection, dimensional parameters, boundary conditions, and automatic mesh
    generation based on velocity models and source frequencies.

    Attributes
    ----------
    input_mesh_dictionary : dict
        Dictionary containing initial mesh parameters.
    dimension : int
        Spatial dimension of the mesh (2 or 3).
    comm : MPI communicator
        MPI communicator for parallel computations.
    quadrilateral : bool
        Whether to use quadrilateral (True) or triangular (False) elements.
    mesh_type : str
        Type of mesh generation method. Options: 'firedrake_mesh',
        'user_mesh', 'SeismicMesh', 'file', or 'spyro_mesh'.
    method : str
        Finite element method. Options: 'mass_lumped_triangle',
        'DG_triangle', 'spectral_quadrilateral', 'DG_quadrilateral', or 'CG'.
    periodic : bool
        Whether the mesh has periodic boundary conditions.
    mesh_file : str
        Path to mesh file (.msh format).
    length_z : float
        Mesh length in the z-direction.
    length_x : float
        Mesh length in the x-direction.
    length_y : float
        Mesh length in the y-direction (for 3D meshes).
    user_mesh : object
        User-provided mesh object.
    output_filename : str
        Output filename for generated mesh (.msh format).
    source_frequency : float
        Source frequency in Hz for wavelength calculations.
    abc_pad_length : float
        Length of absorbing boundary condition padding layer.
    degree : int
        Polynomial degree of finite element basis functions.
    minimum_velocity : float
        Minimum velocity in the model for mesh size calculations.
    velocity_model : object
        Velocity model object for mesh adaptation.
    automatic_mesh : bool
        Whether mesh is automatically generated.
    edge_length : float
        Target edge length for mesh elements.
    cells_per_wavelength : float
        Number of cells per wavelength for mesh sizing.
    grid_velocity_data : dict
        Dictionary containing gridded velocity data with 'vp_values'
        and 'grid_spacing' keys.
    gradient_mask : object
        Mask for gradient calculations in inversions.
    negative_z : bool
        Whether z-axis points is always negative (True) or is positve (False).

    Notes
    -----
    The class enforces mutual exclusivity between 'edge_length' and
    'cells_per_wavelength' parameters. Setting one will clear the other.

    Mesh dimensions (length_x, length_y, length_z, abc_pad_length) are
    checked for unit consistency (meters vs kilometers) and must all use
    the same unit system.
    """

    def __init__(self, input_mesh_dictionary=None, dimension=None, source_frequency=None, comm=None, quadrilateral=False, method=None, degree=None, velocity_model=None, abc_pad_length=None, negative_z=True, use_defaults=True):
        """Initialize the MeshingParameters class.

        Parameters
        ----------
        input_mesh_dictionary : dict, optional
            Dictionary containing initial mesh parameters. Can include keys
            such as 'mesh_type', 'mesh_file', 'length_x', 'length_y',
            'length_z', 'user_mesh', and 'output_filename'. Default is None.
        dimension : int, optional
            Spatial dimension of the mesh (2 or 3). Default is None.
        source_frequency : float, optional
            Source frequency in Hz for wavelength-based mesh sizing.
            Should be in range [1.5, 50] for realistic FWI cases.
            Default is None.
        comm : MPI communicator, optional
            MPI communicator for parallel mesh operations. Default is None.
        quadrilateral : bool, optional
            If True, uses quadrilateral or hexahedral elements; if False,
            uses triangular or tetrahedral elements. Default is False.
        method : str, optional
            Finite element method. Options: 'mass_lumped_triangle',
            'DG_triangle', 'spectral_quadrilateral', 'DG_quadrilateral',
            or 'CG'. Default is None.
        degree : int, optional
            Polynomial degree of finite element basis functions.
            Default is None.
        velocity_model : object, optional
            Velocity model object for mesh adaptation. Default is None.
        abc_pad_length : float, optional
            Length of absorbing boundary condition padding layer.
            Default is None.
        negative_z : bool, optional
            If True, z-axis points downward; if False, z-axis points upward.
            Default is True.
        use_defaults : bool, optional
            If True, automatically call set_mesh with default parameters.
            Default is True.

        Notes
        -----
        If use_defaults is True, the set_mesh method is automatically called
        during initialization to configure the mesh with default values where
        there are missing parameters.
        """
        if input_mesh_dictionary is None:
            input_mesh_dictionary = {}
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
        self.output_filename = self.input_mesh_dictionary.get("output_filename", "automatic_mesh.msh")
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
        """Set a length attribute with automatic unit consistency checking.

        Parameters
        ----------
        attr_name : str
            Name of the attribute to set (e.g., '_length_x', '_length_z').
        value : float or None
            The length value to set. Values > 100 are assumed to be in meters,
            values <= 100 are assumed to be in kilometers.

        Raises
        ------
        ValueError
            If the value is negative or if the inferred unit is inconsistent
            with previously set dimensions.

        Warnings
        --------
        Issues a warning if the inferred unit (meters or km) appears to be
        inconsistent with the unit of previously set dimension attributes.

        Notes
        -----
        This method helps ensure all spatial dimensions use consistent units.
        The unit is inferred from the magnitude: values > 100 are assumed to
        be in meters, while values <= 100 are assumed to be in kilometers.
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
                f"{attr_name} value ({value}) appears to be "
                f"in {new_unit}, but the current unit is "
                f"{self._unit}. Please check for consistency."
            )
        if value is not None:
            if value < 0.0:
                raise ValueError(f"Please do not use negative value for {attr_name}")
        setattr(self, attr_name, value)

    @property
    def grid_velocity_data(self):
        """Get the gridded velocity data.

        Returns
        -------
        dict or None
            Dictionary containing 'vp_values' and 'grid_spacing' keys,
            or None if not set.
        """
        return self._grid_velocity_data

    @grid_velocity_data.setter
    def grid_velocity_data(self, value):
        """Set attribute of the velocity data.

        Parameters
        ----------
        value : dict or None
            Dictionary containing gridded velocity information.
            Must include 'vp_values' and 'grid_spacing' keys.

        Raises
        ------
        ValueError
            If value is not None and does not contain required keys
            'vp_values' and 'grid_spacing'.
        """
        if value is not None:
            necessary_keys = ["vp_values", "grid_spacing"]
            for necessary_key in necessary_keys:
                if necessary_key not in value:
                    raise ValueError(f"Grid velocity data needs {necessary_key} key.")
        self._grid_velocity_data = value

    @property
    def output_filename(self):
        """Get the output filename for mesh generation.

        Returns
        -------
        str or None
            The output filename with .msh extension, or None if not set.
        """
        return self._output_filename

    @output_filename.setter
    def output_filename(self, value):
        """Set the output filename for mesh generation.

        Parameters
        ----------
        value : str or None
            Output filename. Must end with .msh extension.

        Raises
        ------
        ValueError
            If value does not end with .msh extension (except .vtk which
            issues a warning).

        Warnings
        --------
        Issues a warning if .vtk extension is used, as VTK meshes are for
        visualization only and cannot be used for simulation.
        """
        if value is not None:
            if isinstance(value, str) and value.endswith('.vtk'):
                warnings.warn("VTK meshes for visualization only, will not run a simulation.")
            elif not (isinstance(value, str) and value.endswith('.msh')):
                raise ValueError(f"mesh_file '{value}' must be a .msh file")
        self._output_filename = value

    @property
    def edge_length(self):
        """Get the target edge length for mesh elements.

        Returns
        -------
        float or None
            The target edge length for mesh elements, or None if not set.
        """
        return self._edge_length

    @edge_length.setter
    def edge_length(self, value):
        """Set the target edge length for mesh elements.

        Parameters
        ----------
        value : float or None
            The target edge length for mesh elements.

        Warnings
        --------
        Setting edge_length will clear any previously set cells_per_wavelength
        value, as these parameters are mutually exclusive.

        Notes
        -----
        Only one of edge_length or cells_per_wavelength can be set at a time.
        Setting this property will automatically set cells_per_wavelength to None.
        """
        if self.cells_per_wavelength is not None:
            warnings.warn(
                "Mutual exclusion: Both 'edge_length' and "
                "'cells_per_wavelength' control mesh size, "
                "but only one can be set at a time. Setting "
                "'edge_length' will override and remove the "
                "previously set 'cells_per_wavelength'. If "
                "you wish to use 'cells_per_wavelength' instead, "
                "set it after setting 'edge_length'."
            )
            self.cells_per_wavelength = None
        self._edge_length = value

    @property
    def cells_per_wavelength(self):
        """Get the number of cells per wavelength for mesh sizing.

        Returns
        -------
        float or None
            The number of cells per wavelength, or None if not set.
        """
        return self._cells_per_wavelength

    @cells_per_wavelength.setter
    def cells_per_wavelength(self, value):
        """Set the number of cells per wavelength for mesh sizing.

        Parameters
        ----------
        value : float or None
            The desired number of cells per wavelength.

        Warnings
        --------
        Setting cells_per_wavelength will clear any previously set edge_length
        value, as these parameters are mutually exclusive.

        Notes
        -----
        Only one of cells_per_wavelength or edge_length can be set at a time.
        Setting this property will automatically set edge_length to None.
        """
        if self.edge_length is not None:
            warnings.warn("Setting cells_per_wavelength"
                          "removes edge_length parameter")
            self.edge_length = None
        self._cells_per_wavelength = value

    @property
    def method(self):
        """Get the finite element method.

        Returns
        -------
        str or None
            The finite element method name, or None if not set.
        """
        return self._method

    @method.setter
    def method(self, value):
        """Set the finite element method.

        Parameters
        ----------
        value : str or None
            The finite element method to use. Must be one of:
            'mass_lumped_triangle', 'DG_triangle', 'spectral_quadrilateral',
            'DG_quadrilateral', or 'CG'.

        Raises
        ------
        ValueError
            If value is not None and not one of the allowed method types.
        """
        allowed_types = [
            "mass_lumped_triangle",
            "DG_triangle",
            "spectral_quadrilateral",
            "DG_quadrilateral",
            "CG",
        ]

        if value is not None and value not in allowed_types:
            value_parameter_error('method', value, allowed_types)

        self._method = value

    @property
    def mesh_file(self):
        """Get the path to the mesh file.

        Returns
        -------
        str or None
            The path to the mesh file, or None if not set.
        """
        return self._mesh_file

    @mesh_file.setter
    def mesh_file(self, value):
        """Set the path to the mesh file.

        Parameters
        ----------
        value : str or None
            Path to the mesh file. Must end with .msh extension and exist
            in the filesystem.

        Raises
        ------
        ValueError
            If value does not end with .msh extension.
        FileNotFoundError
            If the specified file does not exist.
        """
        if value is not None:
            if not (isinstance(value, str) and value.endswith('.msh')):
                raise ValueError(f"mesh_file '{value}' must be a .msh file")
            if not os.path.exists(value):
                raise FileNotFoundError(f"mesh_file '{value}' does not exist")
        self._mesh_file = value

    @property
    def mesh_type(self):
        """Get the mesh generation type.

        Returns
        -------
        str or None
            The mesh generation type, or None if not set.
        """
        return self._mesh_type

    @mesh_type.setter
    def mesh_type(self, value):
        """Set the mesh generation type.

        Parameters
        ----------
        value : str or None
            The mesh generation type. Must be one of: 'firedrake_mesh',
            'user_mesh', 'SeismicMesh', 'file', or 'spyro_mesh'.

        Raises
        ------
        ValueError
            If value is not one of the allowed mesh types, or if
            'SeismicMesh' is selected with quadrilateral elements
            (not supported).
        """
        allowed_types = ["firedrake_mesh", "user_mesh", "SeismicMesh", "file", "spyro_mesh"]
        if value is not None and value not in allowed_types:
            value_parameter_error("mesh_type", value, allowed_types)
        if value == "SeismicMesh" and self.quadrilateral:
            raise ValueError("SeismicMesh does not work with quads.")
        self._mesh_type = value

    @property
    def source_frequency(self):
        """Get the source frequency.

        Returns
        -------
        float or None
            The source frequency in Hz, or None if not set.
        """
        return self._source_frequency

    @source_frequency.setter
    def source_frequency(self, value):
        """Set the source frequency for wavelength calculations.

        Parameters
        ----------
        value : float, int, or None
            The source frequency in Hz. Should be in range [1.5, 50]
            for realistic FWI applications.

        Raises
        ------
        TypeError
            If value is not None and not a number.

        Warnings
        --------
        Issues a warning if frequency < 1.5 Hz (too low for realistic FWI)
        or if frequency > 50 Hz (too high, should apply low-pass filter).
        """
        if value is None:
            self._source_frequency = value
        elif not isinstance(value, (int, float)):
            raise TypeError("Source frequency must be a number"
                            f", got {type(value).__name__}")
        else:
            if value < 1.5:
                warnings.warn(f"Source frequency of {value} "
                              "too low for realistic FWI case")
            elif value > 50:
                warnings.warn(f"Source frequency of {value} too high for "
                              "realistic FWI case, please low-pass filter")
            self._source_frequency = value

    @property
    def abc_pad_length(self):
        """Get the absorbing boundary condition padding length.

        Returns
        -------
        float or None
            The ABC padding length, or None if not set.
        """
        return self._abc_pad_length

    @abc_pad_length.setter
    def abc_pad_length(self, value):
        """Set the absorbing boundary condition padding length.

        Parameters
        ----------
        value : float or None
            The length of the ABC padding layer. Must be non-negative.

        Raises
        ------
        ValueError
            If value is negative or if the unit appears inconsistent with
            other dimension attributes.
        """
        self._set_length_with_unit_check("_abc_pad_length", value)

    @property
    def length_z(self):
        """Get the mesh extent in the z-direction.

        Returns
        -------
        float or None
            The mesh extent in the z-direction, or None if not set.
        """
        return self._length_z

    @length_z.setter
    def length_z(self, value):
        """Set the mesh extent in the z-direction.

        Parameters
        ----------
        value : float or None
            The mesh extent in the z-direction. Must be non-negative.

        Raises
        ------
        ValueError
            If value is negative or if the inferred unit appears inconsistent
            with other dimension attributes.
        """
        self._set_length_with_unit_check("_length_z", value)

    @property
    def length_x(self):
        """Get the mesh extent in the x-direction.

        Returns
        -------
        float or None
            The mesh extent in the x-direction, or None if not set.
        """
        return self._length_x

    @length_x.setter
    def length_x(self, value):
        """Set the mesh extent in the x-direction.

        Parameters
        ----------
        value : float or None
            The mesh extent in the x-direction. Must be non-negative.

        Raises
        ------
        ValueError
            If value is negative or if the inferred unit appears inconsistent
            with other dimension attributes.
        """
        self._set_length_with_unit_check("_length_x", value)

    @property
    def length_y(self):
        """Get the mesh extent in the y-direction.

        Returns
        -------
        float or None
            The mesh extent in the y-direction, or None if not set.
        """
        return self._length_y

    @length_y.setter
    def length_y(self, value):
        """Set the mesh extent in the y-direction (for 3D meshes).

        Parameters
        ----------
        value : float or None
            The mesh extent in the y-direction. Must be non-negative.

        Raises
        ------
        ValueError
            If value is negative or if the inferred unit appears inconsistent
            with other dimension attributes.
        """
        self._set_length_with_unit_check("_length_y", value)

    @property
    def user_mesh(self):
        """Get the user-provided mesh object.

        Returns
        -------
        object or None
            The user-provided mesh object, or None if not set.
        """
        return self._user_mesh

    @user_mesh.setter
    def user_mesh(self, value):
        """Set a user-provided mesh object.

        Parameters
        ----------
        value : object or None
            A user-provided mesh object.

        Notes
        -----
        Setting a user mesh automatically changes mesh_type to 'user_mesh'.
        """
        if value is not None:
            self.mesh_type = "user_mesh"
        self._user_mesh = value

    @property
    def periodic(self):
        """Get the periodic boundary condition flag.

        Returns
        -------
        bool
            True if periodic boundary conditions are enabled, False otherwise.
        """
        return self._periodic

    @periodic.setter
    def periodic(self, value):
        """Set the periodic boundary condition flag.

        Parameters
        ----------
        value : bool
            If True, enable periodic boundary conditions.

        Raises
        ------
        ValueError
            If value is True but mesh_type is not 'firedrake_mesh',
            as periodic meshes are only supported with Firedrake meshes.
        """
        if self.mesh_type != "firedrake_mesh" and value is True:
            raise ValueError("Periodic meshes are only supported "
                             "with Firedrake meshes for now.")
        self._periodic = value

    def set_mesh(
        self,
        user_mesh=None,
        input_mesh_parameters={},
        abc_pad_length=None,
    ):
        """Configure mesh parameters with user-provided values and defaults.

        This method updates mesh parameters by merging user-provided values
        with current attribute values, ensuring all necessary mesh
        configuration is complete.

        Parameters
        ----------
        user_mesh : spyro.Mesh, optional
            A user-provided mesh object. Default is None.
        input_mesh_parameters : dict, optional
            Dictionary of mesh parameters to set. Can include any attribute
            of the MeshingParameters class. Unspecified parameters will use
            current attribute values as defaults.
        abc_pad_length : float, optional
            Length of absorbing boundary condition padding layer.
            Overrides the value from input_mesh_parameters if provided.
            Default is None.

        Notes
        -----
        This method performs the following steps:

        1. Sets abc_pad_length if provided
        2. Populates input_mesh_parameters with default values from current
           attributes for any unspecified keys
        3. Updates all class attributes that have non-None values in
           input_mesh_parameters
        4. Updates the automatic_mesh flag based on the final mesh_type

        The method only sets attributes that already exist in the class and
        have non-None values in the input_mesh_parameters dictionary.

        Examples
        --------
        >>> mp = MeshingParameters(dimension=2, use_defaults=False)
        >>> mp.set_mesh(input_mesh_parameters={'mesh_type': 'firedrake_mesh',
        ...                                     'length_x': 10.0,
        ...                                     'length_z': 5.0})
        """
        if abc_pad_length is not None:
            self.abc_pad_length = abc_pad_length

        # Setting default mesh parameters
        input_mesh_parameters.setdefault("periodic", self.periodic)
        input_mesh_parameters.setdefault("minimum_velocity",
                                         self.minimum_velocity)
        input_mesh_parameters.setdefault("length_z", self.length_z)
        input_mesh_parameters.setdefault("length_x", self.length_x)
        input_mesh_parameters.setdefault("length_y", self.length_y)
        input_mesh_parameters.setdefault("abc_pad_length", self.abc_pad_length)
        input_mesh_parameters.setdefault("mesh_file", self.mesh_file)
        input_mesh_parameters.setdefault("dimension", self.dimension)
        input_mesh_parameters.setdefault("mesh_type", self.mesh_type)
        input_mesh_parameters.setdefault("source_frequency",
                                         self.source_frequency)
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
