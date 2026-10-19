"""Meshing parameters module for Spyro.

This module provides classes and functions for managing mesh parameters,
including mesh type selection, dimension handling, and automatic mesh
generation based on wavelength constraints.
"""

import difflib
import os
from typing import Any
from warnings import warn

from ..mpi.spyro_mpi import SpyroEnsemble
from .gmsh_options import GmshOptions
from ..utils.error_management import (
    validate_boolean,
    validate_data_structure,
    validate_file,
    validate_numeric,
    validate_parameter,
)


_CELLS_PER_WAVELENGTH = {
    ("mass_lumped_triangle", 2, 2): 7.02,
    ("mass_lumped_triangle", 3, 2): 3.70,
    ("mass_lumped_triangle", 4, 2): 2.67,
    ("mass_lumped_triangle", 5, 2): 2.03,
    ("mass_lumped_triangle", 2, 3): 6.12,
    ("mass_lumped_triangle", 3, 3): 3.72,
}

MESH_TYPES = ("firedrake_mesh", "user_mesh", "SeismicMesh", "file", "spyro_mesh", "gmsh_mesh")
AUTOMATIC_MESH_TYPES = frozenset({"firedrake_mesh", "SeismicMesh", "spyro_mesh", "gmsh_mesh"})
METHODS = ("mass_lumped_triangle", "DG_triangle", "spectral_quadrilateral", "DG_quadrilateral", "CG")
_QUADRILATERAL_METHODS = frozenset({"spectral_quadrilateral", "DG_quadrilateral"})
_TRIANGLE_METHODS = frozenset({"mass_lumped_triangle", "DG_triangle"})
_UNIT_ALIASES = {"m": "meters", "meters": "meters", "km": "km"}

_DOMAIN_LENGTHS = ("length_z", "length_x", "length_y")
_LENGTHS = _DOMAIN_LENGTHS + (
    "abc_pad_length", "padding_x", "padding_y", "padding_z", "h_padding", "min_element_size",
)
# Lengths that fall back to ``abc_pad_length`` when unset.
_PADDINGS = frozenset({"padding_x", "padding_y", "padding_z"})
# Lengths whose default is given in meters and converted to the domain unit.
_DEFAULTS_IN_METERS = {"h_padding": 500.0, "min_element_size": 35.0}
_EDGE_LENGTHS = ("edge_length", "edge_length_z", "edge_length_x", "edge_length_y")


def cells_per_wavelength(method: str, degree: int, dimension: int) -> float | None:
    """Retrieve the number of cells per wavelength for a given method configuration.

    Parameters
    ----------
    method : str
        The finite element method, for example 'mass_lumped_triangle'.
    degree : int
        The polynomial degree of the finite element basis functions.
    dimension : int
        The spatial dimension of the problem (2 or 3).

    Returns
    -------
    float or None
        The minimum number of cells per wavelength needed for accuracy, or
        None if no value is known for this configuration.

    Examples
    --------
    >>> cells_per_wavelength('mass_lumped_triangle', 2, 2)
    7.02
    >>> cells_per_wavelength('mass_lumped_triangle', 3, 3)
    3.72
    """
    return _CELLS_PER_WAVELENGTH.get((method, degree, dimension))


def _guess_unit(value: float | None) -> str | None:
    """Guess the length unit of a value from its magnitude.

    Parameters
    ----------
    value : float or None
        A length.

    Returns
    -------
    str or None
        'meters' for values greater than 100, 'km' for other positive values,
        and None for None or zero, which carry no unit information.
    """
    if value is None or value == 0:
        return None
    return "meters" if value > 100 else "km"


def _get(dictionary: dict, key: str, default: Any = None, keyword_value: Any = None) -> Any:
    """Read a parameter from a dictionary, with an optional keyword override.

    A None entry in the dictionary is treated as missing.

    Parameters
    ----------
    dictionary : dict
        Input dictionary.
    key : str
        Key of the entry in ``dictionary``.
    default : Any, optional
        Value used when neither source provides one. Default is None.
    keyword_value : Any, optional
        Value given as an explicit keyword argument, or None if it was not
        given. Default is None.

    Returns
    -------
    Any
        The keyword value if it is not None, otherwise the dictionary value
        if it is not None, otherwise ``default``.

    Warns
    -----
    UserWarning
        If both sources provide different values.
    """
    dict_value = dictionary.get(key)
    if keyword_value is None:
        return default if dict_value is None else dict_value
    if dict_value is not None and dict_value is not keyword_value:
        try:
            conflict = bool(dict_value != keyword_value)
        except (TypeError, ValueError):
            conflict = True
        if conflict:
            warn(
                f"'{key}' was given both as a keyword argument ({keyword_value!r}) "
                f"and in input_mesh_dictionary ({dict_value!r}). Using the keyword argument."
            )
    return keyword_value


def _length_property(name: str, doc: str) -> property:
    """Build a property for a length stored in ``MeshingParameters._lengths``.

    Parameters
    ----------
    name : str
        Name of the length, a key of ``_lengths``.
    doc : str
        Docstring of the property.

    Returns
    -------
    property
        Property whose getter applies the fallbacks of
        :meth:`MeshingParameters._length` and whose setter validates the
        value with :meth:`MeshingParameters._check_length`.
    """
    def getter(self: "MeshingParameters") -> float | None:
        return self._length(name)

    def setter(self: "MeshingParameters", value: float | None) -> None:
        self._lengths[name] = self._check_length(name, value)

    return property(getter, setter, doc=doc)


def _edge_length_property(name: str, doc: str) -> property:
    """Build a property for an edge length stored in ``MeshingParameters._edge_lengths``.

    Setting a value clears ``cells_per_wavelength`` with a warning, since
    both control the mesh size.

    Parameters
    ----------
    name : str
        Name of the edge length, a key of ``_edge_lengths``.
    doc : str
        Docstring of the property.

    Returns
    -------
    property
        Property that reads and validates the edge length.
    """
    def getter(self: "MeshingParameters") -> float | None:
        return self._edge_lengths[name]

    def setter(self: "MeshingParameters", value: float | None) -> None:
        validate_numeric(name, value, accept_parameter_as_none=True, lower_bound=0)
        if value is not None and self._cells_per_wavelength is not None:
            warn(
                f"Mutual exclusion: Both '{name}' and 'cells_per_wavelength' control "
                f"mesh size, but only one can be set at a time. Setting '{name}' will "
                "override and remove the previously set 'cells_per_wavelength'. If you "
                "wish to use 'cells_per_wavelength' instead, set it after setting "
                f"'{name}'."
            )
            self._cells_per_wavelength = None
        self._edge_lengths[name] = value

    return property(getter, setter, doc=doc)


class MeshingParameters:
    """Manage mesh parameters and configuration for seismic wave simulations.

    Attributes
    ----------
    dimension : int or None
        Spatial dimension of the mesh (2 or 3).
    comm : MPI communicator or None
        MPI communicator for parallel computations.
    quadrilateral : bool
        Whether to use quadrilateral (True) or triangular (False) elements.
    method : str or None
        Finite element method, one of ``METHODS``.
    degree : int or None
        Polynomial degree of finite element basis functions.
    negative_z : bool
        Whether the z-axis is always negative (True) or positive (False).
    mesh_type : str or None
        Mesh generation method, one of ``MESH_TYPES``.
    automatic_mesh : bool
        Whether the mesh is generated automatically. Derived from ``mesh_type``.
    periodic : bool
        Whether the mesh has periodic boundary conditions.
    mesh_file : str or None
        Path to an existing mesh file (.msh).
    user_mesh : object or None
        User-provided mesh object.
    output_filename : str or None
        Output filename for a generated mesh (.msh, or .vtk for visualization).
    unit : str or None
        Length unit, 'meters' or 'km'. Inferred from the domain lengths
        unless given explicitly.
    length_z, length_x, length_y : float or None
        Domain extents.
    abc_pad_length : float or None
        Length of the absorbing boundary condition padding layer.
    padding_x, padding_y, padding_z : float or None
        Gmsh padding sizes. Each falls back to ``abc_pad_length``.
    source_frequency : float or None
        Source frequency in Hz for wavelength calculations.
    edge_length : float or None
        Target edge length for mesh elements.
    edge_length_z, edge_length_x, edge_length_y : float or None
        Direction-specific edge lengths, overriding ``edge_length``.
    cells_per_wavelength : float or None
        Number of cells per wavelength for mesh sizing.
    minimum_velocity : float or None
        Minimum velocity in the model, for mesh sizing.
    velocity_model : object or None
        Velocity model for mesh sizing, for example a grid dictionary with a
        'vp_values' key or a path to a SEGY file.
    segy_velocity_model : str or None
        Deprecated SEGY path. Setting it also sets ``velocity_model``.
    grid_velocity_data : dict or None
        Gridded velocity data with a 'vp_values' key and grid spacing keys.
    gradient_mask : dict or None
        Mask for gradient calculations in inversions.
    is_complete : bool
        Whether all parameters needed to build the mesh are set. Derived.
    h_padding : float
        Constant padding element size used if ``extend_segy`` is False.
        Defaults to 500 m in the domain unit.
    min_element_size : float
        Gmsh structured element size. Defaults to 35 m in the domain unit.
    gmsh : GmshOptions
        Options that only affect Gmsh mesh generation. Their keys are given
        flat in the input dictionary, for example 'grade'.

    Notes
    -----
    Lengths, edge lengths, ``cells_per_wavelength``, ``source_frequency``,
    ``mesh_file`` and ``output_filename`` are checked when assigned. Every
    other parameter, and every rule involving several parameters, is checked
    by :meth:`validate`. It runs at the end of ``__init__`` and
    :meth:`set_mesh`, when ``mesh_type`` or ``periodic`` are assigned, and
    before a mesh is generated, so checks do not depend on assignment order.

    ``edge_length`` (and its directional variants) and
    ``cells_per_wavelength`` are mutually exclusive: setting one clears the
    other with a warning.

    The length unit is inferred from the domain lengths unless 'unit' is
    given. Lengths that look inconsistent with it produce a warning.
    """

    # Keys accepted in ``input_mesh_dictionary`` and by ``set_mesh``.
    INPUT_KEYS = (
        "unit", "dimension", "quadrilateral", "method", "degree", "negative_z",
        "mesh_type", "periodic", "mesh_file", "user_mesh", "output_filename",
        *_LENGTHS, "source_frequency", "cells_per_wavelength", *_EDGE_LENGTHS,
        "minimum_velocity", "velocity_model", "segy_velocity_model",
        "grid_velocity_data", "gradient_mask", *GmshOptions.keys(),
    )

    length_z = _length_property("length_z", "Mesh extent in the z-direction.")
    length_x = _length_property("length_x", "Mesh extent in the x-direction.")
    length_y = _length_property("length_y", "Mesh extent in the y-direction.")
    abc_pad_length = _length_property("abc_pad_length", "Absorbing boundary layer length.")
    padding_x = _length_property("padding_x", "Padding in x; defaults to abc_pad_length.")
    padding_y = _length_property("padding_y", "Padding in y; defaults to abc_pad_length.")
    padding_z = _length_property("padding_z", "Padding in z; defaults to abc_pad_length.")
    h_padding = _length_property("h_padding", "Padding element size; defaults to 500 m.")
    min_element_size = _length_property("min_element_size", "Structured element size; defaults to 35 m.")
    edge_length = _edge_length_property("edge_length", "Target edge length.")
    edge_length_z = _edge_length_property("edge_length_z", "Target edge length in z.")
    edge_length_x = _edge_length_property("edge_length_x", "Target edge length in x.")
    edge_length_y = _edge_length_property("edge_length_y", "Target edge length in y.")

    def __init__(
        self,
        input_mesh_dictionary: dict | None = None,
        dimension: int | None = None,
        source_frequency: float | None = None,
        comm: Any = None,
        quadrilateral: bool | None = None,
        method: str | None = None,
        degree: int | None = None,
        velocity_model: Any = None,
        abc_pad_length: float | None = None,
        negative_z: bool | None = None,
    ) -> None:
        """Initialize the MeshingParameters class.

        Explicit keyword arguments take precedence over the same key in
        ``input_mesh_dictionary``.

        Parameters
        ----------
        input_mesh_dictionary : dict, optional
            Initial mesh parameters. Accepted keys are listed in
            ``MeshingParameters.INPUT_KEYS``. Default is None.
        dimension : int, optional
            Spatial dimension of the mesh (2 or 3). Default is None.
        source_frequency : float, optional
            Source frequency in Hz for wavelength-based mesh sizing.
            Default is None.
        comm : MPI communicator, optional
            MPI communicator for parallel mesh operations. Default is None.
        quadrilateral : bool, optional
            If True, uses quadrilateral or hexahedral elements. Default is
            None, which means False unless the dictionary says otherwise.
        method : str, optional
            Finite element method, one of ``METHODS``. Default is None.
        degree : int, optional
            Polynomial degree of finite element basis functions.
            Default is None.
        velocity_model : object, optional
            Velocity model object for mesh adaptation. Default is None.
        abc_pad_length : float, optional
            Length of absorbing boundary condition padding layer.
            Default is None.
        negative_z : bool, optional
            If True, z-axis points downward. Default is None, which means
            True unless the dictionary says otherwise.

        Raises
        ------
        ValueError
            If a parameter is invalid or parameters are inconsistent.
        TypeError
            If a parameter has the wrong type.
        """
        d = input_mesh_dictionary or {}
        self.comm = comm
        # Cross-parameter checks are deferred until every value is set.
        self._batch_update = True

        # Storage for properties, declared before any setter runs.
        self._unit = None
        self._lengths = dict.fromkeys(_LENGTHS)
        self._edge_lengths = dict.fromkeys(_EDGE_LENGTHS)
        self._cells_per_wavelength = None
        self._source_frequency = None
        self._mesh_type = None
        self._periodic = False
        self._mesh_file = None
        self._user_mesh = None
        self._output_filename = None
        self._segy_velocity_model = None
        self._grid_velocity_data = None

        # Discretization
        self.dimension = _get(d, "dimension", keyword_value=dimension)
        self.quadrilateral = _get(d, "quadrilateral", False, keyword_value=quadrilateral)
        self.method = _get(d, "method", keyword_value=method)
        self.degree = _get(d, "degree", keyword_value=degree)
        self.negative_z = _get(d, "negative_z", True, keyword_value=negative_z)

        # Mesh source
        self.mesh_type = _get(d, "mesh_type")
        self.periodic = _get(d, "periodic", False)
        self.mesh_file = _get(d, "mesh_file")
        self.user_mesh = _get(d, "user_mesh")
        self.output_filename = _get(d, "output_filename", "automatic_mesh.msh")

        # Lengths: the unit and domain lengths first, since they define the unit.
        self.unit = _get(d, "unit")
        for name in _LENGTHS:
            keyword = abc_pad_length if name == "abc_pad_length" else None
            setattr(self, name, _get(d, name, keyword_value=keyword))

        # Mesh size
        self.source_frequency = _get(d, "source_frequency", keyword_value=source_frequency)
        self.cells_per_wavelength = _get(d, "cells_per_wavelength")
        for name in _EDGE_LENGTHS:
            setattr(self, name, _get(d, name))

        # Velocity inputs
        self.minimum_velocity = _get(d, "minimum_velocity")
        self.velocity_model = _get(d, "velocity_model", keyword_value=velocity_model)
        self.segy_velocity_model = _get(d, "segy_velocity_model")
        self.grid_velocity_data = _get(d, "grid_velocity_data")
        self.gradient_mask = _get(d, "gradient_mask")

        self.gmsh = GmshOptions.from_dictionary(d)

        self._batch_update = False
        self.validate()

    # Validation and completeness
    def validate(self) -> None:
        """Check plain-attribute parameters and rules involving several parameters.

        Raises
        ------
        ValueError
            If a parameter is out of range or parameters are inconsistent.
        TypeError
            If a parameter has the wrong type.
        """
        validate_parameter("dimension", self.dimension, [None, 2, 3])
        validate_parameter("method", self.method, [*METHODS, None])
        validate_numeric("degree", self.degree, float_num=False, accept_parameter_as_none=True,
                         lower_bound=1, include_lower_bound=True)
        validate_numeric("minimum_velocity", self.minimum_velocity,
                         accept_parameter_as_none=True, lower_bound=0)
        validate_boolean("quadrilateral", self.quadrilateral)
        validate_boolean("negative_z", self.negative_z)
        self.gmsh.validate(self.dimension)

        if self.periodic and self.mesh_type != "firedrake_mesh":
            raise ValueError("Periodic meshes are only supported with Firedrake meshes for now.")
        if self.mesh_type == "SeismicMesh" and self.quadrilateral:
            raise ValueError("SeismicMesh does not work with quads.")
        if self.method in _QUADRILATERAL_METHODS and not self.quadrilateral:
            raise ValueError(f"Method '{self.method}' requires quadrilateral=True.")
        if self.method in _TRIANGLE_METHODS and self.quadrilateral:
            raise ValueError(f"Method '{self.method}' requires quadrilateral=False.")

    def _validate_if_not_batch(self) -> None:
        """Run :meth:`validate` unless a batch update is in progress.

        Raises
        ------
        ValueError
            If the parameters are inconsistent.
        """
        if not self._batch_update:
            self.validate()

    def missing_parameters(self) -> list[str]:
        """List the reasons why the mesh cannot be built yet.

        Automatic meshes need the domain lengths and either an edge length or
        ``cells_per_wavelength``, which in turn needs ``source_frequency`` and
        a velocity input. File meshes need ``mesh_file`` and user meshes need
        ``user_mesh``.

        Returns
        -------
        list of str
            Descriptions of what is missing. Empty if complete.
        """
        if self.mesh_type is None:
            return ["'mesh_type' is not set."]
        if self.mesh_type == "file":
            return [] if self.mesh_file is not None else ["'mesh_type' is 'file' but 'mesh_file' is not set."]
        if self.mesh_type == "user_mesh":
            return [] if self.user_mesh is not None else ["'mesh_type' is 'user_mesh' but 'user_mesh' object is not set."]

        missing = []
        required = _DOMAIN_LENGTHS if self.dimension == 3 else _DOMAIN_LENGTHS[:2]
        missing += [f"'{self.mesh_type}' requires '{name}'." for name in required if self._lengths[name] is None]

        has_edge_length = any(v is not None for v in self._edge_lengths.values())
        if not has_edge_length and self.cells_per_wavelength is None:
            missing.append(f"'{self.mesh_type}' requires either 'edge_length' or 'cells_per_wavelength'.")

        if self.cells_per_wavelength is not None:
            if self.source_frequency is None:
                missing.append("'cells_per_wavelength' requires 'source_frequency'.")
            if self.minimum_velocity is None and self.grid_velocity_data is None and self.velocity_model is None:
                missing.append(
                    "'cells_per_wavelength' requires 'minimum_velocity', "
                    "'grid_velocity_data', or 'velocity_model'."
                )
        return missing

    @property
    def is_complete(self) -> bool:
        """Whether :meth:`missing_parameters` is empty."""
        return not self.missing_parameters()

    def check_completeness(self, verbose: bool = True) -> bool:
        """Check if mesh parameters are complete for mesh generation.

        Parameters
        ----------
        verbose : bool, optional
            If True, print each missing parameter on rank 0. Default is True.

        Returns
        -------
        bool
            True if the parameters are complete, False otherwise.
        """
        missing = self.missing_parameters()
        if verbose:
            for reason in missing:
                SpyroEnsemble.print(f"Mesh incomplete: {reason}")
        return not missing

    # Bulk update
    def set_mesh(
        self,
        user_mesh: Any = None,
        input_mesh_parameters: dict | None = None,
        abc_pad_length: float | None = None,
    ) -> None:
        """Update mesh parameters after initialization.

        A None value means "leave unchanged". The unit and domain lengths are
        applied first so that other lengths are checked against them. Rules
        involving several parameters are checked once, after all values are set.

        Parameters
        ----------
        user_mesh : firedrake.MeshGeometry, optional
            A user-provided mesh object. Default is None.
        input_mesh_parameters : dict, optional
            Mesh parameters to update. Accepted keys are listed in
            ``MeshingParameters.INPUT_KEYS``. Default is None.
        abc_pad_length : float, optional
            Length of absorbing boundary condition padding layer. Takes
            precedence over the same key in ``input_mesh_parameters``.
            Default is None.

        Raises
        ------
        ValueError
            If ``input_mesh_parameters`` contains an unknown key, a value is
            invalid, or the resulting parameters are inconsistent.

        Examples
        --------
        >>> mp = MeshingParameters(dimension=2)
        >>> mp.set_mesh(input_mesh_parameters={'mesh_type': 'firedrake_mesh',
        ...                                     'length_x': 10.0,
        ...                                     'length_z': 5.0})
        """
        updates = dict(input_mesh_parameters or {})
        unknown = [key for key in updates if key not in self.INPUT_KEYS]
        if unknown:
            hints = []
            for key in unknown:
                close = difflib.get_close_matches(key, self.INPUT_KEYS, n=1)
                hints.append(f"'{key}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
            raise ValueError(f"Unknown mesh parameter(s): {', '.join(hints)}.")

        updates["abc_pad_length"] = _get(updates, "abc_pad_length", keyword_value=abc_pad_length)
        first = ("unit",) + _DOMAIN_LENGTHS
        ordered_keys = [key for key in first if key in updates]
        ordered_keys += [key for key in updates if key not in first]

        self._batch_update = True
        try:
            gmsh_keys = GmshOptions.keys()
            for key in ordered_keys:
                if updates[key] is not None:
                    setattr(self.gmsh if key in gmsh_keys else self, key, updates[key])
            if user_mesh is not None:
                self.user_mesh = user_mesh
        finally:
            self._batch_update = False
        self.validate()

    # Units and lengths
    @property
    def unit(self) -> str | None:
        """Length unit, 'meters' or 'km', given explicitly or inferred from the domain lengths."""
        return self._unit if self._unit is not None else self._infer_unit()

    @unit.setter
    def unit(self, value: str | None) -> None:
        if value is not None:
            value = _UNIT_ALIASES[validate_parameter("unit", value, list(_UNIT_ALIASES))]
        self._unit = value

    def _infer_unit(self, exclude: str | None = None) -> str | None:
        """Infer the length unit from the domain lengths.

        Parameters
        ----------
        exclude : str, optional
            Name of a domain length to ignore. Default is None.

        Returns
        -------
        str or None
            The unit guessed from the first non-zero domain length, or None.
        """
        for name in _DOMAIN_LENGTHS:
            unit = _guess_unit(self._lengths[name]) if name != exclude else None
            if unit is not None:
                return unit
        return None

    def _check_length(self, name: str, value: float | None) -> float | None:
        """Validate a length and warn if it looks inconsistent with the unit.

        Domain lengths are compared with the unit in both directions. Other
        lengths (pads and element sizes) can be small in meters, so they only
        warn when they look like meters in a km domain.

        Parameters
        ----------
        name : str
            Name of the length.
        value : float or None
            Length to check.

        Returns
        -------
        float or None
            ``value`` unchanged.

        Raises
        ------
        TypeError
            If the value is not a number.
        ValueError
            If the value is negative.

        Warns
        -----
        UserWarning
            If the guessed unit of ``value`` disagrees with the domain unit.
        """
        validate_numeric(name, value, accept_parameter_as_none=True,
                         lower_bound=0.0, include_lower_bound=True)
        reference = self._unit if self._unit is not None else self._infer_unit(exclude=name)
        guessed = _guess_unit(value)
        if reference is not None and guessed is not None:
            if name in _DOMAIN_LENGTHS:
                mismatch = guessed != reference
            else:
                mismatch = guessed == "meters" and reference == "km"
            if mismatch:
                warn(
                    f"{name} value ({value}) appears to be in {guessed}, but the "
                    f"current unit is {reference}. Please check for consistency."
                )
        return value

    def _length(self, name: str) -> float | None:
        """Read a stored length, applying its fallback or default.

        Parameters
        ----------
        name : str
            Name of the length.

        Returns
        -------
        float or None
            The stored value. If unset, paddings fall back to
            ``abc_pad_length`` and ``h_padding``/``min_element_size`` to their
            metre defaults converted to the domain unit.
        """
        value = self._lengths[name]
        if value is None and name in _PADDINGS:
            value = self._lengths["abc_pad_length"]
        if value is None and name in _DEFAULTS_IN_METERS:
            value = _DEFAULTS_IN_METERS[name]
            if self.unit == "km":
                value /= 1000.0
        return value

    # Mesh size
    def edge_length_along(self, axis: str) -> float | None:
        """Target edge length along one axis.

        Parameters
        ----------
        axis : str
            'z', 'x' or 'y'.

        Returns
        -------
        float or None
            The direction-specific edge length if set, otherwise
            ``edge_length``.

        Raises
        ------
        ValueError
            If ``axis`` is not 'z', 'x' or 'y'.
        """
        validate_parameter("axis", axis, ["z", "x", "y"])
        value = self._edge_lengths[f"edge_length_{axis}"]
        return self._edge_lengths["edge_length"] if value is None else value

    @property
    def cells_per_wavelength(self) -> float | None:
        """Number of cells per wavelength; setting it clears the edge lengths."""
        return self._cells_per_wavelength

    @cells_per_wavelength.setter
    def cells_per_wavelength(self, value: float | None) -> None:
        validate_numeric("cells_per_wavelength", value, accept_parameter_as_none=True, lower_bound=0)
        if value is not None and any(v is not None for v in self._edge_lengths.values()):
            warn("Setting cells_per_wavelength removes edge_length parameters")
            self._edge_lengths = dict.fromkeys(_EDGE_LENGTHS)
        self._cells_per_wavelength = value

    @property
    def source_frequency(self) -> float | None:
        """Source frequency in Hz; warns outside [1.5, 50] Hz."""
        return self._source_frequency

    @source_frequency.setter
    def source_frequency(self, value: float | None) -> None:
        validate_numeric("source_frequency", value, accept_parameter_as_none=True, lower_bound=0)
        if value is not None and value < 1.5:
            warn(f"Source frequency of {value} too low for realistic FWI case")
        elif value is not None and value > 50:
            warn(f"Source frequency of {value} too high for realistic FWI case, please low-pass filter")
        self._source_frequency = value

    # Mesh source
    @property
    def mesh_type(self) -> str | None:
        """Mesh generation type, one of ``MESH_TYPES``."""
        return self._mesh_type

    @mesh_type.setter
    def mesh_type(self, value: str | None) -> None:
        self._mesh_type = validate_parameter("mesh_type", value, [*MESH_TYPES, None])
        self._validate_if_not_batch()

    @property
    def automatic_mesh(self) -> bool:
        """Whether ``mesh_type`` is one of ``AUTOMATIC_MESH_TYPES``."""
        return self.mesh_type in AUTOMATIC_MESH_TYPES

    @property
    def periodic(self) -> bool:
        """Periodic boundary condition flag; only supported for 'firedrake_mesh'."""
        return self._periodic

    @periodic.setter
    def periodic(self, value: bool) -> None:
        self._periodic = validate_boolean("periodic", value)
        self._validate_if_not_batch()

    @property
    def mesh_file(self) -> str | None:
        """Path to an existing .msh mesh file."""
        return self._mesh_file

    @mesh_file.setter
    def mesh_file(self, value: str | os.PathLike | None) -> None:
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        self._mesh_file = validate_file("mesh_file", value, [".msh"], accept_parameter_as_none=True,
                                        check_file_existence=True)

    @property
    def user_mesh(self) -> Any:
        """User-provided mesh; setting one also sets ``mesh_type`` to 'user_mesh'."""
        return self._user_mesh

    @user_mesh.setter
    def user_mesh(self, value: Any) -> None:
        if value is not None:
            self.mesh_type = "user_mesh"
        self._user_mesh = value

    @property
    def output_filename(self) -> str | None:
        """Output file for a generated mesh (.msh, or .vtk for visualization only)."""
        return self._output_filename

    @output_filename.setter
    def output_filename(self, value: str | os.PathLike | None) -> None:
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        validate_file("output_filename", value, [".msh", ".vtk"], accept_parameter_as_none=True)
        if value is not None and value.lower().endswith(".vtk"):
            warn("VTK meshes for visualization only, will not run a simulation.")
        self._output_filename = value

    # Velocity inputs
    @property
    def segy_velocity_model(self) -> str | None:
        """Deprecated SEGY path; setting one warns and also sets ``velocity_model``."""
        return self._segy_velocity_model

    @segy_velocity_model.setter
    def segy_velocity_model(self, value: str | None) -> None:
        if value is not None:
            warn("Passing SEGY directly to the mesher is deprecated. Please use grid point velocity inputs.")
            self.velocity_model = value
        self._segy_velocity_model = value

    @property
    def grid_velocity_data(self) -> dict | None:
        """Gridded velocity data with 'vp_values' and grid spacing keys."""
        return self._grid_velocity_data

    @grid_velocity_data.setter
    def grid_velocity_data(self, value: dict | None) -> None:
        validate_data_structure("grid_velocity_data", value, "dict", accept_parameter_as_none=True)
        if value is not None:
            if "vp_values" not in value:
                raise ValueError("Grid velocity data needs vp_values key.")
            spacing_keys = ["grid_spacing_z", "grid_spacing_x"]
            if value.get("length_y") not in (None, 0.0):
                spacing_keys.append("grid_spacing_y")
            if "grid_spacing" not in value and not all(key in value for key in spacing_keys):
                raise ValueError(
                    "Grid velocity data needs either grid_spacing or directional "
                    "grid_spacing_z/grid_spacing_x keys."
                )
        self._grid_velocity_data = value
