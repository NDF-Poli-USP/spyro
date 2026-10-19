"""Options that only affect Gmsh-based mesh generation."""

from dataclasses import dataclass, fields

import numpy as np

from ...utils.error_management import (
    validate_boolean,
    validate_numeric,
    validate_parameter,
)


@dataclass
class GmshOptions:
    """Options for Gmsh-based mesh generation (`mesh_type='gmsh_mesh'`).

    Padding sizes, `h_padding` and `min_element_size` are lengths whose
    defaults depend on the domain unit, so they stay on
    :class:`~spyro.meshing.MeshingParameters`.

    Attributes
    ----------
    padding_type : str or None
        Domain padding: 'rectangular', 'hyperelliptical' or None.
    hyper_n : float
        Hyperellipse exponent for hyperelliptical padding.
    hmin_segy : float
        Minimum element size for SEGY interpolation.
    grade : float
        Sizing function grading for smooth element transition.
    water_interface : bool
        If True, detects and implements the water interface curve.
    water_search_value : float
        Water speed used to find the bathymetry.
    vp_water : float or None
        Substitute water speed where the model is zero.
    structured_mesh : bool
        If True, uses a structured quadrilateral mesh.
    unstructured_quad_mesh : bool
        If True, uses a 2D unstructured full-quadrilateral mesh.
    winslow_implementation : str or None
        Winslow smoothing version: 'default', 'fast' or 'numba'.
    apply_winslow : bool
        Whether to apply Winslow smoothing.
    winslow_iterations : int
        Number of Winslow smoothing iterations.
    winslow_omega : float
        Winslow node movement factor, in (0, 1].
    extend_segy : bool
        Whether to extend the SEGY sizing function into the padding.
    segy_nz, segy_nx, segy_ny : int or None
        Samples of a 3D binary velocity model along each axis.
    segy_dz, segy_dx, segy_dy : float or None
        Sample spacing of a 3D binary velocity model along each axis.
    segy_byte_order : str
        Byte order of a 3D binary velocity model, 'big' or 'little'.
    segy_axes_order : tuple of int
        Axis permutation of a 3D binary velocity model.
    segy_axes_order_sort : str
        Memory order of a 3D binary velocity model, 'C' or 'F'.
    segy_dtype : str
        Numpy dtype of a 3D binary velocity model.
    gmsh_parallel : bool
        Whether to compute the 3D sizing callback in parallel.
    gmsh_num_threads : int
        Number of threads used by parallel 3D meshing.
    """

    padding_type: str | None = None
    hyper_n: float = 3.0
    hmin_segy: float = 0.0
    grade: float = 0.9
    water_interface: bool = False
    water_search_value: float = 0.0
    vp_water: float | None = None
    structured_mesh: bool = False
    unstructured_quad_mesh: bool = False
    winslow_implementation: str | None = "numba"
    apply_winslow: bool = True
    winslow_iterations: int = 5000
    winslow_omega: float = 0.5
    extend_segy: bool = True
    segy_nz: int | None = None
    segy_nx: int | None = None
    segy_ny: int | None = None
    segy_dz: float | None = None
    segy_dx: float | None = None
    segy_dy: float | None = None
    segy_byte_order: str = "big"
    segy_axes_order: tuple[int, int, int] = (0, 1, 2)
    segy_axes_order_sort: str = "F"
    segy_dtype: str = "float32"
    gmsh_parallel: bool = False
    gmsh_num_threads: int = 1

    @classmethod
    def keys(cls) -> tuple[str, ...]:
        """Names of all options, as accepted in mesh input dictionaries.

        Returns
        -------
        tuple of str
            The field names.
        """
        return tuple(field.name for field in fields(cls))

    @classmethod
    def from_dictionary(cls, dictionary: dict) -> "GmshOptions":
        """Build options from a mesh input dictionary.

        Keys that are not options are ignored, and None values mean "use the
        default".

        Parameters
        ----------
        dictionary : dict
            Mesh input dictionary.

        Returns
        -------
        GmshOptions
            The options. They are not validated yet.
        """
        return cls(
            **{
                key: dictionary[key]
                for key in cls.keys()
                if dictionary.get(key) is not None
            }
        )

    def validate(self, dimension: int | None = None) -> None:
        """Check every option and the rules between them.

        Parameters
        ----------
        dimension : int, optional
            Spatial dimension of the mesh, used to reject 3D unstructured
            quadrilateral meshes. Default is None.

        Raises
        ------
        ValueError
            If an option is out of range or options are inconsistent.
        TypeError
            If an option has the wrong type.
        """
        for name in (
            "water_interface",
            "structured_mesh",
            "unstructured_quad_mesh",
            "apply_winslow",
            "extend_segy",
            "gmsh_parallel",
        ):
            validate_boolean(name, getattr(self, name))
        validate_parameter(
            "padding_type", self.padding_type, [None, "rectangular", "hyperelliptical"]
        )
        validate_parameter(
            "winslow_implementation",
            self.winslow_implementation,
            [None, "default", "fast", "numba"],
        )
        validate_numeric(
            "winslow_iterations",
            self.winslow_iterations,
            float_num=False,
            lower_bound=0,
            include_lower_bound=True,
        )
        validate_numeric(
            "winslow_omega",
            self.winslow_omega,
            lower_bound=0,
            upper_bound=1,
            include_upper_bound=True,
        )
        validate_numeric("grade", self.grade, lower_bound=0)
        validate_numeric("hyper_n", self.hyper_n, lower_bound=0)
        validate_numeric(
            "hmin_segy", self.hmin_segy, lower_bound=0, include_lower_bound=True
        )
        validate_numeric(
            "gmsh_num_threads",
            self.gmsh_num_threads,
            float_num=False,
            lower_bound=1,
            include_lower_bound=True,
        )
        validate_parameter("segy_byte_order", self.segy_byte_order, ["big", "little"])
        validate_parameter(
            "segy_axes_order_sort", self.segy_axes_order_sort, ["C", "F"]
        )
        if sorted(self.segy_axes_order) != [0, 1, 2]:
            raise ValueError(
                f"segy_axes_order must be a permutation of (0, 1, 2),"
                f"got {self.segy_axes_order!r}"
            )
        try:
            np.dtype(self.segy_dtype)
        except TypeError as error:
            raise ValueError(
                f"segy_dtype {self.segy_dtype!r} is not a valid numpy dtype"
            ) from error

        if self.structured_mesh and self.unstructured_quad_mesh:
            raise ValueError(
                "'structured_mesh' and 'unstructured_quad_mesh' are mutually exclusive."
            )
        if self.unstructured_quad_mesh and dimension == 3:
            raise ValueError(
                "'unstructured_quad_mesh' is currently supported only"
                "for 2D Gmsh meshes."
            )
