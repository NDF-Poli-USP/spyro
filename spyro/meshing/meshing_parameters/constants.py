"""Constants shared by the meshing parameters classes."""

CELLS_PER_WAVELENGTH = {
    ("mass_lumped_triangle", 2, 2): 7.02,
    ("mass_lumped_triangle", 3, 2): 3.70,
    ("mass_lumped_triangle", 4, 2): 2.67,
    ("mass_lumped_triangle", 5, 2): 2.03,
    ("mass_lumped_triangle", 2, 3): 6.12,
    ("mass_lumped_triangle", 3, 3): 3.72,
}

METHODS = (
    "mass_lumped_triangle",
    "DG_triangle",
    "spectral_quadrilateral",
    "DG_quadrilateral",
    "CG",
)
QUADRILATERAL_METHODS = frozenset({"spectral_quadrilateral", "DG_quadrilateral"})
TRIANGLE_METHODS = frozenset({"mass_lumped_triangle", "DG_triangle"})
UNIT_ALIASES = {"m": "meters", "meters": "meters", "km": "km"}

DOMAIN_LENGTHS = ("length_z", "length_x", "length_y")
LENGTHS = DOMAIN_LENGTHS + (
    "abc_pad_length",
    "padding_x",
    "padding_y",
    "padding_z",
    "h_padding",
    "min_element_size",
)
# Lengths that fall back to ``abc_pad_length`` when unset.
PADDINGS = frozenset({"padding_x", "padding_y", "padding_z"})
# Lengths whose default is given in meters and converted to the domain unit.
DEFAULTS_IN_METERS = {"h_padding": 500.0, "min_element_size": 35.0}
EDGE_LENGTHS = ("edge_length", "edge_length_z", "edge_length_x", "edge_length_y")


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
    return CELLS_PER_WAVELENGTH.get((method, degree, dimension))
