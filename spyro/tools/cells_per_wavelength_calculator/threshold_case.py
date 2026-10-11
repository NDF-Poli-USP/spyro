"""Discretization whose receiver error sets the cells-per-wavelength threshold."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdCase:
    """Discretization whose receiver error sets the accepted error threshold.

    Attributes
    ----------
    method : str
        Either ``"mass_lumped_triangle"`` or ``"spectral_quadrilateral"``.
    degree : int
        Spatial polynomial degree.
    cpw : float
        Cells-per-wavelength of the mesh.
    """

    method: str = "spectral_quadrilateral"
    degree: int = 4
    cpw: float = 2.0


def _parse_threshold_case(
    value: "bool | dict | ThresholdCase | None",
) -> ThresholdCase | None:
    """Convert the ``"threshold_case"`` parameter to a threshold case.

    Parameters
    ----------
    value : bool, dict, ThresholdCase or None
        None or False for no threshold case, True for the default
        :class:`ThresholdCase`, or a dict of :class:`ThresholdCase` fields
        overriding the defaults.

    Returns
    -------
    ThresholdCase or None
        The threshold case, or None if the threshold is given directly.

    Raises
    ------
    TypeError
        If ``value`` is of an unsupported type, or a dict has unknown keys.
    """
    if value is None or value is False:
        return None
    if value is True:
        return ThresholdCase()
    if isinstance(value, ThresholdCase):
        return value
    if isinstance(value, dict):
        return ThresholdCase(**value)
    raise TypeError(
        "threshold_case must be a bool, a dict or a ThresholdCase, "
        f"got {type(value).__name__}."
    )
