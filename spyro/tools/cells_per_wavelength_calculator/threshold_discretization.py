"""Discretization whose receiver error sets the cells-per-wavelength threshold."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdDiscretization:
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


def _parse_threshold_discretization(
    value: "bool | dict | ThresholdDiscretization | None",
) -> ThresholdDiscretization | None:
    """Convert the ``"threshold_discretization"`` parameter.

    Parameters
    ----------
    value : bool, dict, ThresholdDiscretization or None
        None or False for no threshold discretization, True for the default
        :class:`ThresholdDiscretization`, or a dict of
        :class:`ThresholdDiscretization` fields overriding the defaults.

    Returns
    -------
    ThresholdDiscretization or None
        The threshold discretization, or None if the threshold is given
        directly.

    Raises
    ------
    TypeError
        If ``value`` is of an unsupported type, or a dict has unknown keys.
    """
    if value is None or value is False:
        return None
    if value is True:
        return ThresholdDiscretization()
    if isinstance(value, ThresholdDiscretization):
        return value
    if isinstance(value, dict):
        return ThresholdDiscretization(**value)
    raise TypeError(
        "threshold_discretization must be a bool, a dict or a "
        f"ThresholdDiscretization, got {type(value).__name__}."
    )
