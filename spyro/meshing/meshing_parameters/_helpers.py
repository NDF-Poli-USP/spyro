"""Helper functions for reading meshing parameters."""

from typing import Any
from warnings import warn


def guess_unit(value: float | None) -> str | None:
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


def get_parameter(
    dictionary: dict, key: str, default: Any = None, keyword_value: Any = None
) -> Any:
    """Read a parameter from a dictionary, with an optional keyword override.

    A None entry in the dictionary is treated as missing.

    Parameters
    ----------
    dictionary : dict
        Input dictionary.
    key : str
        Key of the entry in `dictionary`.
    default : Any, optional
        Value used when neither source provides one. Default is None.
    keyword_value : Any, optional
        Value given as an explicit keyword argument, or None if it was not
        given. Default is None.

    Returns
    -------
    Any
        The keyword value if it is not None, otherwise the dictionary value
        if it is not None, otherwise `default`.

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
                f"and in input_mesh_dictionary ({dict_value!r}). "
                "Using the keyword argument."
            )
    return keyword_value
