"""Error management utilities.

This file contains methods for handling errors in Spyro, either to send
messages to the user or to prevent numerical instability in objects.
"""

from firedrake import Function, FunctionSpace, Mesh
from firedrake.functionspaceimpl import WithGeometry
from numpy import float32, float64, inf, int32, int64, isinf, isnan, ndarray
from os import path
from ufl.form import Form
from ufl.geometry import SpatialCoordinate


def validate_parameter(parameter_name, parameter_value, valid_values):
    """Validate parameter value and raise a ValueError with a specific error message.

    Parameters
    ----------
    parameter_name : `str`
        Name of the parameter to be validated (used in error messages).
    parameter_value : `str`, `int` or `float`
        Value of the parameter to be validated.
    valid_values : `list`
        List of valid values for the parameter.

    Returns
    -------
    parameter_value : `str`, `int` or `float`
        The validated parameter value.

    Raises
    ------
    ValueError
        If the parameter value is not in the list of valid values.
    """
    # Error message about the invalid parameter
    if parameter_value not in valid_values:
        raise ValueError(
            f"Invalid {parameter_name}: '{parameter_value}'. "
            f"Please use {_join_options(valid_values)}."
        )

    return parameter_value


def mutually_exclusive_parameter_error(parameter_name_lst, parameter_value_lst):
    """Raise a ValueError with specific message for mutually exclusive parameters.

    Parameters
    ----------
    parameter_name_lst : `list` of `str`
        List of names of the parameters that are mutually exclusive.
    parameter_value_lst : `list`
        List of values of the parameters that are mutually exclusive.

    Raises
    ------
    ValueError
        If two or more parameters have been provided by the user.
        That is, value of the parameters is not `None`.
    """
    parameter_defined = [
        parameter
        for parameter, value in zip(parameter_name_lst, parameter_value_lst)
        if value is not None
    ]

    # Only raise if two or more parameters are defined
    if len(parameter_defined) > 1:
        # Error message about the invalid parameter
        exc_str = (
            f"Parameters {_join_options(parameter_defined, conjunction='and')} "
            "are mutually exclusive.\n"
        )
        err_str = (
            "Please specify only one of these parameters: "
            f"{_join_options(parameter_name_lst)}."
        )
        raise ValueError(exc_str + err_str)


def validate_model_dimension(parameter_names, parameters, expected_dimension):
    """Raise a ValueError if parameter dimensions mismatch expected model dimension.

    Parameters
    ----------
    parameter_names : `tuple`
        Names of the parameters to check dimensions.
    parameters : `tuple`
        Parameters to check dimensions.
    expected_dimension : `int`
        Expected dimension of the parameters (2 or 3).

    Raises
    ------
    ValueError
        If the dimensions of the parameters do not match the expected dimension.
    """
    reference_name, comparison_name = parameter_names
    parameter_reference, parameter_comparison = parameters

    chk_reference = len(parameter_reference)
    chk_comparison = len(parameter_comparison)
    if expected_dimension != chk_reference or expected_dimension != chk_comparison:
        dim_err = (
            f"Mismatch in domain dimensions\n"
            f"{reference_name} ({chk_reference}), "
            f"{comparison_name} ({chk_comparison}) "
            f"do not match expected model dimension ({expected_dimension}D)."
        )
        raise ValueError(dim_err)


def sanitize_num_array(
    data_arr, nan_values=True, inf_values=True, negative_values=True
):
    """Set NaNs, infinities, and/or negative values to zero in an array.

    This function modifies data_arr in place.

    Parameters
    ----------
    data_arr : `array`
        An array with possible NaN or negative components.
    nan_values : `bool`, optional
        If `True`, replace NaN values with zero. Default is `True`.
    inf_values : `bool`, optional
        If `True`, replace infinite values (both +inf and -inf) with zero.
        Default is `True`.
    negative_values : `bool`, optional
        If `True`, replace negative values with zero. Default is `True`.

    Returns
    -------
    data_arr : `array`
        An array with null or positive components and invalid values replaced by zero.

    Raises
    ------
    TypeError
        If data_arr is not a numpy array or contains elements that are not
        numeric (float or int).
    """
    # Validate input type
    validate_data_structure(
        "data_arr", data_arr, "array", expected_type_element=("float", "int")
    )

    # Build condition mask
    condition = False
    if nan_values:
        condition = condition | isnan(data_arr)

    if inf_values:
        condition = condition | isinf(data_arr)

    if negative_values:
        condition = condition | (data_arr < 0.0)

    # Apply cleaning
    data_arr[condition] = 0.0

    return data_arr


def validate_numeric(
    parameter_name,
    parameter_value,
    float_num=True,
    integer_num=True,
    accept_parameter_as_none=False,
    lower_bound=None,
    upper_bound=None,
    include_lower_bound=False,
    include_upper_bound=False,
):
    """Validate numerical parameters and raise a ValueError if invalid.

    Parameters
    ----------
    parameter_name : `str`
        Name of the parameter to be validated (used in error messages).
    parameter_value : `int` or `float`
        Value of the parameter to be validated.
    float_num : `bool`, optional
        If `True`, the parameter can be a float. Default is `True`.
    integer_num : `bool`, optional
        If `True`, the parameter can be an integer. Default is `False`.
    accept_parameter_as_none : `bool`, optional
        If `True`, the parameter value is allowed to be validated as `None`.
        Default is `False`, in which case `None` is not allowed.
    lower_bound : `int` or `float`, optional
        Lower bound for the parameter value. Default is `None` (no lower bound).
    upper_bound : `int` or `float`, optional
        Upper bound for the parameter value. Default is `None` (no upper bound).
    include_lower_bound : `bool`, optional
        If `True`, the lower bound is included in the valid range. Default is `False`.
    include_upper_bound : `bool`, optional
        If `True`, the upper bound is included in the valid range. Default is `False`.

    Returns
    -------
    parameter_value : `int` or `float`
        The validated parameter value.

    Raises
    ------
    TypeError
        If the parameter value is not of the expected type (`float` or `int`).
    ValueError
        If the parameter value is outside the specified bounds or the bounds are
        invalid.
    """
    if parameter_value is None and accept_parameter_as_none:
        return parameter_value

    # Not int or float
    if not isinstance(parameter_value, (int, float)):
        if float_num and integer_num:
            str_type = "float or an integer"
        elif float_num:
            str_type = "float"
        elif integer_num:
            str_type = "integer"

        raise TypeError(
            f"'{parameter_name}' must be a {str_type}, "
            f"got {type(parameter_value).__name__}."
        )

    # Check if float is allowed when value is integer
    if isinstance(parameter_value, float) and (integer_num and not float_num):
        raise TypeError(
            f"'{parameter_name}' must be an integer, "
            f"got {type(parameter_value).__name__}."
        )

    # Check if integer is allowed when value is integer
    if isinstance(parameter_value, int) and (not integer_num and float_num):
        raise TypeError(
            f"'{parameter_name}' must be a float, "
            f"got {type(parameter_value).__name__}."
        )

    # Set default bounds
    upper_bound = inf if upper_bound is None else upper_bound
    lower_bound = -inf if lower_bound is None else lower_bound

    # Validate bounds
    if upper_bound <= lower_bound:
        raise ValueError(
            f"Invalid bounds: upper_bound ({upper_bound}) must "
            f"be greater than lower_bound ({lower_bound})."
        )

    lower_invalid = (
        parameter_value < lower_bound
        if include_lower_bound
        else parameter_value <= lower_bound
    )
    upper_invalid = (
        parameter_value > upper_bound
        if include_upper_bound
        else parameter_value >= upper_bound
    )

    # Check if value is within bounds
    if lower_invalid or upper_invalid:
        # Build error message based on which bounds are finite
        if lower_bound > -inf and upper_bound < inf:  # Both bounds are finite
            bound_str = f"between {lower_bound} and {upper_bound}"
            if include_lower_bound and include_upper_bound:
                bound_str += " (both bounds inclusive)"
            elif include_lower_bound:
                bound_str += " (lower bound inclusive)"
            elif include_upper_bound:
                bound_str += " (upper bound inclusive)"
            else:
                bound_str += " (both bounds exclusive)"
        elif lower_bound > -inf:  # Only lower bound is finite
            bound_str = (
                f"greater than or equal to {lower_bound}"
                if include_lower_bound
                else f"greater than {lower_bound}"
            )
        elif upper_bound < inf:  # Only upper bound is finite
            bound_str = (
                f"less than or equal to {upper_bound}"
                if include_upper_bound
                else f"less than {upper_bound}"
            )

        raise ValueError(
            f"'{parameter_name}' must be {bound_str}, got {parameter_value}."
        )

    return parameter_value


def validate_enum(parameter_name, parameter_value, valid_enum):
    """Validate and convert an enum parameter, returning the enum instance.

    This method validates that the provided parameter value is either an
    instance of the specified enum class or a string that maps to a valid
    enum value. If valid, it returns the corresponding enum instance.
    Otherwise, it raises an appropriate exception.

    Parameters
    ----------
    parameter_name : `str`
        Name of the parameter to be validated (used in error messages).
    parameter_value : `object`
        Value of the parameter to be validated. Can be an `enum.EnumMeta` or a `str`.
    valid_enum : `enum.EnumMeta`
        Enum class containing the valid values for the parameter.

    Returns
    -------
    `enum.Enum`
        The validated enum instance corresponding to the input value.

    Raises
    ------
    TypeError
        If the parameter value is neither an instance of the valid enum class nor a
        `str`.
    ValueError
        If the parameter value is a `str` that does not match any valid enum value.
    """
    # Check if already a valid enum instance
    if isinstance(parameter_value, valid_enum):
        return parameter_value

    # Check if string maps to valid enum value
    if isinstance(parameter_value, str):
        valid_values = [enum.value for enum in valid_enum]
        validate_parameter(parameter_name, parameter_value, valid_values)
        return valid_enum(parameter_value)

    # Invalid type - neither enum instance nor string
    raise TypeError(
        f"'{parameter_name}' must be {valid_enum.__name__} or str"
        f", got {type(parameter_value).__name__}"
    )


def validate_string(parameter_name, parameter_value, accept_parameter_as_none=False):
    """Validate string parameters and raise a TypeError if invalid.

    Parameters
    ----------
    parameter_name : `str`
        Name of the parameter to be validated (used in error messages).
    parameter_value : `object`
        Value of the parameter to be validated.
    accept_parameter_as_none : `bool`, optional
        If `True`, the parameter value is allowed to be validated as `None`.
        Default is `False`, in which case `None` is not allowed.

    Returns
    -------
    parameter_value : `str`
        The validated parameter value.

    Raises
    ------
    TypeError
        If the parameter value is not of the expected type (`str`).
    """
    if parameter_value is None and accept_parameter_as_none:
        return parameter_value

    # Checking the parameter type
    if not isinstance(parameter_value, str):
        value_type = type(parameter_value).__name__
        raise TypeError(f"'{parameter_name}' must be a string, got {value_type}.")

    return parameter_value


def validate_data_structure(
    parameter_name,
    parameter_value,
    expected_type,
    expected_type_element=None,
    expected_length=None,
    expected_shape=None,
    accept_parameter_as_none=False,
):
    """Validate data structure parameters and raise a TypeError if invalid.

    Parameters
    ----------
    parameter_name : `str`
        Name of the parameter to be validated (used in error messages).
    parameter_value : `object`
        Value of the parameter to be validated.
    expected_type : `str`
        Expected type of the data structure parameter as a `str`. The validation
        supports the types `dict`, `list`, `tuple`, or `ndarray` (NumPy arrays).
    expected_type_element : `tuple`, optional
        Expected type of the data structure elements passed as a `str`. The validation
        supports the types `float`, `int`, `str` or `NoneType`. Exs: ("float", "int")
        for a NumPy array or ("float", "int", "str", "NoneType") for a mixed elements.
        Default is `None`, in which case the elements are not checked.
    expected_length : `int`, optional
        Expected length of the data structure parameter. Default is `None`,
        in which case the length is not checked.
    expected_shape : `tuple`, optional
        Expected shape of the data structure parameter (only for "array2D" or
        "array3D").
        The expected shape should be provided as a tuple of integers. If a length of the
        shape dimension is not known, it can be set to `None`. For example, an expected
        shape of (3, None) means that the first dimension should have length 3, while
        the second dimension can have any length greater than zero. Default is `None`,
        in which case the shape is not checked.
    accept_parameter_as_none : `bool`, optional
        If `True`, the parameter value is allowed to be validated as `None`.
        Default is `False`, in which case `None` is not allowed.

    Returns
    -------
    parameter_value : `dict`, `list`, `tuple`, or `ndarray`
        The validated parameter value.

    Raises
    ------
    TypeError
        If the parameter value is not of the expected type given by 'expected_type' or
        the elements are not of the expected type given by 'expected_type_element'.
    ValueError
        If the parameter value does not have the expected length (if provided).
    """
    if parameter_value is None and accept_parameter_as_none:
        return parameter_value

    parameter_map = {
        "dict": dict,
        "list": list,
        "tuple": tuple,
        "array": ndarray,
        "array2D": ndarray,
        "array3D": ndarray,
    }

    validate_parameter("expected_type", expected_type, parameter_map.keys())

    element_map = {
        "int": int,
        "float": float,
        "list": list,
        "NoneType": type(None),
        "str": str,
        "tuple": tuple,
    }

    # Checking the parameter type
    if not isinstance(parameter_value, parameter_map[expected_type]):
        raise TypeError(
            f"'{parameter_name}' must be a {expected_type}, "
            f"got {type(parameter_value).__name__}."
        )

    if expected_type in ["array2D", "array3D"]:
        expected_dimension = 2 if expected_type == "array2D" else 3
        if parameter_value.ndim != expected_dimension:
            raise ValueError(
                f"'{parameter_name}' must be a {expected_dimension}D array, "
                f"got {parameter_value.ndim}D array."
            )

        if expected_shape is not None:
            parameter_shape = parameter_value.shape
            for dim in range(expected_dimension):
                if (
                    expected_shape[dim] is not None
                    and parameter_shape[dim] != expected_shape[dim]
                ):
                    raise ValueError(
                        f"'{parameter_name}' must have shape {expected_shape}, "
                        f"got shape {parameter_shape}."
                    )

    # Check if the parameter has the expected length
    if expected_length is not None and len(parameter_value) != expected_length:
        raise ValueError(
            f"'{parameter_name}' must have length {expected_length}, "
            f"got length {len(parameter_value)}."
        )

    # Check if all elements are of expected type
    if expected_type_element is not None:
        if isinstance(expected_type_element, str):
            expected_type_element = (expected_type_element,)
        for etype in expected_type_element:
            validate_parameter("expected_type_element", etype, element_map.keys())
        expected_types = tuple(element_map[etype] for etype in expected_type_element)
        expected_types += (
            (
                int32,
                int64,
            )
            if "int" in expected_type_element
            else ()
        )
        expected_types += (
            (
                float32,
                float64,
            )
            if "float" in expected_type_element
            else ()
        )
        parameter_value_check = (
            parameter_value
            if expected_type not in ["array2D", "array3D"]
            else parameter_value.flatten()
        )
        if not all(isinstance(item, expected_types) for item in parameter_value_check):
            opt_str = _join_options(expected_type_element)
            raise TypeError(
                f"All elements of '{parameter_name}' must be of type: {opt_str}."
            )

    return parameter_value


def validate_firedrake_parameter(
    parameter_name, parameter_value, expected_type, accept_parameter_as_none=False
):
    """Validate Firedrake parameters and raise a TypeError if invalid.

    Parameters
    ----------
    parameter_name : `str`
        Name of the parameter to be validated (used in error messages).
    parameter_value : `object`
        Value of the parameter to be validated.
    expected_type : `str`
        Expected type of the Firedrake parameter as a `str`. The validation
        supports the types `Function`, `FunctionSpace`, or `Mesh`.
    accept_parameter_as_none : `bool`, optional
        If `True`, the parameter value is allowed to be validated as `None`.
        Default is `False`, in which case `None` is not allowed.

    Returns
    -------
    parameter_value : `firedrake.Function`, `firedrake.FunctionSpace`, or
        `firedrake.Mesh`
        The validated parameter value.

    Raises
    ------
    TypeError
        If the parameter value is not of the expected type given by 'expected_type'.
    """
    if parameter_value is None and accept_parameter_as_none:
        return parameter_value

    parameter_map = {
        "Form": Form,
        "Function": Function,
        "FunctionSpace": type(FunctionSpace),
        "Mesh": type(Mesh),
        "SpatialCoordinate": SpatialCoordinate,
    }

    validate_parameter("expected_type", expected_type, parameter_map.keys())

    # Checking the parameter type
    expected_valid = (parameter_map[expected_type],)
    expected_valid += (WithGeometry,) if "FunctionSpace" in expected_type else ()
    if not isinstance(parameter_value, expected_valid):
        opt_str = _join_options(cls.__name__ for cls in expected_valid)
        raise TypeError(
            f"'{parameter_name}' must be of type: {opt_str}, "
            f"got {type(parameter_value).__name__}."
        )

    return parameter_value


def validate_file(
    parameter_name,
    parameter_value,
    valid_extensions,
    accept_parameter_as_none=False,
    check_file_existance=False,
):
    """Validate a file parameter and raise a ValueError if invalid.

    Parameters
    ----------
    parameter_name : `str`
        Name of the parameter to be validated (used in error messages).
    parameter_value : `str`
        Value of the parameter to be validated.
    valid_extensions : `list`
        List of valid file extensions for the parameter.
    accept_parameter_as_none : `bool`, optional
        If `True`, the parameter value is allowed to be validated as `None`.
        Default is `False`, in which case `None` is not allowed.
    check_file_existance : `bool`, optional
        If `True`, we will check if the file exists. Default is `False`

    Returns
    -------
    parameter_value : `str`
        The validated file path or file name.

    Raises
    ------
    TypeError
        If the parameter value is not a `str`.
    ValueError
        If the file extension is not in the list of valid extensions.
    FileNotFoundError
        If the file does not exist and check_exists is True
    """
    parameter_value = validate_string(
        parameter_name,
        parameter_value,
        accept_parameter_as_none=accept_parameter_as_none,
    )

    if parameter_value is None and accept_parameter_as_none:
        return parameter_value

    if not path.exists(parameter_value) and check_file_existance:
        raise FileNotFoundError(
            f"'{parameter_name}' '{parameter_value}' does not exist"
        )

    file_extension = path.splitext(parameter_value)[1].lower()
    validate_parameter("extension_type", file_extension, valid_extensions)

    return parameter_value


def _join_options(values, conjunction="or"):
    """Join values into a human-readable English list.

    Examples
    --------
    >>> _join_options(["a"])
    "'a'"

    >>> _join_options(["a", "b"])
    "'a' or 'b'"

    >>> _join_options(["a", "b"], conjunction="and")
    "'a' and 'b'"

    >>> _join_options(["a", "b", "c"])
    "'a', 'b' or 'c'"
    """
    values = tuple(f"'{value}'" for value in values)

    if not values:
        return ""

    if len(values) == 1:
        return values[0]

    return f"{', '.join(values[:-1])} {conjunction} {values[-1]}"
