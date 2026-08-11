"""Error management utilities.

This file contains methods for handling errors in Spyro, either to send
messages to the user or to prevent numerical instability in objects."""

from firedrake import Function, FunctionSpace, Mesh
from firedrake.functionspaceimpl import WithGeometry
from numpy import float32, float64, inf, int32, int64, isinf, isnan, ndarray
from os import path
from ufl.form import Form
from ufl.geometry import SpatialCoordinate


def value_parameter_error(par_name, par_value, valid_values):
    """Validate parameter value and raise a ValueError with a specific error message.

    Parameters
    ----------
    par_name : `str`
        Name of the parameter to be validated (used in error messages).
    par_value : `str`, `int` or `float`
        Value of the parameter to be validated.
    valid_values : `list`
        List of valid values for the parameter.

    Returns
    -------
    par_value : `str`, `int` or `float`
        The validated parameter value.

    Raises
    ------
    ValueError
        If the parameter value is not in the list of valid values.
    """

    # Error message about the invalid parameter
    if par_value not in valid_values:
        err_str = f"Invalid {par_name}: '{par_value}'. Please use: "
        opt_str = ", ".join([f"'{val}'" for val in valid_values])
        last_comma = opt_str.rfind(',')
        opt_str = opt_str[:last_comma] + " or" + opt_str[last_comma + 1:] \
            if len(valid_values) > 1 else opt_str

        raise ValueError(err_str + opt_str)

    return par_value


def mutually_exclusive_parameter_error(par_name_lst, par_value_lst):
    """Raise a ValueError with specific message for mutually exclusive parameters.

    Parameters
    ----------
    par_name_Lst : `list` of `str`
        List of names of the parameters that are mutually exclusive.
    par_value_lst : `list`
        List of values of the parameters that are mutually exclusive.

    Raises
    ------
    ValueError
        If two or more parameters have been provided by the user.
        That is, value of the parameters is not `None`.
    """

    par_defined = [par for par, val in zip(par_name_lst,
                                           par_value_lst)
                   if val is not None]

    # Only raise if two or more parameters are defined
    if len(par_defined) > 1:
        # Error message about the invalid parameter
        exc_str = "Parameters " + ", ".join([f"'{name}'" for name in par_defined])
        last_comma = exc_str.rfind(',')
        exc_str = exc_str[:last_comma] + ' and' + exc_str[last_comma + 1:]
        exc_str += " mutually exclusive.\n"
        err_str = "Please specify only one of these parameters: "
        opt_str = ", ".join([f"'{val}'" for val in par_name_lst])
        last_comma = opt_str.rfind(',')
        opt_str = opt_str[:last_comma] + " or" + opt_str[last_comma + 1:]

        raise ValueError(exc_str + err_str + opt_str)


def value_model_dimension_error(par_names, parameters, expected_dim):
    """Raise a ValueError if parameter dimensions mismatch expected model dimension.

    Parameters
    ----------
    par_names : `tuple`
        Names of the parameters to check dimensions.
    parameters : `tuple`
        Parameters to check dimensions.
    expected_dim : `int`
        Expected dimension of the parameters (2 or 3).

    Raises
    ------
    ValueError
        If the dimensions of the parameters do not match the expected dimension.
    """

    str_reference, str_comparison = par_names
    par_reference, par_comparison = parameters

    chk_reference = len(par_reference)
    chk_comparison = len(par_comparison)
    if expected_dim != chk_reference or expected_dim != chk_comparison:
        dim_err = (f"Mismatch in domain dimensions\n"
                   f"{str_reference} ({chk_reference}), "
                   f"{str_comparison} ({chk_comparison}) "
                   f"do not match expected model dimension ({expected_dim}D).")
        raise ValueError(dim_err)


def sanitize_num_array(data_arr, nan_values=True, inf_values=True, negative_values=True):
    """Set NaNs, infinities, and/or negative values to zero in an array.

    Parameters
    ----------
    data_arr : `array`
        An array with possible with possible NaN or negative components.
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
    type_data_structure_error("data_arr", data_arr, "array",
                              expected_type_element=("float", "int"))

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


def value_numerical_error(par_name, par_value, float_num=True, integer_num=True,
                          none_default=False, lower_bound=None, upper_bound=None,
                          include_lower_bound=False, include_upper_bound=False):
    """Validate numerical parameters and raise a ValueError if invalid.

    Parameters
    ----------
    par_name : `str`
        Name of the parameter to be validated (used in error messages).
    par_value : `int` or `float`
        Value of the parameter to be validated.
    float_num : `bool`, optional
        If `True`, the parameter can be a float. Default is `True`.
    integer_num : `bool`, optional
        If `True`, the parameter can be an integer. Default is `False`.
    none_default : `bool`, optional
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
    par_value : `int` or `float`
        The validated parameter value.

    Raises
    ------
    TypeError
        If the parameter value is not of the expected type (`float` or `int`).
    ValueError
        If the parameter value is outside the specified bounds or the bounds are invalid.
    """

    if par_value is None and none_default:
        return par_value

    # Not int or float
    if not isinstance(par_value, (int, float)):
        if float_num and integer_num:
            str_type = "float or a integer"
        elif float_num:
            str_type = "float"
        elif integer_num:
            str_type = "integer"

        raise TypeError(f"'{par_name}' must be a {str_type} number, "
                        f"got {type(par_value).__name__}.")

    # Check if float is allowed when value is integer
    if isinstance(par_value, float) and (integer_num and not float_num):
        raise TypeError(f"'{par_name}' must be an integer number, "
                        f"got {type(par_value).__name__}.")

    # Check if integer is allowed when value is integer
    if isinstance(par_value, int) and (not integer_num and float_num):
        raise TypeError(f"'{par_name}' must be a float number, "
                        f"got {type(par_value).__name__}.")

    # Set default bounds
    upper_bound = inf if upper_bound is None else upper_bound
    lower_bound = -inf if lower_bound is None else lower_bound

    # Validate bounds
    if upper_bound <= lower_bound:
        raise ValueError(f"Invalid bounds: upper_bound ({upper_bound}) must "
                         f"be greater than lower_bound ({lower_bound}).")

    lower_invalid = par_value < lower_bound if include_lower_bound \
        else par_value <= lower_bound
    upper_invalid = par_value > upper_bound if include_upper_bound \
        else par_value >= upper_bound

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
            bound_str = (f"greater than or equal to {lower_bound}"
                         if include_lower_bound else f"greater than {lower_bound}")
        elif upper_bound < inf:  # Only upper bound is finite
            bound_str = (f"less than or equal to {upper_bound}"
                         if include_upper_bound else f"less than {upper_bound}")

        raise ValueError(f"'{par_name}' must be {bound_str}, got {par_value}.")

    return par_value


def enum_parameter_error(par_name, par_value, valid_enum):
    """Validate and convert an enum parameter, returning the enum instance.

    This method validates that the provided parameter value is either an
    instance of the specified enum class or a string that maps to a valid
    enum value. If valid, it returns the corresponding enum instance.
    Otherwise, it raises an appropriate exception.

    Parameters
    ----------
    par_name : `str`
        Name of the parameter to be validated (used in error messages).
    par_value : `object`
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
        If the parameter value is neither an instance of the valid enum class nor a `str`.
    ValueError
        If the parameter value is a `str` that does not match any valid enum value.
    """

    # Check if already a valid enum instance
    if isinstance(par_value, valid_enum):
        return par_value

    # Check if string maps to valid enum value
    if isinstance(par_value, str):
        valid_values = [enum.value for enum in valid_enum]
        value_parameter_error(par_name, par_value, valid_values)
        return valid_enum(par_value)

    # Invalid type - neither enum instance nor string
    raise TypeError(f"'{par_name}' must be {valid_enum.__name__} or str"
                    f", got {type(par_value).__name__}")


def value_string_error(par_name, par_value, none_default=False):
    """Validate string parameters and raise a TypeError if invalid.

    Parameters
    ----------
    par_name : `str`
        Name of the parameter to be validated (used in error messages).
    par_value : `object`
        Value of the parameter to be validated.
    none_default : `bool`, optional
        If `True`, the parameter value is allowed to be validated as `None`.
        Default is `False`, in which case `None` is not allowed.

    Returns
    -------
    par_value : `str`
        The validated parameter value.

    Raises
    ------
    TypeError
        If the parameter value is not of the expected type (`str`).
    """

    if par_value is None and none_default:
        return par_value

    # Checking the parameter type
    if not isinstance(par_value, str):
        raise TypeError(f"'{par_name}' must be a string, got {type(par_value).__name__}.")

    return par_value


def type_data_structure_error(par_name, par_value, expected_type,
                              expected_type_element=None, expected_length=None,
                              expected_shape=None, none_default=False):
    """Validate data structure parameters and raise a TypeError if invalid.

    Parameters
    ----------
    par_name : `str`
        Name of the parameter to be validated (used in error messages).
    par_value : `object`
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
        Expected shape of the data structure parameter (only for "array2D" or "array3D").
        The expecte shape should be provided as a tuple of integers. If a length of the
        shape dimension is not known, it can be set to `None`. For example, an expected
        shape of (3, None) means that the first dimension should have length 3, while the
        second dimension can have any length greater than zero. Default is `None`,
        in which case the shape is not checked.
    none_default : `bool`, optional
        If `True`, the parameter value is allowed to be validated as `None`.
        Default is `False`, in which case `None` is not allowed.

    Returns
    -------
    par_value : `dict`, `list`, `tuple`, or `ndarray`
        The validated parameter value.

    Raises
    ------
    TypeError
        If the parameter value is not of the expected type given by 'expected_type' or the
        elements are not of the expected type given by 'expected_type_element'.
    ValueError
        If the parameter value does not have the expected length (if provided).
    """

    if par_value is None and none_default:
        return par_value

    parameter_map = {"dict": dict,
                     "list": list,
                     "tuple": tuple,
                     "array": ndarray,
                     "array2D": ndarray,
                     "array3D": ndarray}

    value_parameter_error("expected_type", expected_type, parameter_map.keys())

    element_map = {"int": int,
                   "float": float,
                   "list": list,
                   "NoneType": type(None),
                   "str": str,
                   "tuple": tuple}

    # Checking the parameter type
    if not isinstance(par_value, parameter_map[expected_type]):
        raise TypeError(f"'{par_name}' must be a {expected_type}, "
                        f"got {type(par_value).__name__}.")

    if expected_type in ["array2D", "array3D"]:
        expected_dim = 2 if expected_type == "array2D" else 3
        if par_value.ndim != expected_dim:
            raise ValueError(f"'{par_name}' must be a {expected_dim}D array, "
                             f"got {par_value.ndim}D array.")

        if expected_shape is not None:
            par_shape = par_value.shape
            for dim in range(expected_dim):
                if expected_shape[dim] is not None and \
                        par_shape[dim] != expected_shape[dim]:
                    raise ValueError(f"'{par_name}' must have shape {expected_shape}, "
                                     f"got shape {par_shape}.")

    # Check if the parameter has the expected length
    if expected_length is not None and len(par_value) != expected_length:
        raise ValueError(f"'{par_name}' must have length {expected_length}, "
                         f"got length {len(par_value)}.")

    # Check if all elements are of expected type
    if expected_type_element is not None:
        if isinstance(expected_type_element, str):
            expected_type_element = (expected_type_element,)
        for etype in expected_type_element:
            value_parameter_error("expected_type_element", etype, element_map.keys())

        expected_types = tuple(element_map[etype] for etype in expected_type_element)
        expected_types += (int32, int64,) if "int" in expected_type_element else ()
        expected_types += (float32, float64,) if "float" in expected_type_element else ()
        par_value_check = par_value if expected_type not in ["array2D", "array3D"] \
            else par_value.flatten()
        if not all(isinstance(item, expected_types) for item in par_value_check):
            opt_str = ", ".join([f"'{etype}'" for etype in expected_type_element])
            last_comma = opt_str.rfind(',')
            opt_str = opt_str[:last_comma] + " or" + opt_str[last_comma + 1:] \
                if len(expected_type_element) > 1 else opt_str
            raise TypeError(f"All elements of '{par_name}' must be of type: {opt_str}.")

    return par_value


def type_firedrake_error(par_name, par_value, expected_type, none_default=False):
    """Validate Firedrake parameters and raise a TypeError if invalid.

    Parameters
    ----------
    par_name : `str`
        Name of the parameter to be validated (used in error messages).
    par_value : `object`
        Value of the parameter to be validated.
    expected_type : `str`
        Expected type of the Firedrake parameter as a `str`. The validation
        supports the types `Function`, `FunctionSpace`, or `Mesh`.
    none_default : `bool`, optional
        If `True`, the parameter value is allowed to be validated as `None`.
        Default is `False`, in which case `None` is not allowed.

    Returns
    -------
    par_value : `firedrake.Function`, `firedrake.FunctionSpace`, or `firedrake.Mesh`
        The validated parameter value.

    Raises
    ------
    TypeError
        If the parameter value is not of the expected type given by 'expected_type'.
    """

    if par_value is None and none_default:
        return par_value

    parameter_map = {"Form": Form,
                     "Function": Function,
                     "FunctionSpace": type(FunctionSpace),
                     "Mesh": type(Mesh),
                     "SpatialCoordinate": SpatialCoordinate}

    value_parameter_error("expected_type", expected_type, parameter_map.keys())

    # Checking the parameter type
    expected_valid = (parameter_map[expected_type],)
    expected_valid += (WithGeometry,) if "FunctionSpace" in expected_type else ()
    if not isinstance(par_value, expected_valid):
        opt_str = ", ".join([f"'{etype}'" for etype in expected_valid])
        last_comma = opt_str.rfind(',')
        opt_str = opt_str[:last_comma] + " or" + opt_str[last_comma + 1:] \
            if len(expected_valid) > 1 else opt_str
        raise TypeError(f"'{par_name}' must be of type: {opt_str}, "
                        f"got {type(par_value).__name__}.")

    return par_value


def value_file_error(par_name, par_value, valid_extensions,
                     none_default=False, check_file_existance=False):
    """Validate a file parameter and raise a ValueError if invalid.

    Parameters
    ----------
    par_name : `str`
        Name of the parameter to be validated (used in error messages).
    par_value : `str`
        Value of the parameter to be validated.
    valid_extensions : `list`
        List of valid file extensions for the parameter.
    none_default : `bool`, optional
        If `True`, the parameter value is allowed to be validated as `None`.
        Default is `False`, in which case `None` is not allowed.
    check_file_existance : `bool`, optional
        If `True`, we will check if the file existis. Default is `False`

    Returns
    -------
    par_value : `str`
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

    par_value = value_string_error(par_name, par_value, none_default=none_default)

    if par_value is None and none_default:
        return par_value

    if not path.exists(par_value) and check_file_existance:
        raise FileNotFoundError(f"'{par_name}' '{par_value}' does not exist")

    file_extension = path.splitext(par_value)[1]
    value_parameter_error("extension_type", file_extension, valid_extensions)

    return par_value
