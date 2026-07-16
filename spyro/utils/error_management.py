"""Error management utilities.

This file contains methods for handling errors in Spyro, either to send
messages to the user or to prevent numerical instability in objects."""

from numpy import inf, isinf, isnan, where


def value_parameter_error(par_name, par_value, valid_values):
    """Raise a ValueError with a specific error message.

    Parameters
    ----------
    par_name : `str`
        Name of the parameter that has an invalid value.
    par_value : `str`, `int` or `float`
        Value of the parameter that is invalid.
    valid_values : `list`
        List of valid values for the parameter.

    Raises
    ------
    ValueError
        If the parameter value is not in the list of valid values.
    """

    # Error message about the invalid parameter
    err_str = f"Invalid {par_name}: '{par_value}'. Please use: "
    opt_str = ", ".join([f"'{val}'" for val in valid_values])
    last_comma = opt_str.rfind(',')
    opt_str = opt_str[:last_comma] + " or" + opt_str[last_comma + 1:] \
        if len(valid_values) > 1 else opt_str

    raise ValueError(err_str + opt_str)


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


def value_dimension_error(par_names, par_values, expected_dim):
    """Raise a ValueError if parameter dimensions mismatch expected dimension.

    Parameters
    ----------
    par_names : `tuple`
        Names of the parameters to check dimensions.
    par_values : `tuple`
        Values of the parameters to check dimensions.
    expected_dim : `int`
        Expected dimension of the parameters (2 or 3).

    Raises
    ------
    ValueError
        If the dimensions of the parameters do not match the expected dimension.
    """

    str_reference, str_comparison = par_names
    chk_reference, chk_comparison = par_values

    dim_err = (
        f"Mismatch in domain dimensions\n"
        f"{str_reference} ({chk_reference}), {str_comparison} ({chk_comparison}) "
        f"do not match expected model dimension ({expected_dim}D).")
    raise ValueError(dim_err)


def clean_inst_num(data_arr):
    """Set NaNs and negative values to zero in an array.

    Parameters
    ----------
    data_arr : `array`
        An array with possible with possible NaN or negative components.

    Returns
    -------
    data_arr : `array`
        An array with null or positive components.
    """
    data_arr[where(isnan(data_arr) | isinf(data_arr) | (data_arr < 0.0))] = 0.0
    return data_arr


def value_numerical_error(par_name, par_value, float_num=True, integer_num=False,
                          lower_bound=None, upper_bound=None,
                          include_lower_bound=False, include_upper_bound=False):
    """Raise a ValueError with a specific error message for numerical parameters.

    Parameters
    ----------
    par_name : `str`
        Name of the parameter that has an invalid value.
    par_value : `int` or `float`
        Value of the parameter that is invalid.
    float_num : `bool`, optional
        If `True`, the parameter can be a float. Default is `True`.
    integer_num : `bool`, optional
        If `True`, the parameter can be an integer. Default is `False`.
    lower_bound : `int` or `float`, optional
        Lower bound for the parameter value. Default is `None` (no lower bound).
    upper_bound : `int` or `float`, optional
        Upper bound for the parameter value. Default is `None` (no upper bound).
    include_lower_bound : `bool`, optional
        If `True`, the lower bound is included in the valid range. Default is `False`.
    include_upper_bound : `bool`, optional
        If `True`, the upper bound is included in the valid range. Default is `False`.

    Raises
    ------
    TypeError
        If the parameter value is not of the expected type (float or integer).
    ValueError
        If the parameter value is outside the specified bounds or the bounds are invalid.
    """

    # Checking the parameter type
    if not isinstance(par_value, (int, float)):
        if float_num and integer_num:
            str_type = "float or a integer"
        elif float_num:
            str_type = "float"
        elif integer_num:
            str_type = "integer"

        raise TypeError(f"'{par_name}' must be a {str_type} number, "
                        f"got {type(par_value).__name__}.")

    # Set default bounds
    upper_bound = inf if upper_bound is None else upper_bound
    lower_bound = -inf if lower_bound is None else lower_bound

    # Validate bounds
    if upper_bound <= lower_bound:
        raise ValueError(f"Invalid bounds: upper_bound ({upper_bound}) must "
                         f"be greater than lower_bound ({lower_bound}).")

    # Check if value is within bounds
    if par_value < lower_bound or par_value > upper_bound:
        if lower_bound > -inf and upper_bound < inf:  # Both bounds are finite
            bound_str = f"between {lower_bound} and {upper_bound}"
            if include_lower_bound and include_upper_bound:
                bound_str += " (both bounds inclusive)"
            elif include_lower_bound:
                bound_str += " (lower bound inclusive)"
            elif include_upper_bound:
                bound_str += " (upper bound inclusive)"
        elif lower_bound > -inf:  # Only lower bound is finite
            bound_str = (f"greater than or equal to {lower_bound}"
                         if include_lower_bound else f"greater than {lower_bound}")
        elif upper_bound < inf:  # Only upper bound is finite
            bound_str = (f"less than or equal to {upper_bound}"
                         if include_upper_bound else f"less than {upper_bound}")

        raise ValueError(f"'{par_name}' must be {bound_str}, got {par_value}.")


def enum_parameter_error(par_name, par_value, valid_enum):
    """Validate and convert an enum parameter, returning the enum instance.

    This method validates that the provided parameter value is either an
    instance of the specified enum class or a string that maps to a valid
    enum value. If valid, it returns the corresponding enum instance.
    Otherwise, it raises an appropriate exception.

    Parameters
    ----------
    par_name : `str`
        Name of the parameter being validated (used in error messages).
    par_value : `object`
        Value of the parameter to validate. Can be an `enum.EnumMeta` or a `str`.
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
        if par_value not in valid_values:
            value_parameter_error(par_name, par_value, valid_values)
        return valid_enum(par_value)

    # Invalid type - neither enum instance nor string
    raise TypeError(f"'{par_name}' must be {valid_enum.__name__} or str"
                    f", got {type(par_value).__name__}")
