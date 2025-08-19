# This file contains methods for handling errors in Spyro, either to send
# messages to the user or to prevent numerical instability in objects.
import numpy as np


def value_parameter_error(par_name, par_value, valid_values):
    '''
    Raise a ValueError with a specific error message

    Parameters
    ----------
    par_name : `str`
        Name of the parameter that has an invalid value
    par_value : `str`, `int` or `float`
        Value of the parameter that is invalid
    valid_values : `list`
        List of valid values for the parameter

    Raises
    ------
    ValueError
        If the parameter value is not in the list of valid values
    '''

    # Error message about the invalid parameter
    err_str = f"Invalid {par_name}: '{par_value}'. Please use"
    opt_str = ", ".join([f"'{val}'" for val in valid_values])
    last_comma = opt_str.rfind(',')
    opt_str = opt_str[:last_comma] + ' or' + opt_str[last_comma + 1:] \
        if len(valid_values) > 1 else opt_str

    raise ValueError(err_str + opt_str)


def value_dimension_error(par_names, par_values, expected_dim):
    '''
    Raise a ValueError if the dimensions of the parameters do not match
    the expected dimension.
    Parameters
    ----------
    par_names : `tuple`
        Names of the parameters to check dimensions
    par_values : `tuple`
        Values of the parameters to check dimensions
    expected_dim : `int`
        Expected dimension of the parameters (2 or 3)

    Raises
    ------
    ValueError
        If the dimensions of the parameters do not match the expected dimension
    '''

    str_domd, str_habc = par_names
    chk_domd, chk_habc = par_values

    dim_err = (
        f"Mismatch in domain dimensions\n"
        f"{str_domd} ({chk_domd}), {str_habc} ({chk_habc}) do "
        f"not match expected model dimension ({expected_dim}D).")
    raise ValueError(dim_err)


def clean_inst_num(data_arr):
    ''''
    Set NaNs and negative values to zero in an array

    Parameters
    ----------
    data_arr : `array`
        An array with possible with possible NaN or negative components

    Returns
    -------
    data_arr : `array`
        An array with null or positive components
    '''
    data_arr[np.where(np.isnan(data_arr) | np.isinf(
        data_arr) | (data_arr < 0.0))] = 0.0
    return data_arr
