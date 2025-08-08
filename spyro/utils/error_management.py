
def value_parameter_error(par_name, par_value, valid_values):
    '''
    Raise a ValueError with a specific error message.

    Parameters
    ----------
    par_name : `str`
        Name of the parameter that has an invalid value
    par_value : `str`, `int` or `float`
        Value of the parameter that is invalid
    valid_values : `list`
        List of valid values for the parameter
    '''

    # Error message about the invalid parameter
    err_str = f"Invalid {par_name}: '{par_value}'. Please use"
    opt_str = ", ".join([f"'{val}'" for val in valid_values])
    last_comma = opt_str.rfind(',')
    opt_str = opt_str[:last_comma] + ' or' + opt_str[last_comma + 1:] \
        if len(valid_values) > 1 else opt_str

    raise ValueError(err_str + opt_str)
