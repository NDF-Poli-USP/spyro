"""Statistical tools utilities.

- Calculation the coefficient of determination (R^2) for regression models."""

from numpy import asarray, mean, ndarray, sum
from ..utils.error_management import value_numerical_error


def coeff_of_determination(y_true, y_pred, p):
    '''
    Compute the coefficient of determination (R^2) for regression models.

    Parameters
    ----------
    y_true : `list`
        True target values
    y_pred : `list`
        Predicted target values from the regression model
    p : `int`
        Number of predictors (independent variables) in the model.
        Example: Linear regression (a*x + b) has 1 predictor (x) while
        quadratic regression (a*x^2 + b*x + c) has 2 predictors (x^2 and x)

    Returns
    -------
    r2_adj : `float`
        Adjusted R^2 value
    '''

    # Validate input arguments
    if not isinstance(y_true, ndarray):
        raise TypeError(f"'y_true' must be a array, got {type(y_true).__name__}.")

    if not isinstance(y_pred, ndarray):
        raise TypeError(f"'y_pred' must be a array, got {type(y_pred).__name__}.")

    # Length checking
    nt = len(y_true)
    np = len(y_pred)
    if nt != np:
        raise ValueError(f"Length mismatch: 'y_true' has {nt} elements, "
                         f"but 'y_pred' has {np} elements.")

    # Checking predictors
    value_numerical_error('p', p, float_num=False, integer_num=True,
                          lower_bound=0., include_lower_bound=True)

    # Observations
    n = len(y_true)

    # Convert list to array
    y_true = asarray(y_true)
    y_pred = asarray(y_pred)

    # R^2 calculation
    r2 = 1.
    ss_res = sum((y_true - y_pred) ** 2)
    ss_tot = sum((y_true - mean(y_true)) ** 2)

    if ss_tot > 0. and ss_res > 0.:
        r2 -= (ss_res / ss_tot)

    # Adjusted R^2 calculation (if applicable)
    r2_adj = r2 if (n - p - 1) <= 0 else 1. - (1. - r2) * (n - 1.) / (n - p - 1.)

    return r2_adj
