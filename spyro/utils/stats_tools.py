import numpy as np
import scipy.stats


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

    # Observations
    n = len(y_true)

    # Convert list to array
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # R^2 calculation
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Adjusted R^2 calculation (if applicable)
    r2_adj = r2 if (n - p - 1) <= 0 else 1 - (1 - r2) * (n - 1) / (n - p - 1)

    return r2_adj
