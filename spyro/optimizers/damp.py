import numpy as np

def moving_avg(gamma, m, g):
    """Update moving average

    Parameters
    ----------
    gamma: float
        'forget' factor
    m: array
        current average
    g: array
        data to be included in average

    Returns
    -------
    mn: array
        updated average
    """

    mn = gamma * m + (1 - gamma) * g

    return mn

def remove_bias(gamma, m, counter):
    """Remove bias from moving average

    Parameters
    ----------
    gamma: float
        'forget' factor
    m: array
        current average
    counter: int
        iteration counter

    Returns
    -------
    mn: array
        Average without bias
    """

    mn = m / (1 - gamma ** (counter + 1))

    return mn

def RMSprop(m, v, eps=1e-6):
    """Evaluate damped moving average of gradient

    Parameters
    ----------
    m: array
        first moment vector
    v: array
        second moment vector
    eps: float
        small number to avoid singularity

    Returns
    -------
    g: array
        RMSprop corrected gradient
    """

    g = m / (np.sqrt(v) + eps)

    return g

