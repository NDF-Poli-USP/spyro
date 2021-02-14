import numpy as np


def create_transect(start, end, num):
    """Create a line of `num` of points between coordinates
    `start` and `end`

    Parameters
    ----------
    start: tuple of floats
        starting position coordinate
    end: tuple of floats
        ending position coordinate
    num: integer
        number of receivers between `start` and `end`

    Returns
    -------
    receiver_locations: array-like

    """
    return np.linspace(start, end, num)


def create_2d_grid(start1, end1, start2, end2, num):
    """Create a 2d grid of `num**2` points between `start1`
    and `end1` and `start2` and `end2`

    Parameters
    ----------
    start1: tuple of floats
        starting position coordinate
    end1: tuple of floats
        ending position coordinate
    start2: tuple of floats
        starting position coordinate
    end2: tuple of floats
        ending position coordinate
    num: integer
        number of receivers between `start` and `end`

    Returns
    -------
    receiver_locations: a list of tuples

    """
    x = np.linspace(start1, end1, num)
    y = np.linspace(start2, end2, num)
    X, Y = np.meshgrid(x, y)
    points = np.vstack((X.flatten(), Y.flatten())).T
    return [tuple(point) for point in points]


def insert_fixed_value(points, value, insert):
    """Insert a fixed `value` in each
    tuple at index `insert`.

    Parameters
    ----------
    points: a list of tuples
        A bunch of point coordinates
    value: float
        The constant value to insert
    insert: int
        The position to insert the `value`
    """
    tmp = [list(point) for point in points]
    for point in tmp:
        point.insert(insert, value)
    return [tuple(point) for point in tmp]
