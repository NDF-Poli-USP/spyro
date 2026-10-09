"""Choice of a regularization weight from the L-curve."""

import numpy as np


def l_curve_corner(weights, misfits, penalties) -> int:
    r"""Return the index of the run at the corner of the L-curve.

    Each run, inverted with its own regularization weight, gives a point of
    the L-curve: the data misfit against the penalty without its weight,
    both on log scales. Small weights fit the data and leave a large
    penalty, large weights the other way round, so the curve has the shape
    of an L. Its corner is the weight that balances the two [Hansen1993]_.

    With a few runs, the curvature at a point is that of the circle through
    it and its two neighbours (the Menger curvature), with the sign of the
    turn. Along increasing weights the L turns counterclockwise at its
    corner, so the corner is the point of largest positive curvature.

    Parameters
    ----------
    weights : sequence of float
        Regularization weight of each run, in any order.
    misfits : sequence of float
        Final data misfit of each run.
    penalties : sequence of float
        Final penalty of each run, without its weight.

    Returns
    -------
    int
        Index, into the given sequences, of the run at the corner.

    Raises
    ------
    ValueError
        If there are fewer than three runs, the sequences differ in length,
        a value is not finite and positive, or no point turns
        counterclockwise.

    References
    ----------
    .. [Hansen1993] P. C. Hansen and D. P. O'Leary, "The use of the L-curve
       in the regularization of discrete ill-posed problems", SIAM Journal
       on Scientific Computing 14(6), 1487-1503, 1993.
    """
    weights, misfits, penalties = (
        np.asarray(values, dtype=float) for values in (weights, misfits, penalties)
    )
    if not weights.shape == misfits.shape == penalties.shape or weights.ndim != 1:
        raise ValueError("weights, misfits and penalties need one value per run.")
    if weights.size < 3:
        raise ValueError("The L-curve needs at least three runs.")
    values = np.concatenate((weights, misfits, penalties))
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Weights, misfits and penalties must be finite and positive.")

    order = np.argsort(weights)
    points = np.column_stack((np.log(misfits[order]), np.log(penalties[order])))
    curvatures = np.full(weights.size, -np.inf)
    for i in range(1, weights.size - 1):
        before = points[i] - points[i - 1]
        after = points[i + 1] - points[i]
        chord = points[i + 1] - points[i - 1]
        lengths = (np.linalg.norm(before) * np.linalg.norm(after)
                   * np.linalg.norm(chord))
        if lengths > 0:
            cross = before[0] * after[1] - before[1] * after[0]
            curvatures[i] = 2.0 * cross / lengths
    corner = int(np.argmax(curvatures))
    if curvatures[corner] <= 0:
        raise ValueError("No run turns the L-curve counterclockwise: no corner.")
    return int(order[corner])
