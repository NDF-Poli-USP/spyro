import math
from copy import deepcopy
import spyro

"""Read in an external mesh and interpolate velocity to it"""
from ..inputfiles.Model1_2d_CG import model as oldmodel


def test_ricker_varies_in_time():
    """This test ricker time variation when applied to a time-
    dependent PDE (acoustic wave second order in pressure) in
    firedrake. It tests if the right hand side varies in time
    and if the applied ricker function behaves correctly
    """

    # initial ricker tests
    modelRicker = deepcopy(oldmodel)
    frequency = 2
    amplitude = 3

    # tests if ricker starts at zero
    delay = 1.5 * math.sqrt(6.0) / (math.pi * frequency)
    t = 0.0
    r0 = spyro.sources.timedependentSource(modelRicker, t, frequency, amplitude=amplitude)
    test1 = math.isclose(
        r0,
        0,
        abs_tol=1e-3,
    )

    # tests if the minimum value is correct and occurs at correct locations
    minimum = -amplitude * 2 / math.exp(3.0 / 2.0)
    t = 0.0 + delay + math.sqrt(6.0) / (2.0 * math.pi * frequency)
    rmin1 = spyro.sources.timedependentSource(
        modelRicker, t, frequency, amplitude=amplitude
    )
    test2 = math.isclose(
        rmin1,
        minimum,
    )

    t = 0.0 + delay - math.sqrt(6.0) / (2.0 * math.pi * frequency)
    rmin2 = spyro.sources.timedependentSource(
        modelRicker, t, frequency, amplitude=amplitude
    )
    test3 = math.isclose(rmin2, minimum)

    # tests if maximum value in correct and occurs at correct location
    t = 0.0 + delay
    rmax = spyro.sources.timedependentSource(
        modelRicker, t, frequency, amplitude=amplitude
    )
    test4 = math.isclose(
        rmax,
        amplitude,
    )

    assert all([test1, test2, test3, test4])


if __name__ == "__main__":
    test_ricker_varies_in_time()
