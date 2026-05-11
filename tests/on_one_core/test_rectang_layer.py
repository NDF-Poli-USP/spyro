import math
import numpy as np
import pytest

from spyro.abc.rec_lay import RectangLayer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def layer_2d():
    """2D RectangLayer with a 1 km x 1 km domain and 0.25 km pad."""
    hl = RectangLayer((1., 1.), dimension=2)
    return hl


@pytest.fixture
def layer_3d():
    """3D RectangLayer with a 1 km³ domain and 0.25 km pad."""
    hl = RectangLayer((1., 1., 1.), dimension=3)
    return hl

# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_defaults_2d():
    hl = RectangLayer((1., 1.))
    assert hl.domain_dim == (1.0, 1.0)
    assert hl.dimension == 2


def test_init_defaults_3d():
    hl = RectangLayer((1., 1., 1.), dimension=3)
    assert hl.domain_dim == (1.0, 1.0, 1.0)
    assert hl.dimension == 3


def test_init_list_domain_dim_raises_error():
    # Test that passing None raises a TypeError
    with pytest.raises(TypeError, match="domain_dim must be a tuple"):
        RectangLayer([1., 1.])


def test_init_dimension_raises_error():
    # Test that passing None raises a TypeError
    with pytest.raises(ValueError, match=f"Invalid dimension: '10'."):
        RectangLayer((1., 1.), dimension=10)

# ---------------------------------------------------------------------------
# rectangular_area  (docstring: calc_rec_geom_prop((1, 1), 0.5) == pi/2)
# ---------------------------------------------------------------------------


# def test_half_hyp_area_unit_circle():
#     result = HyperLayer.half_hyp_area(1.0, 1.0, 2)
#     assert math.isclose(result, math.pi / 2, rel_tol=1e-9)


# def test_half_hyp_area_positive():
#     assert HyperLayer.half_hyp_area(2.0, 3.0, 4) > 0


# def test_half_hyp_area_scales_linearly_with_a():
#     A1 = HyperLayer.half_hyp_area(1.0, 1.0, 4)
#     A2 = HyperLayer.half_hyp_area(2.0, 1.0, 4)
#     assert math.isclose(A2, 2.0 * A1, rel_tol=1e-9)


# def test_half_hyp_area_scales_linearly_with_b():
#     A1 = HyperLayer.half_hyp_area(1.0, 1.0, 4)
#     A2 = HyperLayer.half_hyp_area(1.0, 3.0, 4)
#     assert math.isclose(A2, 3.0 * A1, rel_tol=1e-9)
