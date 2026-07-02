import math
import pytest

from spyro.abc.rec_lay import RectangLayer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def layer_2d():
    """2D RectangLayer with a 1 km x 1 km domain and 0.25 km pad."""
    hl = RectangLayer((1., 1.), dimension=2)
    hl.define_rec_hyperaxes(0.25)
    return hl


@pytest.fixture
def layer_3d():
    """3D RectangLayer with a 1 km³ domain and 0.25 km pad."""
    hl = RectangLayer((1., 1., 1.), dimension=3)
    hl.define_rec_hyperaxes(0.25)
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
    # Test that passing 10. raises a ValueError
    with pytest.raises(ValueError, match="Invalid dimension: '10'."):
        RectangLayer((1., 1.), dimension=10)

# ---------------------------------------------------------------------------
# define_rec_hyperaxes
# ---------------------------------------------------------------------------


def test_negative_pad_length_raises_error():
    # Test that passing a negative pad length raises a ValueError
    hl = RectangLayer((1., 1.))
    with pytest.raises(ValueError, match="'pad_len' must be greater than 0"):
        hl.define_rec_hyperaxes(-0.25)


def test_none_pad_length_raises_error():
    # Test that passing a None pad length raises a TypeError
    hl = RectangLayer((1., 1.))
    with pytest.raises(TypeError, match="'pad_len' must be a float or a integer"):
        hl.define_rec_hyperaxes(None)


def test_define_hyperaxes_2d_stores_domain_dim(layer_2d):
    assert layer_2d.domain_dim == (1.0, 1.0)


def test_define_hyperaxes_2d_semi_axes(layer_2d):
    # a_hyp = 0.5 * 1.5 = 0.75, b_hyp = 0.5 * 1.5 = 0.75
    assert layer_2d.hyper_axes == (0.75, 0.75)


def test_define_hyperaxes_3d_stores_domain_dim(layer_3d):
    assert layer_3d.domain_dim == (1.0, 1.0, 1.0)


def test_define_hyperaxes_3d_semi_axes(layer_3d):
    assert layer_3d.hyper_axes == (0.75, 0.75, 0.75)

# ---------------------------------------------------------------------------
# calc_rec_geom_prop
# ---------------------------------------------------------------------------


def test_calc_rec_geom_prop_2d_area(layer_2d):
    domain_lay = (1.5, 1.25)
    pad_len = 0.25
    layer_2d.calc_rec_geom_prop(domain_lay, pad_len)
    assert layer_2d.area > 0.
    assert layer_2d.area_ratio > 1.
    assert layer_2d.f_Ah > 0.
    assert math.isclose(layer_2d.area, 1.5 * 1.25, rel_tol=1e-9)
    assert math.isclose(layer_2d.area_ratio, 1.5 * 1.25, rel_tol=1e-9)
    assert math.isclose(layer_2d.f_Ah, 4., rel_tol=1e-9)


def test_calc_rec_geom_prop_3d_volume(layer_3d):
    domain_lay = (1.5, 1.25, 1.5)
    pad_len = 0.25
    layer_3d.calc_rec_geom_prop(domain_lay, pad_len)
    assert layer_3d.vol > 0
    assert layer_3d.vol_ratio > 1.
    assert layer_3d.f_Vh > 0
    assert math.isclose(layer_3d.vol, 1.5 * 1.25 * 1.5, rel_tol=1e-9)
    assert math.isclose(layer_3d.vol_ratio, 1.5 * 1.25 * 1.5, rel_tol=1e-9)
    assert math.isclose(layer_3d.f_Vh, 8., rel_tol=1e-9)
