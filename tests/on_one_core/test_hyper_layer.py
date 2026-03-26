import math
import numpy as np
import pytest

from spyro.habc.hyp_lay import HyperLayer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def layer_2d():
    """2D HyperLayer with a 1 km x 1 km domain and 0.25 km pad."""
    hl = HyperLayer((1., 1.), n_hyp=2, dimension=2)
    # domain_hyp = domain + 2 * pad  →  [1.0 + 0.5, 1.0 + 0.5]
    hl.define_hyperaxes((1.5, 1.5))
    return hl


@pytest.fixture
def layer_3d():
    """3D HyperLayer with a 1 km³ domain and 0.25 km pad."""
    hl = HyperLayer((1., 1., 1.), n_hyp=2, dimension=3)
    hl.define_hyperaxes((1.5, 1.5, 1.5))
    return hl

# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_defaults():
    hl = HyperLayer((1., 1.))
    assert hl.domain_dim == (1.0, 1.0)
    assert hl.n_hyp == 2.
    assert hl.dimension == 2


def test_init_custom():
    hl = HyperLayer((1., 1., 1.), n_hyp=4, dimension=3)
    assert hl.domain_dim == (1.0, 1.0, 1.0)
    assert hl.n_hyp == 4.
    assert hl.dimension == 3


def test_init_none_n_hyp_raises_error():
    # Test that passing None raises a TypeError
    with pytest.raises(TypeError, match="n_hyp must be a number"):
        HyperLayer((1., 1.), n_hyp=None)

# ---------------------------------------------------------------------------
# define_hyperaxes
# ---------------------------------------------------------------------------


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
# radial_parameter
# ---------------------------------------------------------------------------


def test_radial_parameter_2d(layer_2d):
    """Test radial parameter calculation for 2D"""
    # Point on the boundary should give r = 1
    a, b = layer_2d.hyper_axes
    # For a circle (n=2), point at (a*cos(45°), b*sin(45°)) should be on boundary
    x = a * np.cos(np.pi/4)
    y = b * np.sin(np.pi/4)
    r = layer_2d.radial_parameter((x, y), 2)
    assert math.isclose(r, 1.0, rel_tol=1e-10)

    # Point inside should give r < 1
    r_inside = layer_2d.radial_parameter((x/2, y/2), 2)
    assert r_inside < 1.0

    # Point outside should give r > 1
    r_outside = layer_2d.radial_parameter((x*1.5, y*1.5), 2)
    assert r_outside > 1.0


def test_radial_parameter_3d(layer_3d):
    """Test radial parameter calculation for 3D"""
    a, b, c = layer_3d.hyper_axes
    # Point on the boundary should give r = 1
    x = a / np.sqrt(3)
    y = b / np.sqrt(3)
    z = c / np.sqrt(3)
    r = layer_3d.radial_parameter((x, y, z), 2)
    assert math.isclose(r, 1.0, rel_tol=1e-10)

    # Point inside should give r < 1
    r_inside = layer_3d.radial_parameter((x/2, y/2, z/2), 2)
    assert r_inside < 1.0

    # Point outside should give r > 1
    r_outside = layer_3d.radial_parameter((x*1.5, y*1.5, z*1.5), 2)
    assert r_outside > 1.0

# ---------------------------------------------------------------------------
# central_tendency_criteria
# ---------------------------------------------------------------------------


def test_central_tendency_criteria_2d(layer_2d):
    """Test central tendency criteria for 2D"""
    # Superness coordinates (should be inside the hyperlayer)
    a, b = layer_2d.hyper_axes
    xs = a * 0.8
    ys = b * 0.8
    crit_tend = layer_2d.central_tendency_criteria((xs, ys))
    # Can be None or a list of values
    if crit_tend is not None:
        assert isinstance(crit_tend, list)
        for val in crit_tend:
            assert isinstance(val, (float, np.floating))


def test_central_tendency_criteria_3d(layer_3d):
    """Test central tendency criteria for 3D"""
    # Superness coordinates (should be inside the hyperlayer)
    a, b, c = layer_3d.hyper_axes
    xs = a * 0.8
    ys = b * 0.8
    zs = c * 0.8
    crit_tend = layer_3d.central_tendency_criteria((xs, ys, zs))
    # Can be None or a list of values
    if crit_tend is not None:
        assert isinstance(crit_tend, list)
        for val in crit_tend:
            assert isinstance(val, (float, np.floating))

# ---------------------------------------------------------------------------
# loop_criteria
# ---------------------------------------------------------------------------


def test_loop_criteria_2d(layer_2d):
    """Test loop criteria for 2D"""
    a, b = layer_2d.hyper_axes
    spness = (a * 0.8, b * 0.8)
    n, n_min, n_max = layer_2d.loop_criteria(spness, n_min=2, n_max=20)
    assert isinstance(n, (float, np.floating))
    assert n_min <= n <= n_max


def test_loop_criteria_3d(layer_3d):
    """Test loop criteria for 3D"""
    a, b, c = layer_3d.hyper_axes
    spness = (a * 0.8, b * 0.8, c * 0.8)
    n, n_min, n_max = layer_3d.loop_criteria(spness, n_min=2, n_max=20)
    assert isinstance(n, (float, np.floating))
    assert n_min <= n <= n_max

# ---------------------------------------------------------------------------
# half_hyp_area  (docstring: half_hyp_area(1, 1, 2) == pi/2)
# ---------------------------------------------------------------------------


def test_half_hyp_area_unit_circle():
    result = HyperLayer.half_hyp_area(1.0, 1.0, 2)
    assert math.isclose(result, math.pi / 2, rel_tol=1e-9)


def test_half_hyp_area_positive():
    assert HyperLayer.half_hyp_area(2.0, 3.0, 4) > 0


def test_half_hyp_area_scales_linearly_with_a():
    A1 = HyperLayer.half_hyp_area(1.0, 1.0, 4)
    A2 = HyperLayer.half_hyp_area(2.0, 1.0, 4)
    assert math.isclose(A2, 2.0 * A1, rel_tol=1e-9)


def test_half_hyp_area_scales_linearly_with_b():
    A1 = HyperLayer.half_hyp_area(1.0, 1.0, 4)
    A2 = HyperLayer.half_hyp_area(1.0, 3.0, 4)
    assert math.isclose(A2, 3.0 * A1, rel_tol=1e-9)

# ---------------------------------------------------------------------------
# trunc_half_hyp_area  (docstring: trunc_half_hyp_area(1, 1, 2, 1) == pi/2)
# ---------------------------------------------------------------------------


def test_trunc_half_hyp_area_no_truncation():
    # z0 == b means no truncation; result should equal half_hyp_area
    result = HyperLayer.trunc_half_hyp_area(1.0, 1.0, 2, 1.0)
    assert math.isclose(result, math.pi / 2, rel_tol=1e-6)


def test_trunc_half_hyp_area_zero_plane():
    # z0 == 0 means full truncation; result should be 0
    result = HyperLayer.trunc_half_hyp_area(1.0, 1.0, 2, 0.0)
    assert math.isclose(result, 0.0, abs_tol=1e-12)


def test_trunc_half_hyp_area_less_than_half():
    # Truncating at z0 < b should give less area than the half
    full = HyperLayer.half_hyp_area(1.0, 2.0, 4)
    trunc = HyperLayer.trunc_half_hyp_area(1.0, 2.0, 4, 1.0)  # z0 = b/2
    assert 0 < trunc < full

# ---------------------------------------------------------------------------
# half_hyp_volume  (docstring: half_hyp_volume(1, 1, 1, 2) == 2*pi/3)
# ---------------------------------------------------------------------------


def test_half_hyp_volume_unit_sphere():
    result = HyperLayer.half_hyp_volume(1.0, 1.0, 1.0, 2)
    assert math.isclose(result, 2 * math.pi / 3, rel_tol=1e-9)


def test_half_hyp_volume_positive():
    assert HyperLayer.half_hyp_volume(1.0, 2.0, 3.0, 4) > 0


def test_half_hyp_volume_scales_linearly_with_c():
    V1 = HyperLayer.half_hyp_volume(1.0, 1.0, 1.0, 4)
    V2 = HyperLayer.half_hyp_volume(1.0, 1.0, 2.0, 4)
    assert math.isclose(V2, 2.0 * V1, rel_tol=1e-9)

# ---------------------------------------------------------------------------
# trunc_half_hyp_volume  (docstring: trunc_half_hyp_volume(1,1,1,2,1) == 2*pi/3)
# ---------------------------------------------------------------------------


def test_trunc_half_hyp_volume_no_truncation():
    result = HyperLayer.trunc_half_hyp_volume(1.0, 1.0, 1.0, 2, 1.0)
    assert math.isclose(result, 2 * math.pi / 3, rel_tol=1e-6)


def test_trunc_half_hyp_volume_zero_plane():
    result = HyperLayer.trunc_half_hyp_volume(1.0, 1.0, 1.0, 2, 0.0)
    assert math.isclose(result, 0.0, abs_tol=1e-12)


def test_trunc_half_hyp_volume_less_than_half():
    full = HyperLayer.half_hyp_volume(1.0, 2.0, 1.0, 4)
    trunc = HyperLayer.trunc_half_hyp_volume(1.0, 2.0, 1.0, 4, 1.0)  # z0 = b/2
    assert 0 < trunc < full

# ---------------------------------------------------------------------------
# hyp_full_perimeter  (docstring: hyp_full_perimeter(1,1,2) == 2*pi)
# ---------------------------------------------------------------------------


def test_hyp_full_perimeter_unit_circle():
    # n=2, a=b=1 gives a unit circle with perimeter 2*pi
    result = HyperLayer.hyp_full_perimeter(1.0, 1.0, 2)
    assert math.isclose(result, 2 * math.pi, rel_tol=1e-6)


def test_hyp_full_perimeter_positive():
    assert HyperLayer.hyp_full_perimeter(1.0, 2.0, 4) > 0


def test_hyp_full_perimeter_increases_with_axes():
    p1 = HyperLayer.hyp_full_perimeter(1.0, 1.0, 2)
    p2 = HyperLayer.hyp_full_perimeter(2.0, 2.0, 2)
    assert p2 > p1

# ---------------------------------------------------------------------------
# calc_degree_hypershape (2D)
# ---------------------------------------------------------------------------


def test_calc_degree_hyp2D_min_returns_valid_range(layer_2d):
    x_rel = (0.5 + 0.01, 0.5 + 0.01)
    n = layer_2d.calc_degree_hypershape(x_rel, 'MIN')
    assert 2. <= n <= 20.


def test_calc_degree_hyp2D_max_returns_valid_range(layer_2d):
    x_rel = (0.5 + 0.15, 0.5 + 0.15)
    n = layer_2d.calc_degree_hypershape(x_rel, 'MAX', n_min=2)
    assert 2. <= n <= 20.


def test_calc_degree_hyp2D_min_le_max(layer_2d):
    x_min = (0.5 + 0.01, 0.5 + 0.01)
    x_max = (0.5 + 0.15, 0.5 + 0.15)
    n_min = layer_2d.calc_degree_hypershape(x_min, 'MIN')
    n_max = layer_2d.calc_degree_hypershape(x_max, 'MAX', n_min=n_min)
    assert n_min <= n_max


def test_calc_degree_hyp2D_returns_float(layer_2d):
    n = layer_2d.calc_degree_hypershape((0.51, 0.51), 'MIN')
    assert isinstance(n, (float, np.floating))

# ---------------------------------------------------------------------------
# calc_degree_hypershape (3D)
# ---------------------------------------------------------------------------


def test_calc_degree_hyp3D_min_returns_valid_range(layer_3d):
    x_rel = (0.5 + 0.01, 0.5 + 0.01, 0.5 + 0.01)
    n = layer_3d.calc_degree_hypershape(x_rel, 'MIN')
    assert 2. <= n <= 20.


def test_calc_degree_hyp3D_max_returns_valid_range(layer_3d):
    x_rel = (0.5 + 0.15, 0.5 + 0.15, 0.5 + 0.15)
    n = layer_3d.calc_degree_hypershape(x_rel, 'MAX', n_min=2)
    assert 2. <= n <= 20.


def test_calc_degree_hyp3D_min_le_max(layer_3d):
    x_min = (0.5 + 0.01, 0.5 + 0.01, 0.5 + 0.01)
    x_max = (0.5 + 0.15, 0.5 + 0.15, 0.5 + 0.15)
    n_min = layer_3d.calc_degree_hypershape(x_min, 'MIN')
    n_max = layer_3d.calc_degree_hypershape(x_max, 'MAX', n_min=n_min)
    assert n_min <= n_max

# ---------------------------------------------------------------------------
# define_hyperlayer
# ---------------------------------------------------------------------------


def test_define_hyperlayer_stores_n_bounds(layer_2d):
    layer_2d.define_hyperlayer(pad_len=0.25, lmin=0.01)
    assert hasattr(layer_2d, 'n_bounds')
    n_min, n_max = layer_2d.n_bounds
    assert n_min >= 2.
    assert n_max >= n_min


def test_define_hyperlayer_clamps_n_hyp_up(layer_2d):
    layer_2d.n_hyp = 1.  # below any valid n_min
    layer_2d.define_hyperlayer(pad_len=0.25, lmin=0.01)
    n_min, _ = layer_2d.n_bounds
    assert layer_2d.n_hyp >= n_min


def test_define_hyperlayer_clamps_n_hyp_down(layer_2d):
    layer_2d.n_hyp = 100  # above any valid n_max
    layer_2d.define_hyperlayer(pad_len=0.25, lmin=0.01)
    _, n_max = layer_2d.n_bounds
    assert layer_2d.n_hyp <= n_max


def test_define_hyperlayer_valid_n_hyp_unchanged(layer_2d):
    layer_2d.n_hyp = 4.
    layer_2d.define_hyperlayer(pad_len=0.25, lmin=0.01)
    n_min, n_max = layer_2d.n_bounds
    # n_hyp should remain within bounds
    assert n_min <= layer_2d.n_hyp <= n_max

# ---------------------------------------------------------------------------
# calc_hyp_geom_prop
# ---------------------------------------------------------------------------


def test_calc_hyp_geom_prop_2d_positive_area(layer_2d):
    domain_hyp = (1.5, 1.5)  # domain + 2*pad
    pad_len = 0.25
    lmin = 0.01
    layer_2d.calc_hyp_geom_prop(domain_hyp, pad_len, lmin)
    assert layer_2d.area > 0
    assert layer_2d.area_ratio > 0
    assert layer_2d.f_Ah > 0
    assert layer_2d.perim_hyp > 0


def test_calc_hyp_geom_prop_2d_area_ratio_above_one(layer_2d):
    # The padded area must be larger than the original domain area
    domain_hyp = (1.5, 1.5)  # domain + 2*pad
    pad_len = 0.25
    lmin = 0.01
    layer_2d.calc_hyp_geom_prop(domain_hyp, pad_len, lmin)
    assert layer_2d.area_ratio > 1.0


def test_calc_hyp_geom_prop_3d_positive_volume(layer_3d):
    domain_hyp = (1.5, 1.5, 1.5)  # domain + 2*pad
    pad_len = 0.25
    lmin = 0.01
    layer_3d.calc_hyp_geom_prop(domain_hyp, pad_len, lmin)
    assert layer_3d.vol > 0
    assert layer_3d.vol_ratio > 0
    assert layer_3d.f_Vh > 0
    assert layer_3d.surf_hyp > 0


def test_calc_hyp_geom_prop_3d_volume_ratio_above_one(layer_3d):
    domain_hyp = (1.5, 1.5, 1.5)  # domain + 2*pad
    pad_len = 0.25
    lmin = 0.01
    layer_3d.calc_hyp_geom_prop(domain_hyp, pad_len, lmin)
    assert layer_3d.vol_ratio > 1.0
