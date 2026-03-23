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
    hl = HyperLayer(n_hyp=2, dimension=2)
    # domain_hyp = domain + 2 * pad  →  [1.0 + 0.5, 1.0 + 0.5]
    hl.define_hyperaxes([1.0, 1.0], [1.5, 1.5])
    return hl


@pytest.fixture
def layer_3d():
    """3D HyperLayer with a 1 km³ domain and 0.25 km pad."""
    hl = HyperLayer(n_hyp=2, dimension=3)
    hl.define_hyperaxes([1.0, 1.0, 1.0], [1.5, 1.5, 1.5])
    return hl


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

def test_init_defaults():
    hl = HyperLayer()
    assert hl.n_hyp == 2
    assert hl.dimension == 2


def test_init_custom():
    hl = HyperLayer(n_hyp=4, dimension=3)
    assert hl.n_hyp == 4
    assert hl.dimension == 3


def test_init_none_n_hyp_defaults_to_2():
    hl = HyperLayer(n_hyp=None)
    assert hl.n_hyp == 2


# ---------------------------------------------------------------------------
# define_hyperaxes
# ---------------------------------------------------------------------------

def test_define_hyperaxes_2d_stores_domain_dim(layer_2d):
    assert layer_2d.domain_dim == [1.0, 1.0]


def test_define_hyperaxes_2d_semi_axes(layer_2d):
    # a_hyp = 0.5 * 1.5 = 0.75, b_hyp = 0.5 * 1.5 = 0.75
    assert layer_2d.hyper_axes == [0.75, 0.75]


def test_define_hyperaxes_3d_stores_domain_dim(layer_3d):
    assert layer_3d.domain_dim == [1.0, 1.0, 1.0]


def test_define_hyperaxes_3d_semi_axes(layer_3d):
    assert layer_3d.hyper_axes == [0.75, 0.75, 0.75]


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
# hyp_full_perimeter
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
# bnd_pnts_hyp2D
# ---------------------------------------------------------------------------

def test_bnd_pnts_hyp2D_output_shape():
    pts = HyperLayer.bnd_pnts_hyp2D(1.0, 1.0, 2, 20)
    assert pts.shape == (20, 2)


def test_bnd_pnts_hyp2D_ellipse_equation():
    """For n=2 all boundary points must satisfy (x/a)^2 + (y/b)^2 == 1."""
    a, b = 2.0, 3.0
    pts = HyperLayer.bnd_pnts_hyp2D(a, b, 2, 100)
    r = (pts[:, 0] / a) ** 2 + (pts[:, 1] / b) ** 2
    assert np.allclose(r, 1.0, atol=1e-10)


def test_bnd_pnts_hyp2D_superellipse_equation():
    """For n=4 all boundary points must satisfy |x/a|^4 + |y/b|^4 == 1."""
    a, b, n = 1.5, 2.5, 4
    pts = HyperLayer.bnd_pnts_hyp2D(a, b, n, 60)
    r = np.abs(pts[:, 0] / a) ** n + np.abs(pts[:, 1] / b) ** n
    assert np.allclose(r, 1.0, atol=1e-10)


def test_bnd_pnts_hyp2D_extrema():
    """The bounding box of the boundary points should fit within [-a,a] x [-b,b]."""
    a, b = 1.0, 2.0
    pts = HyperLayer.bnd_pnts_hyp2D(a, b, 2, 50)
    assert np.all(np.abs(pts[:, 0]) <= a + 1e-10)
    assert np.all(np.abs(pts[:, 1]) <= b + 1e-10)


# ---------------------------------------------------------------------------
# calc_degree_hyp2D
# ---------------------------------------------------------------------------

def test_calc_degree_hyp2D_min_returns_valid_range(layer_2d):
    x_rel = [0.5 + 0.01, 0.5 + 0.01]
    n = layer_2d.calc_degree_hyp2D(x_rel, 'MIN')
    assert 2 <= n <= 20


def test_calc_degree_hyp2D_max_returns_valid_range(layer_2d):
    x_rel = [0.5 + 0.15, 0.5 + 0.15]
    n = layer_2d.calc_degree_hyp2D(x_rel, 'MAX', n_min=2)
    assert 2 <= n <= 20


def test_calc_degree_hyp2D_min_le_max(layer_2d):
    x_min = [0.5 + 0.01, 0.5 + 0.01]
    x_max = [0.5 + 0.15, 0.5 + 0.15]
    n_min = layer_2d.calc_degree_hyp2D(x_min, 'MIN')
    n_max = layer_2d.calc_degree_hyp2D(x_max, 'MAX', n_min=n_min)
    assert n_min <= n_max


def test_calc_degree_hyp2D_returns_int(layer_2d):
    n = layer_2d.calc_degree_hyp2D([0.51, 0.51], 'MIN')
    assert isinstance(n, (int, np.integer))


# ---------------------------------------------------------------------------
# calc_degree_hyp3D
# ---------------------------------------------------------------------------

def test_calc_degree_hyp3D_min_returns_valid_range(layer_3d):
    x_rel = [0.5 + 0.01, 0.5 + 0.01, 0.5 + 0.01]
    n = layer_3d.calc_degree_hyp3D(x_rel, 'MIN')
    assert 2 <= n <= 20


def test_calc_degree_hyp3D_max_returns_valid_range(layer_3d):
    x_rel = [0.5 + 0.15, 0.5 + 0.15, 0.5 + 0.15]
    n = layer_3d.calc_degree_hyp3D(x_rel, 'MAX', n_min=2)
    assert 2 <= n <= 20


def test_calc_degree_hyp3D_min_le_max(layer_3d):
    x_min = [0.5 + 0.01, 0.5 + 0.01, 0.5 + 0.01]
    x_max = [0.5 + 0.15, 0.5 + 0.15, 0.5 + 0.15]
    n_min = layer_3d.calc_degree_hyp3D(x_min, 'MIN')
    n_max = layer_3d.calc_degree_hyp3D(x_max, 'MAX', n_min=n_min)
    assert n_min <= n_max


# ---------------------------------------------------------------------------
# define_hyperlayer
# ---------------------------------------------------------------------------

def test_define_hyperlayer_stores_n_bounds(layer_2d):
    layer_2d.define_hyperlayer(pad_len=0.25, lmin=0.01)
    assert hasattr(layer_2d, 'n_bounds')
    n_min, n_max = layer_2d.n_bounds
    assert n_min >= 2
    assert n_max >= n_min


def test_define_hyperlayer_clamps_n_hyp_up(layer_2d):
    layer_2d.n_hyp = 1  # below any valid n_min
    layer_2d.define_hyperlayer(pad_len=0.25, lmin=0.01)
    n_min, _ = layer_2d.n_bounds
    assert layer_2d.n_hyp >= n_min


def test_define_hyperlayer_clamps_n_hyp_down(layer_2d):
    layer_2d.n_hyp = 100  # above any valid n_max
    layer_2d.define_hyperlayer(pad_len=0.25, lmin=0.01)
    _, n_max = layer_2d.n_bounds
    assert layer_2d.n_hyp <= n_max


def test_define_hyperlayer_valid_n_hyp_unchanged(layer_2d):
    layer_2d.n_hyp = 4
    layer_2d.define_hyperlayer(pad_len=0.25, lmin=0.01)
    n_min, n_max = layer_2d.n_bounds
    # n_hyp should remain within bounds
    assert n_min <= layer_2d.n_hyp <= n_max


# ---------------------------------------------------------------------------
# calc_hyp_geom_prop
# ---------------------------------------------------------------------------

def test_calc_hyp_geom_prop_2d_positive_area(layer_2d):
    layer_2d.define_hyperlayer(0.25, 0.01)
    layer_2d.calc_hyp_geom_prop()
    assert layer_2d.area > 0
    assert layer_2d.a_rat > 0
    assert layer_2d.f_Ah > 0
    assert layer_2d.perim_hyp > 0


def test_calc_hyp_geom_prop_2d_area_ratio_above_one(layer_2d):
    # The padded area must be larger than the original domain area
    layer_2d.define_hyperlayer(0.25, 0.01)
    layer_2d.calc_hyp_geom_prop()
    assert layer_2d.a_rat > 1.0


def test_calc_hyp_geom_prop_3d_positive_volume(layer_3d):
    layer_3d.define_hyperlayer(0.25, 0.01)
    layer_3d.calc_hyp_geom_prop()
    assert layer_3d.vol > 0
    assert layer_3d.v_rat > 0
    assert layer_3d.f_Vh > 0


def test_calc_hyp_geom_prop_3d_volume_ratio_above_one(layer_3d):
    layer_3d.define_hyperlayer(0.25, 0.01)
    layer_3d.calc_hyp_geom_prop()
    assert layer_3d.v_rat > 1.0
