"""Unit tests for coeff_of_determination function in spyro.utils.stats_tools."""

from pytest import approx, raises
from numpy import array, random
from spyro.utils.stats_tools import coeff_of_determination


def test_perfect_prediction():
    """Test when predictions are perfect (R² = 1, adjusted R² = 1)"""
    y_true = array([1, 2, 3, 4, 5])
    y_pred = array([1, 2, 3, 4, 5])
    p = 1

    result = coeff_of_determination(y_true, y_pred, p)

    # Perfect fit should give adjusted R² = 1
    assert result == approx(1.0, rel=1e-6)


def test_mean_prediction():
    """Test when predictions are the mean of true values (R² = 0)"""
    y_true = array([1, 2, 3, 4, 5])
    y_pred = array([3, 3, 3, 3, 3])  # Mean is 3
    p = 1

    result = coeff_of_determination(y_true, y_pred, p)

    # R² = 0, so adjusted R² should be negative for p >= 1 with small n
    # 1 - (1-0)*(5-1)/(5-1-1) = 1 - 4/3 = -0.333333...
    assert result == approx(-0.3333333333333333, rel=1e-6)


def test_numpy_array_input():
    """Test that function works with numpy array inputs"""
    y_true = array([1.5, 2.5, 3.5, 4.5])
    y_pred = array([1.4, 2.6, 3.4, 4.6])
    p = 1

    result = coeff_of_determination(y_true, y_pred, p)

    assert isinstance(result, float)
    assert result < 1.0


def test_multiple_predictors():
    """Test with multiple predictors (p > 1)"""
    y_true = array([2, 4, 6, 8, 10])
    y_pred = array([1, 5, 5, 9, 11])
    p = 2

    result = coeff_of_determination(y_true, y_pred, p)

    # Manual calculation verification
    # SS_res = (2-1)^2 + (4-5)^2 + (6-5)^2 + (8-9)^2 + (10-11)^2 = 1+1+1+1+1 = 5
    # SS_tot = (2-6)^2 + (4-6)^2 + (6-6)^2 + (8-6)^2 + (10-6)^2 = 16+4+0+4+16 = 40
    # R² = 1 - 5/40 = 0.875
    # n=5, p=2, n-p-1=2
    # adj R² = 1 - (1-0.875)*(4/2) = 1 - 0.125*2 = 0.75
    assert result == approx(0.75, rel=1e-6)


def test_small_dataset():
    """Test with very small dataset (n = p + 1)"""
    y_true = array([10, 20, 30])
    y_pred = array([12, 18, 31])
    p = 1  # n - p - 1 = 3 - 1 - 1 = 1 > 0

    result = coeff_of_determination(y_true, y_pred, p)

    # Should calculate adjusted R² normally
    assert isinstance(result, float)


def test_negative_r2():
    """Test when model is worse than predicting the mean (R² < 0)"""
    y_true = array([1, 2, 3, 4, 5])
    y_pred = array([10, 20, 30, 40, 50])  # Very bad predictions
    p = 1

    result = coeff_of_determination(y_true, y_pred, p)

    # Should be negative (worse than mean prediction)
    assert result < 0


def test_edge_case_single_observation():
    """Test with single observation"""
    y_true = array([42])
    y_pred = array([42])
    p = 0  # n - p - 1 = 1 - 0 - 1 = 0

    result = coeff_of_determination(y_true, y_pred, p)

    # When n-p-1 <= 0, should return unadjusted R²
    assert result == approx(1.0, rel=1e-6)


def test_adjusted_vs_unadjusted():
    """Test that adjusted R² is <= unadjusted R² for given p."""
    y_true = array([1, 3, 5, 7, 9, 11])
    y_pred = array([2, 3, 6, 6, 10, 12])

    # Calculate both adjusted and unadjusted
    p = 2
    result_adj = coeff_of_determination(y_true, y_pred, p)

    # Calculate unadjusted for comparison
    p = 5
    r2_unadj = coeff_of_determination(y_true, y_pred, p)

    # Adjusted should be <= unadjusted for p >= n - 1 (n = 6)
    assert result_adj <= r2_unadj


def test_float_precision():
    """Test floating point precision with non-integer values"""
    y_true = array([1.1, 2.2, 3.3, 4.4, 5.5])
    y_pred = array([1.15, 2.18, 3.32, 4.41, 5.49])
    p = 1

    result = coeff_of_determination(y_true, y_pred, p)

    # Should be close to expected value
    assert isinstance(result, float)
    assert 0.99 < result < 1.0


def test_zero_predictors():
    """Test with zero predictors"""
    y_true = array([10, 20, 30, 40])
    y_pred = array([15, 18, 35, 38])
    p = 0  # n - p - 1 = 4 - 0 - 1 = 3 > 0

    result = coeff_of_determination(y_true, y_pred, p)

    # Should use standard formula with p=0
    assert isinstance(result, float)


def test_large_dataset():
    """Test with larger dataset to ensure performance"""
    random.seed(42)
    n = 1000
    y_true = random.randn(n)
    y_pred = y_true + random.randn(n) * 0.1  # Add small noise
    p = 5

    result = coeff_of_determination(y_true, y_pred, p)

    # Should be close to 1 - noise_variance / total_variance
    assert 0.9 < result < 1.0


def test_equal_true_and_pred_but_not_perfect():
    """Test when y_true and y_pred are identical arrays"""
    y_true = array([5, 5, 5, 5])
    y_pred = array([5, 5, 5, 5])
    p = 1

    result = coeff_of_determination(y_true, y_pred, p)

    # SS_res = 0, so R² = 1, adjusted should be 1
    assert result == approx(1.0, rel=1e-6)


def test_all_same_true_values():
    """Test when all true values are identical"""
    y_true = array([7, 7, 7, 7, 7])
    y_pred = array([7, 8, 6, 7, 8])
    p = 1

    result = coeff_of_determination(y_true, y_pred, p)

    # SS_tot = 0, so R² = 1, adjusted should be 1 by convention
    assert result == approx(1.0, rel=1e-6)


def test_with_different_p_values():
    """Test that adjusted R² decreases with more predictors for same data"""
    y_true = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_pred = array([1.1, 2.2, 2.8, 4.1, 4.9, 6.0, 7.2, 7.8, 9.1, 9.9])

    r2_adj_p1 = coeff_of_determination(y_true, y_pred, p=1)
    r2_adj_p2 = coeff_of_determination(y_true, y_pred, p=2)
    r2_adj_p3 = coeff_of_determination(y_true, y_pred, p=3)

    # Adding more predictors should decrease adjusted R² if they don't help
    assert r2_adj_p1 >= r2_adj_p2 >= r2_adj_p3


def test_type_checking_y_true():
    """Test that TypeError is raised when y_true is not a numpy array"""
    y_true = [1, 2, 3, 4, 5]  # list, not numpy array
    y_pred = array([1, 2, 3, 4, 5])
    p = 1

    with raises(TypeError, match="'y_true' must be a array"):
        coeff_of_determination(y_true, y_pred, p)


def test_type_checking_y_pred():
    """Test that TypeError is raised when y_pred is not a numpy array"""
    y_true = array([1, 2, 3, 4, 5])
    y_pred = [1, 2, 3, 4, 5]  # list, not numpy array
    p = 1

    with raises(TypeError, match="'y_pred' must be a array"):
        coeff_of_determination(y_true, y_pred, p)


def test_length_mismatch():
    """Test length mismatch when y_true is longer than y_pred"""
    y_true = array([1, 2, 3, 4, 5])
    y_pred = array([1, 2, 3, 4])  # Different length
    p = 1

    with raises(ValueError, match="Length mismatch: 'y_true' has 5 elements"):
        coeff_of_determination(y_true, y_pred, p)


def test_length_mismatch_reversed():
    """Test length mismatch when y_pred is longer than y_true"""
    y_true = array([1, 2, 3])
    y_pred = array([1, 2, 3, 4, 5])
    p = 1

    with raises(ValueError, match="Length mismatch: 'y_true' has 3 elements"):
        coeff_of_determination(y_true, y_pred, p)


def test_p_is_None():
    """Test that ValueError is raised when p is negative"""
    y_true = array([1, 2, 3, 4, 5])
    y_pred = array([1, 2, 3, 4, 5])
    p = None

    with raises(TypeError, match="'p' must be a integer number"):
        coeff_of_determination(y_true, y_pred, p)
