"""Unit tests for the layer size criterion implemented in spyro.abc.laylen.

Thsese tests validate the correctness of the functions responsible for calculating
the size of absorbing layers in numerical simulations with ABCs (HABCs and PML),
ensuring that they meet expected analytical results and known values from documentation.
"""

from unittest import TestCase
from numpy import ceil
from spyro.abc.lay_len import calc_size_lay, calc_zero, f_layer, loop_roots, roundFL


class TestAbsorbingLayer(TestCase):
    """Test cases for absorbing layer sizing methods."""

    def setUp(self):
        """Set up common test parameters."""
        self.c = 1.5                         # Wave speed (km/s)
        self.lref = 1.2                      # Reference length (km)
        self.lmin = 0.1                      # Minimum mesh length (km)
        self.tol = 1e-3                      # Tolerance for root finding
        self.max_roots = 5                   # Maximum number of roots to find
        self.vibration_mode = 1              # Vibration mode
        self.damping_ratio = 0.999           # Damping ratio
        self.z_par = self.c / self.lref      # Frequency parameter (1/s) (z = c / l)

        # Dataset 1 for testing known values from documentation
        self.fref = 5.0                      # Reference frequency (Hz)
        self.a_par = self.z_par / self.fref  # Adimensional parameter (a = z / f)
        self.FL_s1 = [0.1917, 0.2682, 0.2981, 0.4130, 0.4244]  # Expected parameter size

        # Dataset 2 for testing known values from documentation
        self.fref_s2 = 2.25  # Reference frequency (Hz)
        self.a_par_s2 = self.z_par / self.fref_s2  # Adimensional parameter (a = z / f)
        self.FL_s2 = [0.4259, 0.5959, 0.6624, 0.9179, 0.9431]

    def test_f_layer_known_values(self):
        """Test f_layer against known analytical results for parameter size."""
        a_par = self.a_par
        vibration_mode = self.vibration_mode
        damping_ratio = self.damping_ratio
        root_values = self.FL_s1  # Cases from documentation

        # Test FL function zeros (CR - RF should be near 0 at root)
        for x_root in root_values:
            fl_value = f_layer(x_root, a_par, vibration_mode, damping_ratio, "FL")
            # Should be close to zero at the root
            self.assertAlmostEqual(fl_value, 0.0, places=3)

    def test_f_layer_for_reflection_coefficient(self):
        """Test f_layer against known analytical results for reflection coefficient."""
        a_par = self.a_par
        vibration_mode = self.vibration_mode
        damping_ratio = self.damping_ratio

        # Test  only (CR)
        x = 0.1917  # F_L1 from documentation
        cr_value = f_layer(x, a_par, vibration_mode, damping_ratio, 'CR')

        # CR should be between 0 and 1 (it's a reflection coefficient)
        self.assertGreater(cr_value, 0.0)
        self.assertLess(cr_value, 1.0)

        # Calculate expected CR manually for comparison
        denominator = damping_ratio ** 2 + (4 * x / (vibration_mode * a_par)) ** 2
        expected_cr = damping_ratio ** 2 / denominator
        self.assertAlmostEqual(cr_value, expected_cr, places=6)

        # For x=0, CR should be 1 (perfect reflection)
        x = 0.
        cr_zero = f_layer(x, a_par, vibration_mode, damping_ratio, 'CR')
        self.assertAlmostEqual(cr_zero, 1.0)

        # As x increases, CR should decrease (more absorption)
        x = 100.
        cr_large = f_layer(x, a_par, vibration_mode, damping_ratio, 'CR')
        self.assertLess(cr_large, cr_value)
        self.assertAlmostEqual(cr_large, 0.0, places=6)

        # With damping_ratio -> 0, CR - >0 regardless of x (no energy absorbed)
        x = 0.01
        cr_small = f_layer(x, a_par, vibration_mode, 0.001, 'CR')
        self.assertAlmostEqual(cr_small, 0.0, places=3)

        # Test that CR decreases as damping ratio decreases
        x = 0.2
        cr_s1 = f_layer(x, a_par, vibration_mode, 0.5, 'CR')
        cr_s2 = f_layer(x, a_par, vibration_mode, damping_ratio, 'CR')
        self.assertGreater(cr_s2, cr_s1)

        # Test that CR increases as vibration mode increases
        cr_mode1 = f_layer(x, a_par, vibration_mode, damping_ratio, 'CR')
        cr_mode2 = f_layer(x, a_par, 2, damping_ratio, 'CR')
        self.assertNotEqual(cr_mode2, cr_mode1)

        # Test specific expected values (x=0.12, a=0.25, m=1, s=0.999)
        denominator = damping_ratio ** 2 + (4 * x / (vibration_mode * a_par)) ** 2
        expected_cr = damping_ratio ** 2 / denominator
        self.assertAlmostEqual(f_layer(x, a_par, vibration_mode,
                                       damping_ratio, 'CR'), expected_cr)

    def test_calc_zero_convergence(self):
        """Test calc_zero function for root finding."""
        a_par = self.a_par
        tol = self.tol

        # First root
        x1 = calc_zero(0.1, a_par, tol, nz=1)
        self.assertAlmostEqual(x1, self.FL_s1[0], places=3)

        # Second root
        x2 = calc_zero(0.2, a_par, tol, nz=2)
        self.assertAlmostEqual(x2, self.FL_s1[1], places=3)

        # Test tolerance effects
        tol_fine = 1e-6
        x_fine = calc_zero(0.1, a_par, tol_fine, nz=1)
        self.assertAlmostEqual(x_fine, self.FL_s1[0], places=4)

    def test_loop_roots(self):
        """Test loop_roots function for multiple roots."""
        a_par = self.a_par
        lmin = self.lmin
        lref = self.lref
        tol = self.tol
        max_roots = self.max_roots

        FLpos = loop_roots(a_par, lmin, lref, max_roots, tol_rel=tol,
                           show_ig=True, monitor=True)

        # Check number of roots found
        self.assertEqual(len(FLpos), max_roots)

        # Check expected values from documentation
        expected_roots = self.FL_s1
        for i, (found, expected) in enumerate(zip(FLpos, expected_roots)):
            self.assertAlmostEqual(found, expected, places=4, msg=f"Root {i+1} mismatch")

        # Test with different tolerance
        FLpos_tight = loop_roots(a_par, lmin, lref, max_roots, tol_rel=1e-5,
                                 show_ig=True, monitor=True)
        for i, (found, expected) in enumerate(zip(FLpos_tight, expected_roots)):
            self.assertAlmostEqual(found, expected, places=4, msg=f"Root {i+1} mismatch")

    def test_roundFL(self):
        """Test roundFL function for mesh-based adjustments."""
        lmin = self.lmin
        lref = self.lref
        FL_original = self.FL_s1[0]  # Use first root for testing

        FL_rounded, pad_len, ele_pad = roundFL(lmin, lref, FL_original)

        # Check that rounded factor is larger (ceil operation)
        self.assertGreaterEqual(FL_rounded, FL_original)

        # Check that pad_len is multiple of lmin
        self.assertAlmostEqual(int(round(pad_len / lmin, 15)), ele_pad)

        # Check specific values
        expected_ele = int(ceil(FL_original * lref / lmin))
        self.assertEqual(ele_pad, expected_ele)
        expected_factor = (lmin / lref) * expected_ele
        self.assertAlmostEqual(FL_rounded, expected_factor)

    def test_calc_size_lay_basic(self):
        """Test calc_size_lay function for basic functionality."""
        fref = self.fref
        z_par = self.z_par
        lmin = self.lmin
        lref = self.lref
        a_par = self.a_par
        tol = self.tol
        max_roots = self.max_roots

        factor, pad_len, ele_pad, d_norm, a_par, FLpos = calc_size_lay(
            fref, z_par, lmin, lref, nz=max_roots, tol_rel=tol,
            layer_based_on_mesh=False, monitor=True)

        # Check outputs
        self.assertIsInstance(factor, float)
        self.assertIsInstance(pad_len, float)
        self.assertIsInstance(ele_pad, int)
        self.assertIsInstance(d_norm, float)
        self.assertIsInstance(a_par, float)
        self.assertIsInstance(FLpos, list)
        self.assertGreater(pad_len, lmin)
        self.assertGreater(ele_pad, 1)
        self.assertLess(d_norm, 1.0)
        self.assertGreater(d_norm, 0.0)

        # Check consistency
        self.assertAlmostEqual(pad_len, factor * lref)
        self.assertAlmostEqual(a_par, z_par / fref)

        # Check expected values from documentation
        expected_roots = self.FL_s1
        for i, (found, expected) in enumerate(zip(FLpos, expected_roots)):
            self.assertAlmostEqual(found, expected, places=4, msg=f"Root {i+1} mismatch")

    def test_calc_size_lay_with_mesh_adjustment(self):
        """Test calc_size_lay with mesh-based adjustment."""
        fref = self.fref
        z_par = self.z_par
        lmin = self.lmin
        lref = self.lref
        tol = self.tol
        max_roots = self.max_roots

        factor, pad_len, ele_pad, d_norm, a_par, FLpos = calc_size_lay(
            fref, z_par, lmin, lref, nz=max_roots, tol_rel=tol,
            layer_based_on_mesh=True, monitor=True)

        # Check that layer size is a multiple of lmin
        self.assertEqual(pad_len, round(ele_pad * lmin, 15))

        # Factor should be adjusted to ensure integer elements
        expected_factor = (lmin / lref) * ceil(lref * self.FL_s1[0] / lmin)
        self.assertAlmostEqual(factor, expected_factor, places=4)

    def test_calc_size_lay_different_frequencies(self):
        """Test calc_size_lay with different frequency values."""
        z_par = self.z_par
        lmin = self.lmin
        lref = self.lref
        tol = self.tol
        max_roots = self.max_roots

        # Test at different frequencies (a_par values)
        test_cases = [(self.fref, self.a_par, self.FL_s1),
                      (self.fref_s2, self.a_par_s2, self.FL_s2)]

        for fref, a_par_expected, roots_expected in test_cases:
            factor, pad_len, ele_pad, d_norm, a_par, FLpos = calc_size_lay(
                fref, z_par, lmin, lref, nz=max_roots, tol_rel=tol,
                layer_based_on_mesh=True, monitor=True)

            self.assertAlmostEqual(a_par, a_par_expected, places=4)
            for i, expected_root in enumerate(roots_expected):
                self.assertAlmostEqual(FLpos[i], expected_root,
                                       places=4, msg=f"Root {i+1} mismatch")
