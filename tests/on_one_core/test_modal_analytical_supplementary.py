import pytest
import numpy as np
from scipy.special import jn_zeros
from spyro.solvers.modal.modal_ana_sol import Modal_Analytical_Solver


class TestModalAnalyticalSolver:
    """Test suite for Modal_Analytical_Solver class."""

    @pytest.fixture
    def solver_2d(self):
        """Create a 2D solver instance."""
        return Modal_Analytical_Solver(dimension=2)

    @pytest.fixture
    def solver_3d(self):
        """Create a 3D solver instance."""
        return Modal_Analytical_Solver(dimension=3)

    def test_freq_factor_rec_dirichlet_2d(self, solver_2d):
        """Test _freq_factor_rec with Dirichlet BC for 2D rectangle."""
        # Test cases: (hyper_axes, expected_result)
        factor = np.pi / 2
        test_cases = [
            # Rectangle
            ((1., 5.), factor * np.sqrt(1/1.**2 + 1/5.**2)),
            # Square
            ((2., 2.), factor * np.sqrt(2 * (1/2.**2))),
            # Large dimensions
            ((2e3, 1e3), factor * np.sqrt(1 / 2e3**2 + 1 / 1e3**2)),
            # Small dimensions
            ((1e-3, 2e-3), factor * np.sqrt(1 / 1e-3**2 + 1 / 2e-3**2)),
            # Dissimilar dimensions
            ((1e-3, 1e3), factor * np.sqrt(1/1e-3**2 + 1/1e3**2)),
            # Irrational values
            ((1./3., 1./7.), factor * np.sqrt(1/(1./3.)**2 + 1/(1./7.)**2)),
        ]

        for hyper_axes, expected in test_cases:
            f_rec = solver_2d._freq_factor_rec(hyper_axes, bc="Dirichlet")
            np.testing.assert_almost_equal(f_rec, expected, decimal=10,
                                           err_msg=f"Failed for hyper_axes={hyper_axes}")

    def test_freq_factor_rec_dirichlet_3d(self, solver_3d):
        """Test _freq_factor_rec with Dirichlet BC for 3D prism."""
        # Test cases: (hyper_axes, expected_result)
        factor = np.pi / 2
        test_cases = [
            # Prism
            ((2., 3., 4.), factor * np.sqrt(1/2.**2 + 1/3.**2 + 1/4.**2)),
            # Cube
            ((2., 2., 2.), factor * np.sqrt(3 * (1/2.**2))),
            # Long prism
            ((1.0, 2.0, 10.), factor * np.sqrt(1/1.**2 + 1/2.**2 + 1/10.**2)),
            # Short prism
            ((10., 10., 1.), factor * np.sqrt(2 * (1/10.**2) + 1/1.**2)),
            # Large dimensions
            ((2e3, 1e3, 4e3), factor * np.sqrt(1 / 2e3**2 + 1 / 1e3**2 + 1 / 4e3**2)),
            # Small dimensions
            ((1e-3, 2e-3, 1e-4), factor * np.sqrt(1 / 1e-3**2 + 1 / 2e-3**2 + 1 / 1e-4**2)),
            # Dissimilar dimensions
            ((1e-3, 1e3, 5), factor * np.sqrt(1/1e-3**2 + 1/1e3**2 + 1/5**2)),
            # Irrational values
            ((1./3., 1./7., 1/9.), factor * np.sqrt(1/(1./3.)**2 + 1/(1./7.)**2 + 1/(1/9.)**2))
        ]

        for hyper_axes, expected in test_cases:
            f_rec = solver_3d._freq_factor_rec(hyper_axes, bc="Dirichlet")
            np.testing.assert_almost_equal(f_rec, expected, decimal=10,
                                           err_msg=f"Failed for hyper_axes={hyper_axes}")

    def test_freq_factor_rec_dirichlet_vs_neumann(self, solver_2d):
        """Compare Dirichlet and Neumann BC results."""
        hyper_axes = (2., 3.)
        dir_f_rec = solver_2d._freq_factor_rec(hyper_axes, bc="Dirichlet")
        neu_f_rec = solver_2d._freq_factor_rec(hyper_axes, bc="Neumann")

        # Dirichlet should be larger than Neumann (since it sums squares)
        assert dir_f_rec > neu_f_rec

        # Verify the relationship for square
        square_axes = (2., 2.)
        dir_square = solver_2d._freq_factor_rec(square_axes, bc="Dirichlet")
        neu_square = solver_2d._freq_factor_rec(square_axes, bc="Neumann")

        # For a square: Dirichlet = pi/2 * sqrt(2)/a, Neumann = pi/(2a)
        # So Dirichlet/Neumann = sqrt(2)
        np.testing.assert_almost_equal(dir_square / neu_square, np.sqrt(2), decimal=10)

    def test_freq_factor_ell_dirichlet_2d_circular(self, solver_2d):
        """Test _freq_factor_ell with Dirichlet BC for 2D circular membrane.

        For circular membrane with Dirichlet BC: f_ell = J01 / a
        where J01 is the first zero of Bessel function J0.
        """
        J01 = jn_zeros(0, 1)[0]  # First zero of J0

        test_cases = [
            # Radius a
            (1.0, J01 / 1.0),
            (2.0, J01 / 2.0),
            (3.0, J01 / 3.0),
            (0.5, J01 / 0.5),
            (10.0, J01 / 10.0),
            (np.pi, J01 / np.pi),
        ]

        for radius, expected in test_cases:
            hyper_axes = (radius, radius)  # Equal axes = circle
            f_cir = solver_2d._freq_factor_ell(hyper_axes, bc="Dirichlet",
                                               all_axes_equal=True)
            np.testing.assert_almost_equal(f_cir, expected, decimal=10,
                                           err_msg=f"Failed for radius={radius}")

    def test_freq_factor_ell_dirichlet_2d_elliptical(self, solver_2d):
        """Test _freq_factor_ell with Dirichlet BC for 2D elliptical membrane.

        For Dirichlet BC, the frequency factor is computed using Modified Mathieu
        function: (2/f0) * sqrt(M01) where M01 is the root of the Modified Mathieu
        function and f0 is the eccentricity. These test cases use the actual
        implementation and expected values are computed by the method itself.
        Here, consistency and reasonable values are tested rather than exact numerical
        values, since the Mathieu function roots are not trivial to compute analytically.
        """
        test_cases = [
            # (semi-axes a, b) with a > b (eccentric)
            (2., 1.),     # Ellipse with moderate eccentricity
            (3., 1.),     # Ellipse with high eccentricity
            (2., 1.5),    # Ellipse with low eccentricity
            (5., 2.),     # Ellipse with high eccentricity
            (1., 0.5),    # Small ellipse
            (1e-6, 5e-7),  # Very small ellipse
            (1e6, 5e5),   # Very large ellipse
            (50., 1.),    # Very eccentric ellipse
        ]

        for hyper_axes in test_cases:
            # Get result and verify it's a positive finite number
            f_ell = solver_2d._freq_factor_ell(hyper_axes, bc="Dirichlet")
            assert np.isfinite(f_ell) and f_ell > 0.

            # Compare with rectangular case
            a, b = hyper_axes
            f_rec = solver_2d._freq_factor_rec((a, b), bc="Dirichlet")
            # Elliptical frequency factor should be greater than rectangular
            # because ellipse has smaller area than rectangle with same semi-axes
            assert f_ell > f_rec, f"Elliptical f_ell={f_ell} " \
                f"should be > rectangular f_rec={f_rec}"

    def test_freq_factor_ell_dirichlet_2d_circle_vs_ellipse(self, solver_2d):
        """Compare circular and elliptical cases with same semi-axis."""
        # Circle with radius a
        a = 2.
        circle_axes = (a, a)
        f_cir = solver_2d._freq_factor_ell(circle_axes, bc="Dirichlet",
                                           all_axes_equal=True)

        # Ellipse with semi-axes (a, b) where b < a
        b = 1.5
        ellipse_axes = (a, b)
        f_ell = solver_2d._freq_factor_ell(ellipse_axes, bc="Dirichlet")

        # Circle should have smaller frequency factor than ellipse because
        # ellipse has smaller area than circle for the given semi-axes
        assert f_ell > f_cir, f"Elliptical f_ell={f_ell} " \
            f"should be > circular f_cir={f_cir}"

    def test_freq_factor_ell_dirichlet_3d_spherical(self, solver_3d):
        """Test _freq_factor_ell with Dirichlet BC for 3D sphere.

        For spherical model with Dirichlet BC, the frequency factor uses roots
        of spherical Bessel function. The method uses fsolve to find the root.
        """

        # Radius a
        test_cases = [1., 2., 3., 0.5, 10.]

        for radius in test_cases:
            hyper_axes = (radius, radius, radius)  # Equal axes = sphere
            f_cir = solver_3d._freq_factor_ell(hyper_axes, bc="Dirichlet",
                                               all_axes_equal=True)

            # Verify f_cir is positive and finite
            assert np.isfinite(f_cir) and f_cir > 0.

            # For sphere, frequency factor should scale as 1/a
            if radius == 1.:
                f_cir_ref = f_cir
            else:
                # For same problem, f_ell * a should be constant
                ratio = f_cir * radius / f_cir_ref
                np.testing.assert_almost_equal(
                    ratio, 1., decimal=10, err_msg=f"Scaling failed for radius={radius}")

    def test_freq_factor_ell_dirichlet_3d_ellipsoidal(self, solver_3d):
        """Test _freq_factor_ell with Dirichlet BC for 3D ellipsoid.

        For Dirichlet BC on ellipsoid, the method uses a combination of
        Modified Mathieu functions for each pair of semi-axes.
        """
        test_cases = [
            # (semi-axes a, b, c)
            (3.0, 2.0, 1.0),   # Triaxial ellipsoid
            (2.0, 1.5, 1.0),   # Prolate spheroid
            (2.0, 2.0, 1.0),   # Oblate spheroid
            (4.0, 3.0, 2.0),   # Large ellipsoid
            (0.5, 0.3, 0.2),   # Small ellipsoid
        ]

        for hyper_axes in test_cases:
            f_ell = solver_3d._freq_factor_ell(hyper_axes, bc="Dirichlet")

            # Verify f_ell is positive and finite
            assert np.isfinite(f_ell) and f_ell > 0.

            # Compare with prism case. Ellipsoid has smaller volume than prism for the
            # same semi-axes then the frequency factor should be larger for ellipsoid.
            f_rec = solver_3d._freq_factor_rec(hyper_axes, bc="Dirichlet")
            assert f_ell > f_rec, f"Ellipsoidal f_ell={f_ell} " \
                f"should be > rectangular f_rec={f_rec}"

    def test_freq_factor_ell_dirichlet_consistency(self, solver_2d, solver_3d):
        """Test consistency of _freq_factor_ell with Dirichlet BC."""

        # Test 2D: As ellipse becomes more circular, result should approach circle case
        a = 2.0
        b_values = [a * factor for factor in [0.9, 0.95, 0.99, 0.999]]
        f_cir = solver_2d._freq_factor_ell((a, a), bc="Dirichlet", all_axes_equal=True)

        for b in b_values:
            f_ell = solver_2d._freq_factor_ell((a, b), bc="Dirichlet")
            # As b approaches a, ellipse result should approach circle result
            diff = abs(f_ell - f_cir)
            assert diff < 0.1, f"Elliptical f_ell={f_ell} " \
                f"should approach circular f_cir={f_cir}"

        # Test 3D: As ellipsoid becomes more spherical, result should approach sphere case
        a = 2.0
        c_values = [a * factor for factor in [0.9, 0.95, 0.99, 0.999]]
        f_sph = solver_3d._freq_factor_ell((a, a, a), bc="Dirichlet", all_axes_equal=True)
        for c in c_values:
            f_ell = solver_3d._freq_factor_ell((a, a, c), bc="Dirichlet")
            # As c approaches a, ellipsoid result should approach sphere result
            diff = abs(f_ell - f_sph)
            assert diff < 0.1, f"Ellipsoidal f_ell={f_ell} " \
                f"should approach spherical f_sph={f_sph}"

    def test_freq_factor_ell_dirichlet_neumann_comparison(self, solver_2d):
        """Compare Dirichlet and Neumann BC for elliptical case."""
        hyper_axes = (2.0, 1.0)

        dir_f_ell = solver_2d._freq_factor_ell(hyper_axes, bc="Dirichlet")
        neu_f_ell = solver_2d._freq_factor_ell(hyper_axes, bc="Neumann")

        # For the same geometry, Dirichlet should generally be larger than Neumann
        assert dir_f_ell > neu_f_ell, f"Dirichlet f_ell={dir_f_ell} " \
            f"should be > Neumann f_ell={neu_f_ell}"

    def test_freq_factor_ell_dirichlet_comparison_with_rec(self, solver_2d):
        """Compare elliptical and rectangular frequency factors for Dirichlet BC."""

        a, b = 2.0, 1.0

        # Test for various aspect ratios
        aspect_ratios = [1.1, 1.5, 2.0, 3.0, 5.0, 10.0]

        for ratio in aspect_ratios:
            b = a / ratio
            hyper_axes = (a, b)

            f_ell = solver_2d._freq_factor_ell(hyper_axes, bc="Dirichlet")
            f_rec = solver_2d._freq_factor_rec(hyper_axes, bc="Dirichlet")

            # Ellipse should have higher frequency factor
            assert f_ell > f_rec, f"Elliptical f_ell={f_ell} " \
                f"should be > rectangular f_rec={f_rec} for ratio={ratio}"

    def test_reg_geometry_hyp_no_cut(self, solver_2d, solver_3d):
        """Test _reg_geometry_hyp for 2D  and 3D with no cut (cut_plane_percent=1.)."""
        pn, qn, fr_ell, fr_rec = solver_2d._reg_geometry_hyp(cut_plane_percent=1.)

        # Check that fitted parameters are positive and finite
        assert np.isfinite(pn) and pn > 0.
        assert np.isfinite(qn) and qn > 0.

        # For no cut, fr_ell and fr_rec should be 1.0 (no truncation)
        np.testing.assert_almost_equal(fr_ell, 1., decimal=10)
        np.testing.assert_almost_equal(fr_rec, 1., decimal=10)

        pn, qn, fr_ell, fr_rec = solver_3d._reg_geometry_hyp(cut_plane_percent=1.0)

        # Check that fitted parameters are positive and finite
        assert np.isfinite(pn) and pn > 0.
        assert np.isfinite(qn) and qn > 0.

        # For no cut, fr_ell and fr_rec should be 1.0 (no truncation)
        np.testing.assert_almost_equal(fr_ell, 1., decimal=10)
        np.testing.assert_almost_equal(fr_rec, 1., decimal=10)
