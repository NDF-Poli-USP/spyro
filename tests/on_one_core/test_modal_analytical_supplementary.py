"""More unit tests for the Analytical Modal solver in spyro.solvers.modal.modal_ana_sol.

These tests verify the analytical modal solver in 2D and 3D, comparing the computed
frequency with expected values. Also, the tests also check the behavior of the solver
for boundary conditions of Dirichlet and Neumann.
"""

from pytest import fixture, raises
from firedrake import (Function, FunctionSpace, SpatialCoordinate,
                       UnitCubeMesh, UnitSquareMesh)
from numpy import all, arange, array, isfinite, pi, setdiff1d, sqrt
from numpy.testing import assert_almost_equal
from scipy.special import jn_zeros
from spyro.solvers.modal.modal_ana_sol import Modal_Analytical_Solver
from spyro.utils.error_management import type_firedrake_error


class TestModalAnalyticalSolver:
    """Test suite for Modal_Analytical_Solver class."""

    @ fixture
    def solver_2d(self):
        """Create a 2D solver instance."""
        return Modal_Analytical_Solver(dimension=2)

    @ fixture
    def solver_3d(self):
        """Create a 3D solver instance."""
        return Modal_Analytical_Solver(dimension=3)

    @ fixture
    def mesh_2d(self):
        """Create a 2D mesh."""
        return UnitSquareMesh(4, 4)

    @ fixture
    def mesh_3d(self):
        """Create a 3D mesh."""
        return UnitCubeMesh(4, 4, 4)

    @ fixture
    def V_2d(self, mesh_2d):
        """Create a 2D function space."""
        return FunctionSpace(mesh_2d, "KMV", 4)

    @ fixture
    def V_3d(self, mesh_3d):
        """Create a 3D function space."""
        return FunctionSpace(mesh_3d, "KMV", 3)

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

    def test_reg_geometry_hyp_cut_percent_variation(self, solver_2d, solver_3d):
        """Test _reg_geometry_hyp for different cut percentages."""
        cut_percents = [0.01, 0.3, 0.5, 0.7, 0.9, 0.99]

        prev_fr_ell = None
        prev_fr_rec = None

        for cut in cut_percents:
            pn, qn, fr_ell, fr_rec = solver_2d._reg_geometry_hyp(cut_plane_percent=cut)

            # Check fitted parameters are valid
            assert np.isfinite(pn) and pn > 0.
            assert np.isfinite(qn) and qn > 0.

            # As cut_percent increases (less truncation), fr_ell and fr_rec should increase
            if prev_fr_ell is not None:
                assert fr_ell > prev_fr_ell, \
                    f"fr_ell should increase with cut_percent: {prev_fr_ell} -> {fr_ell}"
                assert fr_rec > prev_fr_rec, \
                    f"fr_rec should increase with cut_percent: {prev_fr_rec} -> {fr_rec}"

            prev_fr_ell = fr_ell
            prev_fr_rec = fr_rec

        prev_fr_ell = None
        prev_fr_rec = None

        for cut in cut_percents:
            pn, qn, fr_ell, fr_rec = solver_3d._reg_geometry_hyp(cut_plane_percent=cut)

            # Check fitted parameters are valid
            assert np.isfinite(pn) and pn > 0.
            assert np.isfinite(qn) and qn > 0.

            # As cut_percent increases (less truncation), fr_ell and fr_rec should increase
            if prev_fr_ell is not None:
                assert fr_ell > prev_fr_ell, \
                    f"fr_ell should increase with cut_percent: {prev_fr_ell} -> {fr_ell}"
                assert fr_rec > prev_fr_rec, \
                    f"fr_rec should increase with cut_percent: {prev_fr_rec} -> {fr_rec}"

            prev_fr_ell = fr_ell
            prev_fr_rec = fr_rec

    def test_freq_factor_hyp_2d(self, solver_2d):
        """Test _freq_factor_hyp with Neumann BC for 2D."""
        # Test parameters
        hyp_degrees = [3, 5, 10, 20, 50, 100, 200, 1000]
        hyper_axes = (2., 3.)
        c_eq = 3.0

        for n_hyp in hyp_degrees:

            # Fundamental frequency factors
            f_ell_neu = solver_2d._freq_factor_ell(hyper_axes, bc="Neumann")
            f_rec_neu = solver_2d._freq_factor_rec(hyper_axes, bc="Neumann")
            f_ell_dir = solver_2d._freq_factor_ell(hyper_axes, bc="Dirichlet")
            f_rec_dir = solver_2d._freq_factor_rec(hyper_axes, bc="Dirichlet")
            fbc_dirichlet = (f_ell_dir / f_ell_neu, f_rec_dir / f_rec_neu)

            f_hyp_dir, c_reg_dir = solver_2d._freq_factor_hyp(n_hyp, f_rec_dir, f_ell_dir,
                                                              c_eq, bc="Dirichlet",
                                                              fbc_dirichlet=fbc_dirichlet)

            # Check results are positive and finite
            assert np.isfinite(f_hyp_dir) and f_hyp_dir > 0.
            assert np.isfinite(c_reg_dir) and c_reg_dir > 0.

            # f_hyp_dir should be between f_rec_dir and f_ell_dir
            assert f_rec_dir < f_hyp_dir < f_ell_dir

            f_hyp_neu, c_reg_neu = solver_2d._freq_factor_hyp(n_hyp, f_rec_neu, f_ell_neu,
                                                              c_eq, bc="Neumann")

            # Check results are positive and finite
            assert np.isfinite(f_hyp_neu) and f_hyp_neu > 0.
            assert np.isfinite(c_reg_neu) and c_reg_neu > 0.

            # f_hyp_neu should be between f_rec_neu and f_ell_neu
            assert f_rec_neu < f_hyp_neu < f_ell_neu

            # f_hyp_dir should greater than f_hyp_neu for the same n_hyp and c_eq
            assert f_rec_neu < f_hyp_neu < f_hyp_dir < f_ell_dir

    def test_freq_factor_hyp_3d(self, solver_3d):
        """Test _freq_factor_hyp with Neumann BC fpr 3D."""
        # Test parameters
        hyp_degrees = [3, 5, 10, 20, 50, 100, 200, 1000]
        hyper_axes = (2., 3., 1.)
        c_eq = 3.0
        for n_hyp in hyp_degrees:

            # Fundamental frequency factors
            f_ell_neu = solver_3d._freq_factor_ell(hyper_axes, bc="Neumann")
            f_rec_neu = solver_3d._freq_factor_rec(hyper_axes, bc="Neumann")
            f_ell_dir = solver_3d._freq_factor_ell(hyper_axes, bc="Dirichlet")
            f_rec_dir = solver_3d._freq_factor_rec(hyper_axes, bc="Dirichlet")
            fbc_dirichlet = (f_ell_dir / f_ell_neu, f_rec_dir / f_rec_neu)

            f_hyp_dir, c_reg_dir = solver_3d._freq_factor_hyp(n_hyp, f_rec_dir, f_ell_dir,
                                                              c_eq, bc="Dirichlet",
                                                              fbc_dirichlet=fbc_dirichlet)

            # Check results are positive and finite
            assert np.isfinite(f_hyp_dir) and f_hyp_dir > 0.
            assert np.isfinite(c_reg_dir) and c_reg_dir > 0.

            # f_hyp_dir should be between f_rec_dir and f_ell_dir
            assert f_rec_dir < f_hyp_dir < f_ell_dir

            f_hyp_neu, c_reg_neu = solver_3d._freq_factor_hyp(n_hyp, f_rec_neu, f_ell_neu,
                                                              c_eq, bc="Neumann")

            # Check results are positive and finite
            assert np.isfinite(f_hyp_neu) and f_hyp_neu > 0.
            assert np.isfinite(c_reg_neu) and c_reg_neu > 0.

            # f_hyp_neu should be between f_rec_neu and f_ell_neu
            assert f_rec_neu < f_hyp_neu < f_ell_neu

            # f_hyp_dir should greater than f_hyp_neu for the same n_hyp and c_eq
            assert f_rec_neu < f_hyp_neu < f_hyp_dir < f_ell_dir

    def test_dummy_load_static(self, solver_2d, V_2d, solver_3d, V_3d):
        """Test basic functionality of dummy_load_static."""
        dof_load = np.array([0, 5, 10])
        amplitude_load = np.array([1., -2., 3.])

        for dimension in [2, 3]:
            if dimension == 2:
                solver = solver_2d
                V = V_2d
            else:
                solver = solver_3d
                V = V_3d

            q_dummy, q_ref = solver.dummy_load_static(V, dof_load, amplitude_load)

            # Check that q_dummy is a Function
            assert type_firedrake_error("q_dummy", q_dummy, "Function")
            assert q_dummy.function_space() == V

            # Check values at specified DOFs
            assert np.all(q_dummy.dat.data[dof_load] == amplitude_load)

            # Check that other DOFs remain zero
            other_dofs = np.setdiff1d(np.arange(V.dim()), dof_load)
            assert sum(q_dummy.dat.data[other_dofs]) == 0.

            # q_ref should be None when V_ref is not provided
            assert q_ref is None

    def test_dummy_load_static_with_reference(self, solver_2d, V_2d, solver_3d, V_3d):
        """Test dummy_load_static with reference function space."""
        dof_load = np.array([0, 5, 10])
        amplitude_load = np.array([1., -2., 3.])

        for dimension in [2, 3]:
            if dimension == 2:
                solver = solver_2d
                V = V_2d
                mesh = V_2d.mesh()

            else:
                solver = solver_3d
                V = V_3d
                mesh = V_3d.mesh()

            # Create a different function space for reference
            V_ref = FunctionSpace(mesh, "CG", 3)

            q_dummy, q_ref = solver_2d.dummy_load_static(V, dof_load,
                                                         amplitude_load,
                                                         V_ref=V_ref)

            # q_dummy should be in V
            assert type_firedrake_error("q_dummy", q_dummy, "Function")
            assert q_dummy.function_space() == V

            # Check values at specified DOFs
            assert np.all(q_dummy.dat.data[dof_load] == amplitude_load)

            # Check that other DOFs remain zero
            other_dofs = np.setdiff1d(np.arange(V.dim()), dof_load)
            assert sum(q_dummy.dat.data[other_dofs]) == 0.

            # q_ref should be in V_ref
            assert type_firedrake_error("q_ref", q_ref, "Function")
            assert q_ref.function_space() == V_ref

    def test_c_equivalent_volume_homog_constant_2d(self, solver_2d, V_2d):
        """Test c_equivalent with volume homogenization for constant velocity in 2D."""
        # Create constant velocity
        c = Function(V_2d)
        c.assign(3.0)  # Constant velocity of 3.0

        # Compute equivalent velocity
        c_eq = solver_2d.c_equivalent(c, V_2d, type_homog="volume")

        # For constant velocity, c_eq should equal the constant value
        np.testing.assert_almost_equal(c_eq, 3.0, decimal=10)

    def test_c_equivalent_volume_homog_constant_3d(self, solver_3d, V_3d):
        """Test c_equivalent with volume homogenization for constant velocity in 3D."""
        # Create constant velocity
        c = Function(V_3d)
        c.assign(2.5)

        # Compute equivalent velocity
        c_eq = solver_3d.c_equivalent(c, V_3d, type_homog="volume")

        # For constant velocity, c_eq should equal the constant value
        np.testing.assert_almost_equal(c_eq, 2.5, decimal=10)

    def test_c_equivalent_volume_homog_variable_2d(self, solver_2d, V_2d):
        """Test c_equivalent with volume homogenization for variable velocity in 2D."""
        # Create variable velocity: c(x,y) = 1 + x + y
        x = SpatialCoordinate(V_2d.mesh())
        c = Function(V_2d)
        c.interpolate(1.0 + x[0] + x[1])

        # Compute equivalent velocity
        c_eq = solver_2d.c_equivalent(c, V_2d, type_homog="volume")

        # For volume homogenization, c_eq should be the average velocity
        # Expected average of (1 + x + y) over unit square = 1 + 0.5 + 0.5 = 2.0
        np.testing.assert_almost_equal(c_eq, 2.0, decimal=6)

    def test_c_equivalent_volume_homog_variable_3d(self, solver_3d, V_3d):
        """Test c_equivalent with volume homogenization for variable velocity in 3D."""
        # Create variable velocity: c(x,y,z) = 2 + x + y + z
        x = SpatialCoordinate(V_3d.mesh())
        c = Function(V_3d)
        c.interpolate(2.0 + x[0] + x[1] + x[2])

        # Compute equivalent velocity
        c_eq = solver_3d.c_equivalent(c, V_3d, type_homog="volume")

        # Expected average of (2 + x + y + z) over unit cube = 2 + 0.5 + 0.5 + 0.5 = 3.5
        np.testing.assert_almost_equal(c_eq, 3.5, decimal=6)

    def test_c_equivalent_energy_homog_constant_2d(self, solver_2d, V_2d):
        """Test c_equivalent with energy homogenization for constant velocity in 2D."""
        # Create constant velocity
        c = Function(V_2d).assign(3.)

        # Create a static load
        dof_load = np.array([0])
        amplitude_load = np.array([1.])
        q_dummy = solver_2d.dummy_load_static(V_2d, dof_load, amplitude_load)[0]

        # Compute equivalent velocity
        c_eq = solver_2d.c_equivalent(c, V_2d, type_homog="energy",
                                      static_load_for_ceq=q_dummy)

        # For constant velocity, c_eq should be equal to the constant value
        np.testing.assert_almost_equal(c_eq, 3., decimal=6)

    def test_c_equivalent_energy_homog_constant_3d(self, solver_3d, V_3d):
        """Test c_equivalent with energy homogenization for constant velocity in 3D."""
        # Create constant velocity
        c = Function(V_3d).assign(3.)

        # Create a static load (point load at center)
        dof_load = np.array([0])
        amplitude_load = np.array([1.])
        q_dummy = solver_3d.dummy_load_static(V_3d, dof_load, amplitude_load)[0]

        # Compute equivalent velocity
        c_eq = solver_3d.c_equivalent(c, V_3d, type_homog="energy",
                                      static_load_for_ceq=q_dummy)

        # For constant velocity, c_eq should be equal to the constant value
        np.testing.assert_almost_equal(c_eq, 3., decimal=6)

    def test_c_equivalent_invalid_type_homog(self, solver_2d, V_2d):
        """Test c_equivalent with invalid type_homog."""
        c = Function(V_2d)
        c.assign(3.)

        # Should raise ValueError for invalid type_homog
        with raises(ValueError):
            solver_2d.c_equivalent(c, V_2d, type_homog="invalid")
