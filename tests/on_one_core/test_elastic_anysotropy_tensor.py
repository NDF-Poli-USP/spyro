import pytest
from numpy import allclose, any, all, isnan, isinf
from firedrake import Constant, Function, UnitCubeMesh, FunctionSpace
from ufl.tensors import ListTensor
from spyro.solvers.elastic_wave.anisotropy import AnisotropyTensor


class MockPropISO:
    """Mock isotropic properties for testing."""

    def __init__(self, mesh, W, vP_value=2500.0, vS_value=1200.0, rho_value=2200.0):
        self.vP = Function(W).assign(Constant(vP_value))
        self.vS = Function(W).assign(Constant(vS_value))
        self.rho = Function(W).assign(Constant(rho_value))


class MockPropVTI:
    """Mock VTI properties for testing."""

    def __init__(self, mesh, W, epsilon=0.2, gamma=0.1, delta=0.15, anisotropy='weak'):
        self.epsilon = Function(W).assign(Constant(epsilon))
        self.gamma = Function(W).assign(Constant(gamma))
        self.delta = Function(W).assign(Constant(delta))
        self.anisotropy = anisotropy


class MockPropTTI:
    """Mock TTI properties for testing."""

    def __init__(self, mesh, W, theta=30.0, phi=0.0):
        self.theta = Function(W).assign(Constant(theta))
        self.phi = Function(W).assign(Constant(phi))


@pytest.fixture
def mesh():
    """Create a simple mesh for testing"""
    return UnitCubeMesh(2, 2, 2)


@pytest.fixture
def W(mesh):
    """Create function space for testing"""
    return FunctionSpace(mesh, "KMV", 3)


@pytest.fixture
def iso_props(mesh, W):
    """Fixture for isotropic properties"""
    return MockPropISO(mesh, W)


@pytest.fixture
def vti_props_exact(mesh, W):
    """Fixture for VTI properties with exact formulation"""
    return MockPropVTI(mesh, W, epsilon=0.2, gamma=0.1, delta=0.15, anisotropy='exact')


@pytest.fixture
def vti_props_weak(mesh, W):
    """Fixture for VTI properties with weak formulation"""
    return MockPropVTI(mesh, W, epsilon=0.1, gamma=0.05, delta=0.08, anisotropy='weak')


class TestAnisotropyTensor:
    """Test battery for AnisotropyTensor class."""
    @pytest.mark.parametrize("anisotropy_type", ['weak', 'exact'])
    def test_c_vti_tensor_isotropic_limit(self, mesh, W, iso_props, anisotropy_type):
        """Test that VTI tensor reduces to isotropic when parameters are zero."""
        vti_props_zero = MockPropVTI(mesh, W, epsilon=0.0, gamma=0.0,
                                     delta=0.0, anisotropy=anisotropy_type)

        C_vti = AnisotropyTensor.c_vti_tensor(iso_props, vti_props_zero)

        # Since C_vti is a UFL tensor, we need to interpolate to evaluate
        C11_expr = C_vti[0, 0]
        C33_expr = C_vti[2, 2]
        C44_expr = C_vti[3, 3]
        C66_expr = C_vti[5, 5]

        # Create functions to evaluate
        C11_func = Function(W).interpolate(C11_expr)
        C33_func = Function(W).interpolate(C33_expr)
        C44_func = Function(W).interpolate(C44_expr)
        C66_func = Function(W).interpolate(C66_expr)

        # For isotropic case, C11 should equal C33
        assert allclose(C11_func.dat.data, C33_func.dat.data, rtol=1e-10)
        # C44 should equal C66 for isotropic
        assert allclose(C44_func.dat.data, C66_func.dat.data, rtol=1e-10)

    @pytest.mark.parametrize("vti_props_type", ['vti_props_weak', 'vti_props_exact'])
    def test_c_vti_tensor_component_relationships(self, mesh, W, iso_props,
                                                  vti_props_type, request):
        """Test relationships between elastic tensor components for VTI."""
        vti_props_obj = request.getfixturevalue(vti_props_type)
        C_vti = AnisotropyTensor.c_vti_tensor(iso_props, vti_props_obj)

        # Test symmetry
        C12 = Function(W).interpolate(C_vti[0, 1])
        C21 = Function(W).interpolate(C_vti[1, 0])
        assert allclose(C12.dat.data, C21.dat.data, rtol=1e-10)
        C13 = Function(W).interpolate(C_vti[0, 2])
        C31 = Function(W).interpolate(C_vti[2, 0])
        assert allclose(C13.dat.data, C31.dat.data, rtol=1e-10)
        C23 = Function(W).interpolate(C_vti[1, 2])
        C32 = Function(W).interpolate(C_vti[2, 1])
        assert allclose(C23.dat.data, C32.dat.data, rtol=1e-10)

        # Test C12 = C11 - 2*C66
        C11 = Function(W).interpolate(C_vti[0, 0])
        C66 = Function(W).interpolate(C_vti[5, 5])
        C12_computed = C11.dat.data - 2 * C66.dat.data
        assert allclose(C12.dat.data, C12_computed, rtol=1e-10)

    @pytest.mark.parametrize("vti_props_type", ['vti_props_weak', 'vti_props_exact'])
    def test_c_vti_tensor_positive_definiteness(self, mesh, W, iso_props,
                                                vti_props_type, request):
        """Test that the elastic tensor is positive definite."""
        vti_props_obj = request.getfixturevalue(vti_props_type)
        C_vti = AnisotropyTensor.c_vti_tensor(iso_props, vti_props_obj)

        # Extract diagonal components
        C11 = Function(W).interpolate(C_vti[0, 0]).dat.data
        C22 = Function(W).interpolate(C_vti[1, 1]).dat.data
        C33 = Function(W).interpolate(C_vti[2, 2]).dat.data
        C44 = Function(W).interpolate(C_vti[3, 3]).dat.data
        C55 = Function(W).interpolate(C_vti[4, 4]).dat.data
        C66 = Function(W).interpolate(C_vti[5, 5]).dat.data

        # All diagonal components should be positive
        assert all(C11 > 0)
        assert all(C22 > 0)
        assert all(C33 > 0)
        assert all(C44 > 0)
        assert all(C55 > 0)
        assert all(C66 > 0)

    @pytest.mark.parametrize("epsilon, gamma, delta", [(0.1, 0.05, 0.08),
                                                       (0.3, 0.15, 0.2),
                                                       (0.5, 0.25, 0.4)])
    @pytest.mark.parametrize("anisotropy_type", ['weak', 'exact'])
    def test_c_vti_tensor_different_anisotropy_values(self, mesh, W, iso_props, epsilon,
                                                      gamma, delta, anisotropy_type):
        """Test VTI tensor with different anisotropy parameter values."""
        vti_props = MockPropVTI(mesh, W, epsilon=epsilon, gamma=gamma,
                                delta=delta, anisotropy=anisotropy_type)
        C_vti = AnisotropyTensor.c_vti_tensor(iso_props, vti_props)

        # Extract components
        C11_func = Function(W).interpolate(C_vti[0, 0])
        C33_func = Function(W).interpolate(C_vti[2, 2])
        C66_func = Function(W).interpolate(C_vti[5, 5])
        C44_func = Function(W).interpolate(C_vti[3, 3])

        # Test anisotropy relationships
        # C11 should be > C33 when epsilon > 0
        expected_C11_ratio = (1 + 2*epsilon)
        assert allclose(C11_func.dat.data / C33_func.dat.data,
                        expected_C11_ratio, rtol=1e-10)

        # C66 should be > C44 when gamma > 0
        expected_C66_ratio = (1 + 2*gamma)
        assert allclose(C66_func.dat.data / C44_func.dat.data,
                        expected_C66_ratio, rtol=1e-10)

    @pytest.mark.parametrize("anisotropy_type", ['weak', 'exact'])
    def test_c_vti_tensor_anisotropy_formulations(self, mesh, W,
                                                  iso_props, anisotropy_type):
        """Test both weak and exact anisotropy formulations."""
        vti_props = MockPropVTI(mesh, W, epsilon=0.2, gamma=0.1,
                                delta=0.15, anisotropy=anisotropy_type)
        C_vti = AnisotropyTensor.c_vti_tensor(iso_props, vti_props)

        # Both formulations should produce valid tensors
<<<<<<< HEAD
        assert isinstance(C_vti, fire.ufl.tensors.ListTensor)
=======
        assert isinstance(C_vti, ListTensor)
>>>>>>> public/main

        # Check that C13 is computed (not NaN or infinite)
        C13_func = Function(W).interpolate(C_vti[0, 2])
        assert not any(isnan(C13_func.dat.data))
        assert not any(isinf(C13_func.dat.data))

    @pytest.mark.parametrize("vti_props_type", ['vti_props_weak', 'vti_props_exact'])
    def test_c_tti_tensor_rotation_identity(self, mesh, W, iso_props,
                                            vti_props_type, request):
        """Test that TTI tensor equals VTI tensor when rotation angles are zero."""
        vti_props_obj = request.getfixturevalue(vti_props_type)
        C_vti = AnisotropyTensor.c_vti_tensor(iso_props, vti_props_obj)
        tti_props_zero = MockPropTTI(mesh, W, theta=0.0, phi=0.0)
        C_tti = AnisotropyTensor.c_tti_tensor(C_vti, tti_props_zero)

        # Compare component by component
        for i in range(6):
            for j in range(6):
                C_vti_ij = Function(W).interpolate(C_vti[i, j])
                C_tti_ij = Function(W).interpolate(C_tti[i, j])
                assert allclose(C_vti_ij.dat.data, C_tti_ij.dat.data, rtol=1e-10)

    @pytest.mark.parametrize("theta", [0, 30, 45, 60, 90])
    @pytest.mark.parametrize("vti_props_type", ['vti_props_weak', 'vti_props_exact'])
    def test_c_tti_tensor_rotation_angles(self, mesh, W, iso_props,
                                          vti_props_type, theta, request):
        """Test TTI tensor with different tilt angles."""
        vti_props_obj = request.getfixturevalue(vti_props_type)
        C_vti = AnisotropyTensor.c_vti_tensor(iso_props, vti_props_obj)
        tti_props = MockPropTTI(mesh, W, theta=theta, phi=0.0)
        C_tti = AnisotropyTensor.c_tti_tensor(C_vti, tti_props)

        if theta == 0:
            # For theta=0, should be same as VTI
            for i in range(6):
                for j in range(6):
                    C_vti_ij = Function(W).interpolate(C_vti[i, j])
                    C_tti_ij = Function(W).interpolate(C_tti[i, j])
                    assert allclose(C_vti_ij.dat.data, C_tti_ij.dat.data, rtol=1e-10)

        else:
            # For other angles, should be different from VTI
            C11_vti = Function(W).interpolate(C_vti[0, 0])
            C11_tti = Function(W).interpolate(C_tti[0, 0])
            assert not allclose(C11_vti.dat.data, C11_tti.dat.data, rtol=1e-6)

    @pytest.mark.parametrize("phi", [0, 45, 90, 135])
    @pytest.mark.parametrize("vti_props_type", ['vti_props_weak', 'vti_props_exact'])
    def test_c_tti_tensor_azimuth_angles(self, mesh, W, iso_props,
                                         vti_props_type, phi, request):
        """Test TTI tensor with different azimuth angles."""
        vti_props_obj = request.getfixturevalue(vti_props_type)
        C_vti = AnisotropyTensor.c_vti_tensor(iso_props, vti_props_obj)
        tti_props = MockPropTTI(mesh, W, theta=45.0, phi=phi)
        C_tti = AnisotropyTensor.c_tti_tensor(C_vti, tti_props)

        # Reference at phi=0
        tti_props_ref = MockPropTTI(mesh, W, theta=45.0, phi=0)
        C_tti_ref = AnisotropyTensor.c_tti_tensor(C_vti, tti_props_ref)

        C11 = Function(W).interpolate(C_tti[0, 0]).dat.data
        C11_ref = Function(W).interpolate(C_tti_ref[0, 0]).dat.data

        if phi == 0:
            assert allclose(C11, C11_ref, rtol=1e-10)  # Should match
        else:
            # For phi not multiple of 180, tensors should differ
            assert not allclose(C11, C11_ref, rtol=1e-6)

    @pytest.mark.parametrize("vti_props_type", ['vti_props_weak', 'vti_props_exact'])
    def test_c_tti_tensor_transformation_matrix_properties(self, mesh, W, iso_props,
                                                           vti_props_type, request):
        """Test properties of the transformation matrix."""
        vti_props_obj = request.getfixturevalue(vti_props_type)
        C_vti = AnisotropyTensor.c_vti_tensor(iso_props, vti_props_obj)
        tti_props = MockPropTTI(mesh, W, theta=30.0, phi=45.0)
        C_tti = AnisotropyTensor.c_tti_tensor(C_vti, tti_props)

        # Check symmetry of resulting tensor
        for i in range(6):
            for j in range(6):
                C_tti_ij = Function(W).interpolate(C_tti[i, j])
                C_tti_ji = Function(W).interpolate(C_tti[j, i])
                assert allclose(C_tti_ij.dat.data, C_tti_ji.dat.data, rtol=1e-10)

    @pytest.mark.parametrize("anisotropy_type", ['weak', 'exact'])
    def test_c_vti_tensor_edge_cases(self, mesh, W, anisotropy_type):
        """Test edge cases for VTI tensor"""
        # Zero density case
        iso_props_zero_rho = MockPropISO(mesh, W, vP_value=2500.0,
                                         vS_value=1200.0, rho_value=0.0)
        vti_props = MockPropVTI(mesh, W, epsilon=0.2, gamma=0.1,
                                delta=0.15, anisotropy=anisotropy_type)

        # Physically meaningless
        C_vti = AnisotropyTensor.c_vti_tensor(iso_props_zero_rho, vti_props)
<<<<<<< HEAD
        assert isinstance(C_vti, fire.ufl.tensors.ListTensor)
=======
        assert isinstance(C_vti, ListTensor)
>>>>>>> public/main

        # Zero velocity case
        iso_props_zero_v = MockPropISO(mesh, W, vP_value=0.0,
                                       vS_value=0.0, rho_value=2200.0)
        C_vti_zero = AnisotropyTensor.c_vti_tensor(iso_props_zero_v, vti_props)

        # All components should be zero
        for i in range(6):
            comp = Function(W).interpolate(C_vti_zero[i, i])
            assert allclose(comp.dat.data, 0.0, atol=1e-12)
