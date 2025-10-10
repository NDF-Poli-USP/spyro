"""
Test script to verify that Firedrake correctly reads physical tags 
from GMSH meshes for dx and ds integrals.

This tests:
1. Domain tags (2D physical groups) for dx integrals
2. Boundary tags (1D physical groups) for ds integrals
3. Compatibility with spyro's acoustic solver
"""

import pytest
import numpy as np
import os
import tempfile
import firedrake as fire
from spyro.meshing.meshing_parameters import MeshingParameters
from spyro.meshing.meshing_functions import build_big_rect_with_inner_element_group


@pytest.fixture
def test_mesh_with_tags():
    """Create a test mesh with both boundary and domain tags."""
    
    # Use temporary file
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp_file:
        output_filename = tmp_file.name
    
    # Create mesh parameters with gradient mask to get inner/outer domains
    input_mesh_parameters = {
        "length_z": 2.0,
        "length_x": 3.0,
        "mesh_type": "spyro_mesh",
        "output_filename": output_filename,
        "edge_length": 0.2,
        "dimension": 2,
        "grid_velocity_data": None,
        "gradient_mask": {
            "z_min": -1.5,
            "z_max": -0.5,
            "x_min": 0.5,
            "x_max": 2.5,
        },
    }
    
    mesh_parameters = MeshingParameters(input_mesh_dictionary=input_mesh_parameters)
    build_big_rect_with_inner_element_group(mesh_parameters)
    
    yield output_filename
    
    # Cleanup
    if os.path.exists(output_filename):
        os.remove(output_filename)


def test_firedrake_domain_tags(test_mesh_with_tags):
    """Test if Firedrake can correctly read and use domain tags for dx integrals."""
    
    mesh_file = test_mesh_with_tags
    mesh = fire.Mesh(mesh_file)
    
    # Create a function space
    V = fire.FunctionSpace(mesh, "CG", 1)
    
    # Test dx without tag (should integrate over entire domain)
    area_all = fire.assemble(fire.Constant(1.0) * fire.dx(domain=mesh))
    assert area_all > 0, "Total domain area should be positive"
    
    # Test specific domain tags (if gradient mask was used)
    area_outer = fire.assemble(fire.Constant(1.0) * fire.dx(1, domain=mesh))
    area_inner = fire.assemble(fire.Constant(1.0) * fire.dx(2, domain=mesh))
    
    assert area_outer > 0, "Outer domain area should be positive"
    assert area_inner > 0, "Inner domain area should be positive"
    
    # Check area conservation
    total_area_sum = area_outer + area_inner
    area_diff = abs(area_all - total_area_sum)
    assert area_diff < 1e-10, f"Area conservation failed: {area_all} != {total_area_sum}"


def test_firedrake_boundary_tags(test_mesh_with_tags):
    """Test if Firedrake can correctly read and use boundary tags for ds integrals."""
    
    mesh_file = test_mesh_with_tags
    mesh = fire.Mesh(mesh_file)
    
    # Test ds without tag (should integrate over entire boundary)
    perimeter_all = fire.assemble(fire.Constant(1.0) * fire.ds(domain=mesh))
    expected_perimeter = 2 * (2.0 + 3.0)  # Rectangle perimeter
    
    assert abs(perimeter_all - expected_perimeter) < 1e-10, \
        f"Total perimeter mismatch: {perimeter_all} != {expected_perimeter}"
    
    # Test specific boundary tags
    boundary_names = {1: "Top", 2: "Bottom", 3: "Right", 4: "Left"}
    expected_lengths = {1: 3.0, 2: 3.0, 3: 2.0, 4: 2.0}
    
    total_boundary_sum = 0
    for boundary_id, expected_length in expected_lengths.items():
        length_boundary = fire.assemble(fire.Constant(1.0) * fire.ds(boundary_id, domain=mesh))
        assert abs(length_boundary - expected_length) < 1e-10, \
            f"Boundary {boundary_id} length mismatch: {length_boundary} != {expected_length}"
        total_boundary_sum += length_boundary
    
    # Check perimeter conservation
    perimeter_diff = abs(perimeter_all - total_boundary_sum)
    assert perimeter_diff < 1e-10, \
        f"Perimeter conservation failed: {perimeter_all} != {total_boundary_sum}"


def test_acoustic_solver_style_integrals(test_mesh_with_tags):
    """Test integrals similar to what's used in the acoustic solver."""
    
    mesh_file = test_mesh_with_tags
    mesh = fire.Mesh(mesh_file)
    
    V = fire.FunctionSpace(mesh, "CG", 1)
    u = fire.TrialFunction(V)
    v = fire.TestFunction(V)
    
    # Test acoustic solver-like forms
    c = fire.Function(V)
    c.assign(fire.Constant(1500.0))  # Sound speed
    dt = fire.Constant(0.001)
    
    # Mass matrix term (domain integral)
    mass_form = (1 / (c * c)) * u * v * fire.dx
    M = fire.assemble(mass_form)
    assert M is not None, "Mass matrix assembly failed"
    
    # Stiffness matrix term (domain integral)  
    stiffness_form = fire.dot(fire.grad(u), fire.grad(v)) * fire.dx
    K = fire.assemble(stiffness_form)
    assert K is not None, "Stiffness matrix assembly failed"
    
    # Absorbing boundary terms (boundary integrals)
    u_n = fire.Function(V)
    u_nm1 = fire.Function(V)
    
    weak_expr_abc = fire.dot((u_n - u_nm1) / dt, v)
    f_abc = (1 / c) * weak_expr_abc
    
    # Test all boundary conditions
    for boundary_id in [1, 2, 3, 4]:
        abc_form = f_abc * fire.ds(boundary_id)
        abc_vector = fire.assemble(abc_form)
        assert abc_vector is not None, f"ABC boundary {boundary_id} assembly failed"


def test_spyro_acoustic_solver_compatibility(test_mesh_with_tags):
    """Test full compatibility with spyro's acoustic solver expectations."""
    
    mesh_file = test_mesh_with_tags
    mesh = fire.Mesh(mesh_file)
    
    V = fire.FunctionSpace(mesh, "CG", 1)
    u = fire.TrialFunction(V)
    v = fire.TestFunction(V)
    
    # Simulate Wave_object parameters
    c = fire.Function(V)
    c.assign(fire.Constant(1500.0))
    dt_val = 0.001
    dt = fire.Constant(dt_val)
    
    # Test the exact integral forms from acoustic solver
    u_n = fire.Function(V)
    u_nm1 = fire.Function(V)
    
    # Mass matrix term (as in actual solver)
    m1 = (1 / (c * c)) * ((u - 2.0 * u_n + u_nm1) / (dt**2)) * v * fire.dx
    
    # Stiffness term
    a = fire.dot(fire.grad(u), fire.grad(v)) * fire.dx
    
    # ABC terms - exactly as in the solver
    weak_expr_abc = fire.dot((u_n - u_nm1) / dt, v)
    f_abc = (1 / c) * weak_expr_abc
    
    # Combine all boundary terms
    le = f_abc * fire.ds(1) + f_abc * fire.ds(2) + f_abc * fire.ds(3) + f_abc * fire.ds(4)
    
    # Assemble the complete form (as in actual solver)
    form = m1 + a + le
    lhs = fire.lhs(form)
    rhs = fire.rhs(form)
    
    # Test assembly
    A = fire.assemble(lhs, mat_type="matfree")
    assert A is not None, "Complete acoustic solver form assembly failed"
    
    # Test with linear solver setup (as in the actual solver)
    solver_parameters = {
        "ksp_type": "preonly", 
        "pc_type": "jacobi",
    }
    solver = fire.LinearSolver(A, solver_parameters=solver_parameters)
    assert solver is not None, "Linear solver setup failed"


def test_mesh_properties(test_mesh_with_tags):
    """Test basic mesh properties and structure."""
    
    mesh_file = test_mesh_with_tags
    mesh = fire.Mesh(mesh_file)
    
    # Basic mesh validation
    assert mesh.num_vertices() > 0, "Mesh should have vertices"
    assert mesh.num_cells() > 0, "Mesh should have cells"
    
    # Function space creation
    V = fire.FunctionSpace(mesh, "CG", 1)
    assert V.dim() > 0, "Function space should have DOFs"
    
    # Basic function operations
    f = fire.Function(V)
    f.assign(fire.Constant(1.0))
    
    # Test that function assignment worked
    assert np.allclose(f.dat.data, 1.0), "Function assignment failed"


if __name__ == "__main__":
    # Allow running as standalone script for debugging
    pytest.main([__file__])
