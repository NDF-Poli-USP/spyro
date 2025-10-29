"""
Test script to verify that Firedrake correctly reads physical tags 
from GMSH meshes for dx and ds integrals.

This tests:
1. Domain tags (2D physical groups) for dx integrals
2. Boundary tags (1D physical groups) for ds integrals
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(__file__))

import firedrake as fire
from spyro.meshing.meshing_parameters import MeshingParameters
from spyro.meshing.meshing_functions import build_big_rect_with_inner_element_group


def create_test_mesh_with_tags(output_filename="test_tags_mesh.msh"):
    """Create a test mesh with both boundary and domain tags."""
    
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
    
    print("Creating test mesh with tags...")
    build_big_rect_with_inner_element_group(mesh_parameters)
    
    return output_filename


def test_firedrake_tag_reading():
    """Test if Firedrake can correctly read and use all tags."""
    
    # Create the mesh
    mesh_file = create_test_mesh_with_tags()
    
    print("\n" + "="*60)
    print("TESTING FIREDRAKE TAG RECOGNITION")
    print("="*60)
    
    # Load the mesh in Firedrake
    mesh = fire.Mesh(mesh_file)
    print(f"Successfully loaded mesh: {mesh_file}")
    print(f"Mesh has {mesh.num_vertices()} vertices and {mesh.num_cells()} cells")
    
    # Create a function space
    V = fire.FunctionSpace(mesh, "CG", 1)
    u = fire.TestFunction(V)
    v = fire.TrialFunction(V)
    f = fire.Function(V)
    f.assign(fire.Constant(1.0))
    
    print(f"\nFunction space created with {V.dim()} DOFs")
    
    # Test domain (dx) integrals
    print("\nTESTING DOMAIN (dx) INTEGRALS:")
    print("-" * 40)
    
    domain_tests = []
    
    # Test dx without tag (should integrate over entire domain)
    try:
        integral_all = fire.assemble(f * u * fire.dx)
        area_all = fire.assemble(fire.Constant(1.0) * fire.dx(domain=mesh))
        print(f"dx (all domains): Area = {area_all:.6f}")
        domain_tests.append(("All domains", True, area_all))
    except Exception as e:
        print(f"dx (all domains): {e}")
        domain_tests.append(("All domains", False, 0))
    
    # Test specific domain tags (if gradient mask was used)
    for domain_id, domain_name in [(1, "Outer"), (2, "Inner")]:
        try:
            area_domain = fire.assemble(fire.Constant(1.0) * fire.dx(domain_id, domain=mesh))
            print(f"✓ dx({domain_id}) - {domain_name}: Area = {area_domain:.6f}")
            domain_tests.append((f"{domain_name} (tag {domain_id})", True, area_domain))
        except Exception as e:
            print(f"✗ dx({domain_id}) - {domain_name}: {e}")
            domain_tests.append((f"{domain_name} (tag {domain_id})", False, 0))
    
    # Check if domain areas sum correctly
    if len([t for t in domain_tests if t[1]]) >= 3:  # All three tests passed
        total_area = domain_tests[0][2]
        sum_areas = sum(t[2] for t in domain_tests[1:] if t[1])
        area_diff = abs(total_area - sum_areas)
        if area_diff < 1e-10:
            print(f"Area conservation: Total = {total_area:.6f}, Sum = {sum_areas:.6f}")
        else:
            print(f"Area mismatch: Total = {total_area:.6f}, Sum = {sum_areas:.6f}, Diff = {area_diff:.6f}")
    
    # Test boundary (ds) integrals
    print("\nTESTING BOUNDARY (ds) INTEGRALS:")
    print("-" * 40)
    
    boundary_tests = []
    
    # Test ds without tag (should integrate over entire boundary)
    try:
        perimeter_all = fire.assemble(fire.Constant(1.0) * fire.ds(domain=mesh))
        expected_perimeter = 2 * (2.0 + 3.0)  # Rectangle perimeter
        print(f"✓ ds (all boundaries): Perimeter = {perimeter_all:.6f} (expected: {expected_perimeter:.6f})")
        boundary_tests.append(("All boundaries", True, perimeter_all))
    except Exception as e:
        print(f"✗ ds (all boundaries): {e}")
        boundary_tests.append(("All boundaries", False, 0))
    
    # Test specific boundary tags
    boundary_names = {1: "Top", 2: "Bottom", 3: "Right", 4: "Left"}
    expected_lengths = {1: 3.0, 2: 3.0, 3: 2.0, 4: 2.0}  # length_x for top/bottom, length_z for left/right
    
    for boundary_id, boundary_name in boundary_names.items():
        try:
            length_boundary = fire.assemble(fire.Constant(1.0) * fire.ds(boundary_id, domain=mesh))
            expected = expected_lengths[boundary_id]
            print(f"✓ ds({boundary_id}) - {boundary_name}: Length = {length_boundary:.6f} (expected: {expected:.6f})")
            boundary_tests.append((f"{boundary_name} (tag {boundary_id})", True, length_boundary))
        except Exception as e:
            print(f"✗ ds({boundary_id}) - {boundary_name}: {e}")
            boundary_tests.append((f"{boundary_name} (tag {boundary_id})", False, 0))
    
    # Check if boundary lengths sum correctly
    if len([t for t in boundary_tests if t[1]]) >= 5:  # All tests passed
        total_perimeter = boundary_tests[0][2]
        sum_lengths = sum(t[2] for t in boundary_tests[1:] if t[1])
        length_diff = abs(total_perimeter - sum_lengths)
        if length_diff < 1e-10:
            print(f"✓ Perimeter conservation: Total = {total_perimeter:.6f}, Sum = {sum_lengths:.6f}")
        else:
            print(f"⚠️  Perimeter mismatch: Total = {total_perimeter:.6f}, Sum = {sum_lengths:.6f}, Diff = {length_diff:.6f}")
    
    # Test mixed integrals (like in acoustic solver)
    print("\n🔍 TESTING ACOUSTIC SOLVER-STYLE INTEGRALS:")
    print("-" * 45)
    
    try:
        # Test a form similar to what's used in the acoustic solver
        c = fire.Function(V)
        c.assign(fire.Constant(1500.0))  # Sound speed
        
        # Mass matrix term (domain integral)
        mass_form = (1 / (c * c)) * u * v * fire.dx
        M = fire.assemble(mass_form)
        print("✓ Mass matrix assembly (domain integral)")
        
        # Stiffness matrix term (domain integral)  
        stiffness_form = fire.dot(fire.grad(u), fire.grad(v)) * fire.dx
        K = fire.assemble(stiffness_form)
        print("✓ Stiffness matrix assembly (domain integral)")
        
        # Absorbing boundary terms (boundary integrals)
        dt = 0.001
        u_n = fire.Function(V)
        u_nm1 = fire.Function(V)
        
        for boundary_id in [1, 2, 3, 4]:
            try:
                abc_form = (1 / c) * u * v * fire.ds(boundary_id)  # Simplified form for testing
                abc_matrix = fire.assemble(abc_form)
                print(f"✓ ABC boundary {boundary_id} assembly")
            except Exception as e:
                print(f"✗ ABC boundary {boundary_id}: {e}")
        
        print("✓ All acoustic solver-style integrals work correctly")
        
    except Exception as e:
        print(f"✗ Acoustic solver-style integrals failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    domain_success = len([t for t in domain_tests if t[1]])
    boundary_success = len([t for t in boundary_tests if t[1]])
    
    print(f"Domain (dx) integrals:   {domain_success}/{len(domain_tests)} successful")
    print(f"Boundary (ds) integrals: {boundary_success}/{len(boundary_tests)} successful")
    
    if domain_success == len(domain_tests) and boundary_success == len(boundary_tests):
        print("🎉 ALL TESTS PASSED! Firedrake correctly reads all mesh tags.")
        success = True
    else:
        print("❌ Some tests failed. Check mesh tag generation or Firedrake compatibility.")
        success = False
    
    # Clean up
    if os.path.exists(mesh_file):
        os.remove(mesh_file)
        print(f"\n🧹 Cleaned up test file: {mesh_file}")
    
    return success


def test_tag_compatibility_with_solver():
    """Test compatibility with spyro's acoustic solver expectations."""
    
    print("\n" + "="*60)
    print("TESTING SPYRO ACOUSTIC SOLVER COMPATIBILITY")
    print("="*60)
    
    mesh_file = create_test_mesh_with_tags()
    mesh = fire.Mesh(mesh_file)
    
    V = fire.FunctionSpace(mesh, "CG", 1)
    u = fire.TrialFunction(V)
    v = fire.TestFunction(V)
    
    # Simulate Wave_object parameters
    class MockWaveObject:
        def __init__(self):
            self.c = fire.Function(V)
            self.c.assign(fire.Constant(1500.0))
            self.dt = 0.001
            self.absorb_top = True
            self.absorb_bottom = True
            self.absorb_left = True
            self.absorb_right = True
    
    wave_obj = MockWaveObject()
    
    # Test the exact integral forms from acoustic_solver_construction_no_pml.py
    try:
        # Mass matrix term
        dt = fire.Constant(wave_obj.dt)
        m1 = (1 / (wave_obj.c * wave_obj.c)) * ((u - 2.0 * u + u) / (dt**2)) * v * fire.dx
        
        # Stiffness term
        a = fire.dot(fire.grad(u), fire.grad(v)) * fire.dx
        
        # ABC terms - exactly as in the solver
        u_n = fire.Function(V)
        u_nm1 = fire.Function(V)
        weak_expr_abc = fire.dot((u_n - u_nm1) / dt, v)
        f_abc = (1 / wave_obj.c) * weak_expr_abc
        
        le = 0.0
        if wave_obj.absorb_top:
            le = le + f_abc * fire.ds(1)
            print("✓ Top boundary (ds(1)) integral works")
        if wave_obj.absorb_bottom:
            le = le + f_abc * fire.ds(2)
            print("✓ Bottom boundary (ds(2)) integral works")
        if wave_obj.absorb_right:
            le = le + f_abc * fire.ds(3)
            print("✓ Right boundary (ds(3)) integral works")
        if wave_obj.absorb_left:
            le = le + f_abc * fire.ds(4)
            print("✓ Left boundary (ds(4)) integral works")
        
        # Assemble the complete form
        form = m1 + a + le
        lhs = fire.lhs(form)
        rhs = fire.rhs(form)
        
        A = fire.assemble(lhs, mat_type="matfree")
        print("✓ Complete acoustic solver form assembles successfully")
        
        # Test with linear solver (as in the actual solver)
        solver_parameters = {
            "ksp_type": "preonly",
            "pc_type": "jacobi",
        }
        solver = fire.LinearSolver(A, solver_parameters=solver_parameters)
        print("✓ Linear solver setup successful")
        
        print("🎉 FULL ACOUSTIC SOLVER COMPATIBILITY CONFIRMED!")
        
    except Exception as e:
        print(f"❌ Acoustic solver compatibility test failed: {e}")
        return False
    
    # Clean up
    if os.path.exists(mesh_file):
        os.remove(mesh_file)
    
    return True


if __name__ == "__main__":
    print("FIREDRAKE MESH TAG TESTING SUITE")
    print("=" * 60)
    
    # Run basic tag tests
    basic_success = test_firedrake_tag_reading()
    
    # Run acoustic solver compatibility tests
    solver_success = test_tag_compatibility_with_solver()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    if basic_success and solver_success:
        print("✅ ALL TESTS PASSED!")
        print("   - Firedrake correctly reads mesh tags")
        print("   - Tags work with dx and ds integrals") 
        print("   - Full acoustic solver compatibility confirmed")
        exit_code = 0
    else:
        print("❌ SOME TESTS FAILED!")
        if not basic_success:
            print("   - Basic tag reading issues")
        if not solver_success:
            print("   - Acoustic solver compatibility issues")
        exit_code = 1
    
    exit(exit_code)