from firedrake import ConvergenceError, Function, \
    FunctionSpace, UnitSquareMesh, UnitCubeMesh
from spyro.utils.eval_functions_to_ufl import generate_ufl_functions


def run_test_eval_ufl_functions():
    '''
    Run comprehensive tests for 2D/3D models and dangerous operations.
    '''

    print("=" * 80)
    print("COMPREHENSIVE UFL EXPRESSION GENERATOR TEST SUITE")
    print("=" * 80)

    run_test_eval_ufl_functions_2d()
    run_test_eval_ufl_functions_3d()
    run_test_eval_danger_ops()


def run_test_eval_ufl_functions_2d():
    # ==================== 2D MODELS ====================
    print("\n" + "=" * 40)
    print("2D MODELS TEST")
    print("=" * 40)

    mesh_2d = UnitSquareMesh(10, 10)
    dimension_2d = 2

    test_expressions_2d = [
        # Basic operations
        ("2 * cos(pi * x) - sqrt(z/4)", "Basic arithmetic"),
        ("x**2 + sin(2*pi*z)", "Polynomial with trig"),
        ("exp(-x**2) * cos(z)", "Gaussian wave"),
        ("sqrt(x**2 + z**2)", "Distance from origin"),
        ("tanh(3*x) * sin(4*z)", "Soliton-like"),

        # Layered models (common in seismic)
        ("1.5 + 0.5*tanh((x-0.5)/0.1)", "Velocity gradient"),
        ("2.0 + 0.3*sin(4*pi*x) + 0.2*cos(4*pi*z)", "Checkerboard"),
        ("1.0 + 0.5*exp(-10*((x-0.3)**2 + (z-0.7)**2))", "Gaussian lens"),

        # More complex
        ("atan2(2*x-1, 2*z-1)", "Angle from center"),
        ("ln(1 + x**2 + z**2)", "Logarithmic field"),
        ("sin(pi*x) * cos(pi*z) * exp(-(x**2 + z**2))", "Decaying wave"),

        # Constants and unary operations
        ("-x**2 + +z", "Mixed unary operators"),
        ("pi * e * x * z", "Mathematical constants"),

        # Edge cases
        ("x/(z + 1e-16)", "Division"),
        ("(x + z) * (x - z)", "Polynomial expansion"),
        ("sin(x)**2 + cos(z)**2", "Trig identity"),
    ]

    print(f"\nTesting {len(test_expressions_2d)} 2D expressions...")
    print("-" * 80)

    successes_2d = 0
    for expr, description in test_expressions_2d:
        try:
            ufl_expr = generate_ufl_functions(mesh_2d, expr, dimension_2d)

            # Create function space and interpolate
            V = FunctionSpace(mesh_2d, "KMV", 4)
            f = Function(V, name="test_field")
            f.interpolate(ufl_expr)

            # Compute some statistics
            data = f.dat.data
            success_msg = (
                f"✅ {description}\n"
                f"   Expression: {expr}\n"
                f"   Type: {type(ufl_expr).__name__}\n"
                f"   Range: [{data.min():.4f}, {data.max():.4f}]\n"
                f"   Mean: {data.mean():.4f}"
            )
            print(success_msg)
            successes_2d += 1

        except ConvergenceError as e:
            print(f"❌ {description}")
            print(f"   Expression: {expr}")
            print(f"   Error: {e}")
        print("-" * 80)

    # ==================== SUMMARY ====================
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)

    total_2d = len(test_expressions_2d)
    print(f"\n2D Models: {successes_2d}/{total_2d} successful "
          f"({100*successes_2d/total_2d:.1f}%)")


def run_test_eval_ufl_functions_3d():
    # ==================== 3D MODELS ====================
    print("\n" + "=" * 40)
    print("3D MODELS TEST")
    print("=" * 40)

    mesh_3d = UnitCubeMesh(10, 10, 10)
    dimension_3d = 3

    test_expressions_3d = [
        # 3D specific
        ("sqrt(x**2 + y**2 + z**2)", "3D distance from origin"),
        ("sin(pi*x) * cos(pi*y) * tanh(pi*z)", "3D wave"),
        ("exp(-(x**2 + y**2 + z**2))", "3D Gaussian"),

        # Geophysical models
        ("2.0 + 0.5*tanh((z-0.5)/0.2)", "Layered earth model"),
        ("1.5 + 0.3*sin(2*pi*x)*cos(2*pi*y)*sin(2*pi*z)", "3D checkerboard"),
        ("atan2(y, x)", "Azimuthal angle"),

        # Complex 3D
        ("ln(1 + x**2 + y**2 + z**2)", "3D logarithmic"),
        ("x*y*z", "Triple product"),
        ("sin(x)*cos(y) + cos(x)*sin(z) + sin(y)*cos(z)", "3D trig mix"),
    ]

    print(f"\nTesting {len(test_expressions_3d)} 3D expressions...")
    print("-" * 80)

    successes_3d = 0
    for expr, description in test_expressions_3d:
        try:
            ufl_expr = generate_ufl_functions(mesh_3d, expr, dimension_3d)

            # Create function space and interpolate
            V = FunctionSpace(mesh_3d, "KMV", 3)
            f = Function(V, name="test_3d")
            f.interpolate(ufl_expr)

            data = f.dat.data
            success_msg = (
                f"✅ {description}\n"
                f"   Expression: {expr}\n"
                f"   Type: {type(ufl_expr).__name__}\n"
                f"   Range: [{data.min():.4f}, {data.max():.4f}]"
            )
            print(success_msg)
            successes_3d += 1

        except ConvergenceError as e:
            print(f"❌ {description}")
            print(f"   Expression: {expr}")
            print(f"   Error: {e}")
        print("-" * 80)

    # ==================== SUMMARY ====================
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)

    total_3d = len(test_expressions_3d)
    print(f"3D Models: {successes_3d}/{total_3d} successful "
          f"({100*successes_3d/total_3d:.1f}%)")


def run_test_eval_danger_ops():

    # ==================== DANGEROUS OPERATIONS ====================
    print("\n" + "=" * 40)
    print("DANGEROUS OPERATIONS BLOCKING TEST")
    print("=" * 40)

    dangerous_expressions = [
        # Code injection
        ("__import__('os').system('ls')", "Import and system call"),
        ("eval('1+1')", "Nested eval call"),
        ("exec('import os')", "Exec statement"),
        ("open('/etc/passwd').read()", "File access"),

        # Attribute access
        ("x.__class__", "Class access"),
        ("x.__dict__", "Dict access"),
        ("cos.__code__", "Function code access"),

        # Complex control flow
        ("[i for i in range(10)]", "List comprehension"),
        ("x if x > 0 else 0", "Conditional expression"),
        ("lambda x: x**2", "Lambda function"),

        # Unauthorized operators
        ("x // 2", "Floor division"),
        ("x % 2", "Modulo operator"),
        ("x & 1", "Bitwise AND"),
        ("x | 1", "Bitwise OR"),
        ("x ^ 1", "Bitwise XOR"),
        ("x << 1", "Bit shift"),
        ("x >> 1", "Right shift"),
        ("x @ y", "Matrix multiply"),

        # Subscripting and slicing
        ("x[0]", "Subscript"),
        ("x[0:2]", "Slice"),

        # Comparisons (if not explicitly allowed)
        ("x > 0", "Comparison"),
        ("x == y", "Equality"),

        # Unauthorized functions
        ("abs(x)", "abs function"),
        ("round(x)", "round function"),
        ("pow(x, 2)", "pow function"),

        # Invalid syntax patterns
        ("x, y", "Tuple"),
        ("x; y", "Multiple statements"),
        ("{x: y}", "Dictionary"),
        ("[x, y]", "List"),

        # Undefined variables
        ("unknown_var + x", "Undefined variable"),
        ("some_function(x)", "Undefined function"),

        # Nested/dangerous calls
        ("cos(cos.__name__)", "Meta programming"),
        ("globals()", "Global namespace"),
        ("locals()", "Local namespace"),
    ]

    print(f"\nTesting {len(dangerous_expressions)} "
          f"dangerous expressions (should all fail)...")
    print("-" * 80)

    blocked_dangerous = 0
    for expr, description in dangerous_expressions:
        try:
            # Use appropriate mesh based on expression
            if 'y' in expr and 'z' not in expr:
                # Some dangerous expressions might use y as variable
                mesh = mesh_2d
                dim = 2
            else:
                mesh = mesh_2d
                dim = 2

            ufl_expr = generate_ufl_functions(mesh, expr, dim)
            print(f"❌ FAILED TO BLOCK: {description}")
            print(f"   Expression: {expr}")
            print(f"   Created: {type(ufl_expr)}")

        except ValueError as e:
            print(f"✅ BLOCKED: {description}")
            print(f"   Expression: {expr}")
            print(f"   Reason: {str(e)[:60]}...")
            blocked_dangerous += 1
        except Exception as e:
            # Other exceptions (syntax errors, etc.) also count as blocked
            print(f"✅ BLOCKED: {description}")
            print(f"   Expression: {expr}")
            print(f"   Error type: {type(e).__name__}")
            blocked_dangerous += 1
        print("-" * 80)

    # ==================== SUMMARY ====================
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)

    total_dangerous = len(dangerous_expressions)
    print(f"Dangerous ops: {blocked_dangerous}/{total_dangerous} blocked "
          f"({100*blocked_dangerous/total_dangerous:.1f}%)")


if __name__ == "__main__":
    run_test_eval_ufl_functions()
