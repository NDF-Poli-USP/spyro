import ast
from firedrake import SpatialCoordinate, acos, asin, atan, atan2, \
    cos, cosh, e, erf, exp, ln, pi, sin, sinh, sqrt, tan, tanh


def available_functions_to_eval(mesh, dimension):

    namespace = {"acos": acos,
                 "asin": asin,
                 "atan": atan,
                 "atan2": atan2,
                 "cos": cos,
                 "cosh": cosh,
                 "e": e,
                 "erf": erf,
                 "exp": exp,
                 "ln": ln,
                 "pi": pi,
                 "sin": sin,
                 "sinh": sinh,
                 "sqrt": sqrt,
                 "tan": tan,
                 "tanh": tanh}

    coords = SpatialCoordinate(mesh)
    z, x = coords[0], coords[1]
    namespace.update({'z': z, 'x': x})

    if dimension == 3:  # 3D
        y = coords[2]
        namespace.update({'y': y})

    return namespace


def generate_ufl_functions(mesh, expression, dimension):
    '''
    Use AST to validate expression structure before eval.
    '''

    # Get available functions and variables
    namespace = available_functions_to_eval(mesh, dimension)

    # Parse to AST
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e}")

    # Validate AST nodes
    for node in ast.walk(tree):
        # Only allow specific node types including context nodes
        allowed_node_types = (ast.Expression, ast.BinOp, ast.UnaryOp,
                              ast.Call, ast.Name, ast.Add, ast.Sub,
                              ast.Mult, ast.Div, ast.Pow, ast.USub,
                              ast.UAdd, ast.Constant, ast.Load)

        if not isinstance(node, allowed_node_types):
            raise ValueError(
                f"Disallowed syntax element: {type(node).__name__}"
            )

        if not isinstance(node, allowed_node_types):
            raise ValueError(
                f"Disallowed syntax element: {type(node).__name__}")

        # Validate function calls
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed")
            if node.func.id not in namespace:
                raise ValueError(f"Unknown function: {node.func.id}")

        # Validate names
        if isinstance(node, ast.Name):
            if node.id not in namespace:
                raise ValueError(f"Unknown variable: {node.id}")

    # Safe to compile and eval
    code = compile(tree, '<ufl_expr>', 'eval')

    # Evaluate with restricted namespace
    ufl_function = eval(code, {"__builtins__": {}}, namespace)

    return ufl_function
