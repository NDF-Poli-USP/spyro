"""Utilities to safely evaluate expression strings into UFL objects.

This module validates a user-provided mathematical expression with Python's
AST before evaluating it in a restricted namespace containing Firedrake/UFL
functions, constants, and spatial coordinates.
"""

from ast import parse, walk, Add, BinOp, Call, Constant, Div, \
    Expression, Load, Name, Mult, Pow, Sub, UnaryOp, UAdd, USub
from firedrake import SpatialCoordinate, acos, asin, atan, atan2, \
    cos, cosh, e, erf, exp, ln, pi, sin, sinh, sqrt, tan, tanh
from spyro.utils.error_management import value_parameter_error


def available_functions_to_eval(mesh, dimension):
    """Return a dictionary of available functions and variables for eval.

    This function creates a namespace dictionary that includes mathematical functions
    from Firedrake and the spatial coordinates of the mesh. The namespace is used to
    safely evaluate user-provided expressions as UFL functions.

    Parameters
    ----------
    mesh : `Firedrake.Mesh`
        The mesh on which the UFL function will be defined.
    dimension : `int`
        The spatial dimension of the mesh. It should be 2 or 3
    """
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
    """Use AST to validate expression structure before eval.

    This function takes a string expression, validates its syntax using the Abstract
    Syntax Tree (AST) module, and safely evaluates it to produce a UFL function.

    Parameters
    ----------
    mesh : `Firedrake.Mesh`
        The mesh on which the UFL function will be defined.
    expression : `str`
        The string expression to be evaluated as a UFL function.
    dimension : `int`
        The spatial dimension of the mesh. It should be 2 or 3
    """

    # Check model dimension
    if dimension not in [2, 3]:
        value_parameter_error('dimension', dimension, [2, 3])

    # Get available functions and variables
    namespace = available_functions_to_eval(mesh, dimension)

    # Parse to AST
    try:
        tree = parse(expression, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e}")

    # Validate AST nodes
    for node in walk(tree):
        # Only allow specific node types including context nodes
        allowed_node_types = (Expression, BinOp, UnaryOp, Call, Name, Add,
                              Sub, Mult, Div, Pow, USub, UAdd, Constant, Load)

        if not isinstance(node, allowed_node_types):
            raise ValueError(
                f"Disallowed syntax element: {type(node).__name__}"
            )

        if not isinstance(node, allowed_node_types):
            raise ValueError(
                f"Disallowed syntax element: {type(node).__name__}")

        # Validate function calls
        if isinstance(node, Call):
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
