"""Utilities to safely evaluate expression strings into UFL objects.

This module validates a user-provided mathematical expression with Python's
AST before evaluating it in a restricted namespace containing Firedrake/UFL
functions, constants, and spatial coordinates.
"""

import ast
from firedrake import (
    SpatialCoordinate,
    acos,
    asin,
    atan,
    atan2,
    cos,
    cosh,
    e,
    erf,
    exp,
    ln,
    pi,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
)


def available_functions_to_eval(mesh, dimension):
    """Build the allowed namespace for expression evaluation.

    Parameters
    ----------
    mesh : firedrake.mesh.MeshGeometry
        Firedrake mesh used to obtain spatial coordinates.
    dimension : int
        Spatial dimension of the problem. Supported values are ``2`` and ``3``.

    Returns
    -------
    dict[str, object]
        Mapping from allowed symbol names to UFL-compatible objects. The
        namespace includes mathematical functions/constants and coordinate
        symbols ``z`` and ``x`` (plus ``y`` for 3D).
    """
    namespace = {
        "acos": acos,
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
        "tanh": tanh,
    }

    coords = SpatialCoordinate(mesh)
    z, x = coords[0], coords[1]
    namespace.update({"z": z, "x": x})

    if dimension == 3:  # 3D
        y = coords[2]
        namespace.update({"y": y})

    return namespace


def generate_ufl_functions(mesh, expression, dimension):
    """Validate and evaluate an expression string into a UFL object.

    The expression is parsed with ``ast.parse`` and checked node-by-node to
    allow only basic arithmetic, unary operators, names, constants, and simple
    function calls from the approved namespace.

    Parameters
    ----------
    mesh : firedrake.mesh.MeshGeometry
        Firedrake mesh used to define spatial coordinates.
    expression : str
        Mathematical expression to evaluate, for example
        ``"sin(pi*x) * exp(-z)"``.
    dimension : int
        Spatial dimension of the problem. Supported values are ``2`` and ``3``.

    Returns
    -------
    object
        Evaluated UFL expression built from the validated string.

    Raises
    ------
    ValueError
        If the expression has invalid syntax, uses disallowed AST nodes,
        references unknown names, or calls unknown functions.
    """
    # Get available functions and variables
    namespace = available_functions_to_eval(mesh, dimension)

    # Parse to AST
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e}")

    # Validate AST nodes
    for node in ast.walk(tree):
        # Only allow specific node types including context nodes
        allowed_node_types = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Call,
            ast.Name,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Constant,
            ast.Load,
        )

        if not isinstance(node, allowed_node_types):
            raise ValueError(f"Disallowed syntax element: {type(node).__name__}")

        if not isinstance(node, allowed_node_types):
            raise ValueError(f"Disallowed syntax element: {type(node).__name__}")

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
    code = compile(tree, "<ufl_expr>", "eval")

    # Evaluate with restricted namespace
    ufl_function = eval(code, {"__builtins__": {}}, namespace)

    return ufl_function
