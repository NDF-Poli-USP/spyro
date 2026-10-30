from functools import cache
from importlib.metadata import version
from importlib.util import find_spec
from packaging.version import Version
import firedrake  # noqa: F401


def is_firedrake_new(print_version=False, comm=None):
    """Check whether the installed Firedrake version is recent enough.

    Parameters
    ----------
    print_version : bool, optional
        If True, print the installed Firedrake version before returning.
        Default is False.

    Returns
    -------
    bool
        True when the installed Firedrake version is greater than or equal
        to ``2026.4``.
    """
    return Version(version("firedrake")) >= Version("2026.4")


@cache
def is_firedrake_pyop3() -> bool:
    """Check whether the installed Firedrake is built on pyop3.

    Firedrake ``main`` replaced pyop2 with pyop3, which changed several mesh
    and function space APIs. The released ``2026.4`` series still uses
    pyop2, so this cannot be decided from the version number alone.

    Returns
    -------
    bool
        True when the ``pyop3`` package is importable.
    """
    return find_spec("pyop3") is not None
