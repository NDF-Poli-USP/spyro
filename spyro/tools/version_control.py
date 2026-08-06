from packaging.version import Version
from importlib.metadata import version
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
