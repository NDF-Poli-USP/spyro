import pytest
import matplotlib
matplotlib.use("Agg")


def pytest_addoption(parser):
    parser.addoption(
        "--skip-slow", action="store_true", default=False, help="skip tests marked as slow"
    )
    parser.addoption(
        "--only-slow", action="store_true", default=False, help="run only slow tests"
    )
    parser.addoption(
        "--skip-high-memory", action="store_true", default=False, help="skip tests marked as high_memory"
    )
    parser.addoption(
        "--only-high-memory", action="store_true", default=False, help="run only tests marked as high_memory"
    )
    parser.addoption(
        "--skip-older-firedrake",
        action="store_true",
        default=False,
        help="skip tests marked as older_firedrake",
    )
    parser.addoption(
        "--only-older-firedrake",
        action="store_true",
        default=False,
        help="run only tests marked as older_firedrake",
    )
    parser.addoption(
        "--skip-newer-firedrake",
        action="store_true",
        default=False,
        help="skip tests marked as newer_firedrake",
    )
    parser.addoption(
        "--only-newer-firedrake",
        action="store_true",
        default=False,
        help="run only tests marked as newer_firedrake",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "high_memory: mark test as requiring high memory")
    config.addinivalue_line(
        "markers", "older_firedrake: mark test as only compatible with older firedrake versions"
    )
    config.addinivalue_line(
        "markers", "newer_firedrake: mark test as only compatible with newer firedrake versions"
    )


def pytest_collection_modifyitems(config, items):
    skip_slow = config.getoption("--skip-slow")
    only_slow = config.getoption("--only-slow")
    skip_high_memory = config.getoption("--skip-high-memory")
    only_high_memory = config.getoption("--only-high-memory")
    skip_older_firedrake = config.getoption("--skip-older-firedrake")
    only_older_firedrake = config.getoption("--only-older-firedrake")
    skip_newer_firedrake = config.getoption("--skip-newer-firedrake")
    only_newer_firedrake = config.getoption("--only-newer-firedrake")

    if skip_slow and only_slow:
        raise pytest.UsageError("Cannot use both --skip-slow and --only-slow")
    if skip_high_memory and only_high_memory:
        raise pytest.UsageError("Cannot run both --skip and only options for high memory")

    if skip_older_firedrake and only_older_firedrake:
        raise pytest.UsageError("Cannot run both --skip and only options for older firedrake")
    if skip_newer_firedrake and only_newer_firedrake:
        raise pytest.UsageError("Cannot run both --skip and only options for newer firedrake")

    selected_items = []
    deselected = []

    for item in items:
        is_slow = "slow" in item.keywords
        is_high_memory = "high_memory" in item.keywords
        is_older_firedrake = "older_firedrake" in item.keywords
        is_newer_firedrake = "newer_firedrake" in item.keywords
        if only_slow and not is_slow:
            deselected.append(item)
        elif skip_slow and is_slow:
            deselected.append(item)
        elif skip_high_memory and is_high_memory:
            deselected.append(item)
        elif only_high_memory and not is_high_memory:
            deselected.append(item)
        elif only_older_firedrake and not is_older_firedrake:
            deselected.append(item)
        elif skip_older_firedrake and is_older_firedrake:
            deselected.append(item)
        elif only_newer_firedrake and not is_newer_firedrake:
            deselected.append(item)
        elif skip_newer_firedrake and is_newer_firedrake:
            deselected.append(item)
        else:
            selected_items.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected_items
