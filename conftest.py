import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skip-slow", action="store_true", default=False, help="skip tests marked as slow"
    )
    parser.addoption(
        "--only-slow", action="store_true", default=False, help="run only slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow")


def pytest_collection_modifyitems(config, items):
    skip_slow = config.getoption("--skip-slow")
    only_slow = config.getoption("--only-slow")

    if skip_slow and only_slow:
        raise pytest.UsageError("Cannot use both --skip-slow and --only-slow")

    selected_items = []
    deselected = []

    for item in items:
        is_slow = "slow" in item.keywords
        if only_slow and not is_slow:
            deselected.append(item)
        elif skip_slow and is_slow:
            deselected.append(item)
        else:
            selected_items.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected_items
