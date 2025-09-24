import pytest


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


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "high_memory: mark test as requiring high memory")


def pytest_collection_modifyitems(config, items):
    skip_slow = config.getoption("--skip-slow")
    only_slow = config.getoption("--only-slow")
    skip_high_memory = config.getoption("--skip-high-memory")
    only_high_memory = config.getoption("--only-high-memory")

    if skip_slow and only_slow:
        raise pytest.UsageError("Cannot use both --skip-slow and --only-slow")
    
    if skip_high_memory and only_slow:
        raise pytest.UsageError("Cannot run both --skip and only options for high memory")

    selected_items = []
    deselected = []

    for item in items:
        is_slow = "slow" in item.keywords
        is_high_memory = "high_memory" in item.keywords
        if only_slow and not is_slow:
            deselected.append(item)
        elif skip_slow and is_slow:
            deselected.append(item)
        elif skip_high_memory and is_high_memory:
            deselected.append(item)
        elif only_high_memory and not is_high_memory:
            deselected.append(item)
        else:
            selected_items.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected_items
