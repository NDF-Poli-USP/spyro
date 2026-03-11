import pytest
from pyadjoint import Tape, pause_annotation, set_working_tape
from pyadjoint.tape import get_working_tape


@pytest.fixture
def set_test_tape():
    """Use a fresh tape per test and clear it during teardown."""
    pause_annotation()
    with set_working_tape(Tape()):
        yield
        pause_annotation()
        tape = get_working_tape()
        if tape is not None:
            tape.clear_tape()
