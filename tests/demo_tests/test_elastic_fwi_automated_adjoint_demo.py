"""Run the isotropic elastic FWI demo, coarsened, on 3 ranks.

The test pulls the code out of the demo's ``.. code-block:: python`` blocks
itself (no pylit needed), so it runs the demo as committed, except for the
three settings that fix its cost, coarsened to keep the test to minutes.
Run it with one process per shot, as the demo asks for::

    mpiexec -n 3 pytest tests/demo_tests/test_elastic_fwi_automated_adjoint_demo.py
"""
from pathlib import Path
import re

import firedrake as fire
import numpy as np
import pytest

import spyro


DEMO = (
    Path(__file__).resolve().parents[2]
    / "demos"
    / "elastic_fwi_automated_adjoint"
    / "elastic_fwi_automated_adjoint.py.rst"
)
CODE_BLOCK_MARKER = ".. code-block:: python"
CODE_INDENT = " " * 4
# The demo's settings, and what the test runs with instead: five-by-five
# elements, a time step at their stability limit, and two iterations.
COARSE_SETTINGS = {"edge_length": "0.2", "dt": "0.0025", "maxiter": "2"}


def extract_code(path: Path) -> str:
    """Return the Python of the code blocks of a literate demo, in order.

    A block starts at a ``.. code-block:: python`` line and runs through the
    indented lines after it; the first non-blank, unindented line ends it.

    Parameters
    ----------
    path : pathlib.Path
        The ``.py.rst`` file.

    Returns
    -------
    str
        The extracted program, dedented.
    """
    code = []
    in_block = False
    for line in path.read_text().splitlines():
        if line.strip() == CODE_BLOCK_MARKER:
            in_block = True
            continue
        if not in_block:
            continue
        if line.startswith(CODE_INDENT):
            code.append(line[len(CODE_INDENT):])
        elif not line.strip():
            code.append("")
        else:
            in_block = False
    return "\n".join(code) + "\n"


def coarsen(code: str) -> str:
    """Replace the demo's cost-setting assignments by the test's values.

    Parameters
    ----------
    code : str
        The demo's program.

    Returns
    -------
    str
        The program with each name in :data:`COARSE_SETTINGS` reassigned.

    Raises
    ------
    AssertionError
        If a setting is not assigned exactly once at the top level, which is
        the shape the substitution relies on.
    """
    for name, value in COARSE_SETTINGS.items():
        code, count = re.subn(
            rf"^{name} = .*$", f"{name} = {value}", code, flags=re.MULTILINE,
        )
        assert count == 1, f"expected one top-level assignment of {name}"
    return code


@pytest.mark.newer_firedrake
@pytest.mark.parallel(3)
def test_elastic_fwi_automated_adjoint_demo(tmp_path, monkeypatch):
    """The demo runs through, and moves the model the way it says it does."""
    # Every rank works in the same directory, so the files the demo writes
    # land in one place rather than in one temporary directory per process.
    run_directory = fire.COMM_WORLD.bcast(str(tmp_path), root=0)
    monkeypatch.chdir(run_directory)

    namespace = {"__name__": "elastic_fwi_automated_adjoint_demo"}
    exec(compile(coarsen(extract_code(DEMO)), str(DEMO), "exec"), namespace)

    fwi = namespace["fwi"]
    assert isinstance(fwi.wave, spyro.IsotropicWave)
    assert fwi.comm.ensemble_comm.size == 3, "one ensemble member per shot"

    # The gradient computed by hand at the starting model, one field per
    # control, is the L2 Riesz representer: a Function, not a Cofunction.
    gradient = namespace["gradient"]
    assert set(gradient) == namespace["control_parameters"]
    assert all(
        isinstance(field, fire.Function) for field in gradient.values()
    )
    # The Taylor test the demo runs on it converged at second order.
    assert namespace["rate"] > 1.9

    # Two controls, the velocities, in the order the bounds were given in.
    vp_result, vs_result = namespace["controls"]
    assert [vp_result.name(), vs_result.name()] == [
        "p_wave_velocity", "s_wave_velocity",
    ]
    for control, (low, high) in (
        (vp_result, namespace["vp_bounds"]),
        (vs_result, namespace["vs_bounds"]),
    ):
        assert control.dat.data_ro.min() >= low - 1e-10
        assert control.dat.data_ro.max() <= high + 1e-10

    # Both velocities moved from the starting model.
    space = vp_result.function_space()
    for result, start in (
        (vp_result, namespace["vp_start"]), (vs_result, namespace["vs_start"]),
    ):
        starting = fire.Function(space).interpolate(start).dat.data_ro
        assert not np.allclose(result.dat.data_ro, starting)

    # The misfit went down, and the S-wave velocity at the centre of the
    # circle moved from the starting value towards the true one.
    history = fwi.functional_history
    assert namespace["maxiter"] == 2, "the coarsened settings were applied"
    assert 2 <= len(history) <= namespace["maxiter"] + 1
    assert history[-1] < history[0]
    vs_start_center = float(
        fire.PointEvaluator(fwi.wave.mesh, [(namespace["center_z"], namespace["center_x"])])
        .evaluate(fire.Function(space).interpolate(namespace["vs_start"]))[0]
    )
    assert vs_start_center < namespace["vs_center"] < namespace["vs_circle"]

    # The first ensemble member draws the fields; every member saves the
    # record of its own shot, numbered.
    member = fwi.comm.ensemble_comm.rank
    figures = [f"elastic_fwi_observed_uz[{member}].png", f"elastic_fwi_observed_ux[{member}].png"]
    if namespace["first_member"]:
        figures += [
            "elastic_fwi_true_model.png",
            "elastic_fwi_starting_model.png",
            "elastic_fwi_models.png",
            "elastic_fwi_functional_history.png",
        ]
    if fwi.comm.comm.rank == 0:
        for figure in figures:
            assert (Path(run_directory) / figure).exists(), figure
