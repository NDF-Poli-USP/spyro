"""Exercise campaign validation and completion tracking without running a PDE."""

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

_PATH = (Path(__file__).resolve().parents[2] / "demos/elastic_fwi_automated_adjoint"
         / "run_proximal_cases.py")
_SPEC = importlib.util.spec_from_file_location("proximal_campaign", _PATH)
campaign = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(campaign)


@pytest.mark.parametrize("overrides", [
    {"ITERS": "0"}, {"OUTER": "0"}, {"OUTER": "11"}, {"OUTER": "3"},
    {"ITERS": "2.5"}, {"JOBS": "0"}, {"RANKS": "2"},
    {"STEP": "0"}, {"STEP": "-1"}, {"STEP": "nan"}, {"STEP": "inf"},
])
def test_invalid_campaign_settings(overrides: dict) -> None:
    """Reject invalid configurations before creating output directories.

    Parameters
    ----------
    overrides : dict
        Invalid environment settings.
    """
    with pytest.raises(ValueError):
        campaign.settings(overrides)


def test_campaign_case_coverage() -> None:
    """Cover the baseline, five variants and two restart controls.

    Notes
    -----
    Every case has the same maximum number of TAO iterations per stage.
    """
    config = campaign.settings({})
    assert len(campaign.CASES) == 8
    assert config["jobs"] == 1
    for name in campaign.CASES:
        arguments, inner = campaign.case_arguments(name, config)
        proximal = arguments.get("proximal", {})
        assert inner * proximal.get("outer_iterations", 1) == config["iters"]
        if name.endswith("restart"):
            assert proximal["scales"] == [0.0, 0.0]


@pytest.mark.parametrize("name, latent, kind", [
    ("bqnls_physical", False, None), ("bqnls_latent", True, None),
    ("l2_proximal_physical", False, "l2"), ("l2_proximal_latent", True, "l2"),
    ("bregman_proximal_physical", False, "bregman"),
    ("bregman_proximal_latent", True, "bregman"),
    ("bqnls_physical_restart", False, "l2"), ("bqnls_latent_restart", True, "l2"),
])
def test_case_names_select_the_intended_method(name: str, latent: bool, kind: str | None) -> None:
    """Preserve solver semantics under the descriptive case names.

    Parameters
    ----------
    name : str
        Public case identifier.
    latent : bool
        Expected optimization coordinates.
    kind : str or None
        Expected proximal divergence, or no proximal subproblem.
    """
    arguments, _ = campaign.case_arguments(name, campaign.settings({}))
    assert arguments["latent"] is latent
    assert arguments.get("proximal", {}).get("kind") == kind


@pytest.fixture
def fake_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Replace the converter and MPI launcher with deterministic test doubles.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated output directory.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace process creation.

    Returns
    -------
    tuple
        Campaign configuration and mutable process/output controls.
    """
    config = campaign.settings({"DEMO_DIR": str(tmp_path)})
    state = {"calls": 0, "exit_code": 0, "outputs": True, "log": ""}

    def run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
        """Generate a demo or emulate an MPI run.

        Parameters
        ----------
        command : list
            Converter or MPI command.
        **kwargs : object
            Process arguments, including the run directory and log stream.

        Returns
        -------
        subprocess.CompletedProcess
            Requested process exit status.
        """
        if "pylit" in command:
            Path(command[-1]).write_text("controls = fwi.run_fwi(\n    maxiter=maxiter,\n)\n")
        else:
            state["calls"] += 1
            directory = Path(kwargs["cwd"])
            kwargs["stdout"].write(state["log"])
            if state["outputs"]:
                (directory / "result.npy").write_bytes(b"test result")
                (directory / "control_end.pvd").write_text(
                    '<VTKFile><Collection><DataSet file="end.pvtu"/></Collection></VTKFile>')
                (directory / "end.pvtu").write_text(
                    '<VTKFile><PUnstructuredGrid><Piece Source="end.vtu"/></PUnstructuredGrid></VTKFile>')
                (directory / "end.vtu").write_text("test field")
        return subprocess.CompletedProcess(command, state["exit_code"])

    monkeypatch.setattr(campaign.subprocess, "run", run)
    return config, state


def test_campaign_reuse_and_configuration_identity(fake_campaign: tuple) -> None:
    """Reuse only verified output from identical parameters and source.

    Parameters
    ----------
    fake_campaign : tuple
        Configuration and fake process controls.
    """
    config, state = fake_campaign
    source = {"source_sha256": "first"}
    directory = campaign.run_case("l2_proximal_physical", config, source)
    assert json.loads((directory / "status.json").read_text())["status"] == "completed"
    campaign.run_case("l2_proximal_physical", config, source)
    assert state["calls"] == 1
    changed = campaign.run_case("l2_proximal_physical", {**config, "step": 50.0}, source)
    assert changed != directory
    changed_source = campaign.run_case("l2_proximal_physical", config, {"source_sha256": "second"})
    assert changed_source != directory
    (directory / "end.vtu").unlink()
    with pytest.raises(RuntimeError, match="Incomplete or failed"):
        campaign.run_case("l2_proximal_physical", config, source)


@pytest.mark.parametrize("failure", ["process", "missing_outputs", "tao"])
def test_campaign_failure_is_not_completion(fake_campaign: tuple, failure: str) -> None:
    """Keep failed attempts and allow an explicit fresh run tag.

    Parameters
    ----------
    fake_campaign : tuple
        Configuration and fake process controls.
    failure : str
        Whether the process fails, omits artifacts or reports a TAO failure.
    """
    config, state = fake_campaign
    state.update(exit_code=1 if failure == "process" else 0,
                 outputs=failure != "missing_outputs",
                 log="reason: DIVERGED_LS_FAILURE" if failure == "tao" else "")
    with pytest.raises(RuntimeError, match="failed or missing"):
        campaign.run_case("bqnls_latent", config, {})
    directory = next(Path(config["output"]).iterdir())
    assert json.loads((directory / "status.json").read_text())["status"] == "failed"
    state.update(exit_code=0, outputs=True, log="reason: DIVERGED_MAXITS")
    with pytest.raises(RuntimeError, match="Incomplete or failed"):
        campaign.run_case("bqnls_latent", config, {})
    retry = campaign.run_case("bqnls_latent", {**config, "run_tag": "retry"}, {})
    status = json.loads((retry / "status.json").read_text())
    assert status["status"] == "completed"
    assert status["tao_termination_reasons"] == ["DIVERGED_MAXITS"]


def test_campaign_dry_run(fake_campaign: tuple) -> None:
    """Leave output and processes untouched during a configuration preview.

    Parameters
    ----------
    fake_campaign : tuple
        Configuration and fake process controls.
    """
    config, state = fake_campaign
    directory = campaign.run_case("bqnls_physical", config, {}, dry_run=True)
    assert not directory.exists()
    assert state["calls"] == 0
