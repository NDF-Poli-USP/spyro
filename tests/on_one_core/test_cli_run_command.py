import json
import subprocess
import sys

import numpy as np


SAMPLE_CONFIG = {
    "mesh": {"mesh_file": "meshes/model.msh"},
    "time": {"initial_time": 0.0, "final_time": 2.0, "dt": 0.001},
    "geometry": {
        "sources": [[-0.10, 0.5]],
        "receivers": [[-0.10, 0.1], [-0.10, 0.2]],
    },
    "observed_data": {"path": "shots/observed_shots.npy"},
    "optimizer": {"name": "L-BFGS-B", "max_iterations": 25, "tolerance": 1e-6},
    "checkpoint": {"directory": "checkpoints", "every": 5},
    "output": {"directory": "results"},
}
SNAPSHOT_FILENAME = "resolved_inversion_config.json"
INITIAL_OBJECTIVE_FILENAME = "initial_objective.json"


def _write_config_and_required_inputs(
    tmp_path,
    *,
    create_mesh=True,
    create_observed_data=True,
    config_data=None,
):
    if config_data is None:
        config_data = SAMPLE_CONFIG

    mesh_path = tmp_path / config_data["mesh"]["mesh_file"]
    observed_data_path = tmp_path / config_data["observed_data"]["path"]
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    observed_data_path.parent.mkdir(parents=True, exist_ok=True)

    if create_mesh:
        mesh_path.write_text("mesh", encoding="utf-8")
    if create_observed_data:
        time_samples = (
            int(
                np.floor(
                    (config_data["time"]["final_time"] - config_data["time"]["initial_time"])
                    / config_data["time"]["dt"]
                )
            )
            + 1
        )
        observed_data = np.zeros(
            (
                len(config_data["geometry"]["sources"]),
                time_samples,
                len(config_data["geometry"]["receivers"]),
            ),
            dtype=float,
        )
        np.save(observed_data_path, observed_data)

    config_path = tmp_path / "inversion.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")
    return config_path


def test_cli_run_accepts_config_and_exits_zero(tmp_path):
    config_path = _write_config_and_required_inputs(tmp_path)

    completed = subprocess.run(
        [sys.executable, "-m", "spyro_cli", "run", "--config", str(config_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "Starting non-interactive run with config" in completed.stdout
    assert "Iteration 0 objective J:" in completed.stdout
    assert (
        tmp_path / SAMPLE_CONFIG["output"]["directory"] / INITIAL_OBJECTIVE_FILENAME
    ).exists()


def test_cli_run_fails_fast_when_mesh_path_missing(tmp_path):
    config_path = _write_config_and_required_inputs(tmp_path, create_mesh=False)

    completed = subprocess.run(
        [sys.executable, "-m", "spyro_cli", "run", "--config", str(config_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "mesh.mesh_file" in completed.stderr
    assert "does not exist" in completed.stderr
    assert "Update 'mesh.mesh_file'" in completed.stderr


def test_cli_run_fails_fast_when_observed_data_path_missing(tmp_path):
    config_path = _write_config_and_required_inputs(
        tmp_path, create_observed_data=False
    )

    completed = subprocess.run(
        [sys.executable, "-m", "spyro_cli", "run", "--config", str(config_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "observed_data.path" in completed.stderr
    assert "does not exist" in completed.stderr
    assert "Update 'observed_data.path'" in completed.stderr


def test_cli_run_rejects_unstable_timestep_before_forward_solve(tmp_path):
    unstable_config = json.loads(json.dumps(SAMPLE_CONFIG))
    unstable_config["time"]["dt"] = 1.5
    config_path = _write_config_and_required_inputs(
        tmp_path, config_data=unstable_config
    )

    completed = subprocess.run(
        [sys.executable, "-m", "spyro_cli", "run", "--config", str(config_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "before forward solve" in completed.stderr
    assert "dt=1.5" in completed.stderr
    assert "allowed bound=1.0" in completed.stderr


def test_cli_run_writes_resolved_snapshot_reusable_for_reproduction(tmp_path):
    config_path = _write_config_and_required_inputs(tmp_path)

    first_run = subprocess.run(
        [sys.executable, "-m", "spyro_cli", "run", "--config", str(config_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert first_run.returncode == 0
    snapshot_path = tmp_path / SAMPLE_CONFIG["output"]["directory"] / SNAPSHOT_FILENAME
    assert snapshot_path.exists()

    first_snapshot = snapshot_path.read_text(encoding="utf-8")
    snapshot_data = json.loads(first_snapshot)
    assert snapshot_data["mesh"]["mesh_file"] == str(
        (tmp_path / SAMPLE_CONFIG["mesh"]["mesh_file"]).resolve()
    )
    assert snapshot_data["observed_data"]["path"] == str(
        (tmp_path / SAMPLE_CONFIG["observed_data"]["path"]).resolve()
    )
    assert snapshot_data["checkpoint"]["directory"] == str(
        (tmp_path / SAMPLE_CONFIG["checkpoint"]["directory"]).resolve()
    )
    assert snapshot_data["output"]["directory"] == str(
        (tmp_path / SAMPLE_CONFIG["output"]["directory"]).resolve()
    )

    second_run = subprocess.run(
        [sys.executable, "-m", "spyro_cli", "run", "--config", str(snapshot_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert second_run.returncode == 0
    assert snapshot_path.read_text(encoding="utf-8") == first_snapshot


def test_cli_run_initial_objective_is_deterministic_for_fixed_inputs(tmp_path):
    config_path = _write_config_and_required_inputs(tmp_path)
    objective_path = tmp_path / SAMPLE_CONFIG["output"]["directory"] / INITIAL_OBJECTIVE_FILENAME
    objective_values = []

    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "-m", "spyro_cli", "run", "--config", str(config_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0
        persisted_objective = json.loads(objective_path.read_text(encoding="utf-8"))
        objective_values.append(float(persisted_objective["objective"]))

    first_objective, second_objective = objective_values
    relative_difference = abs(first_objective - second_objective) / max(
        abs(first_objective), abs(second_objective), 1.0
    )
    assert relative_difference <= 1.0e-12
