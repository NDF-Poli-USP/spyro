from copy import deepcopy
import importlib.util
from pathlib import Path
import re

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "spyro" / "io" / "inversion_config.py"
)
SPEC = importlib.util.spec_from_file_location("inversion_config", MODULE_PATH)
inversion_config = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(inversion_config)

ConfigValidationError = inversion_config.ConfigValidationError
load_inversion_config = inversion_config.load_inversion_config


SAMPLE_CONFIG = {
    "mesh": {"mesh_file": "meshes/model.msh"},
    "time": {"initial_time": 0.0, "final_time": 2.0, "dt": 0.001},
    "geometry": {
        "sources": [(-0.10, 0.5)],
        "receivers": [(-0.10, 0.1), (-0.10, 0.2)],
    },
    "observed_data": {"path": "shots/observed_shots.npy"},
    "optimizer": {"name": "L-BFGS-B", "max_iterations": 25, "tolerance": 1e-6},
    "checkpoint": {"directory": "checkpoints", "every": 5},
    "output": {"directory": "results"},
}


def _remove_field(config, field_name):
    parts = field_name.split(".")
    parent = config
    for part in parts[:-1]:
        parent = parent[part]
    del parent[parts[-1]]


def test_load_complete_inversion_config_succeeds():
    inversion_config = load_inversion_config(deepcopy(SAMPLE_CONFIG))

    assert inversion_config.mesh.mesh_file == SAMPLE_CONFIG["mesh"]["mesh_file"]
    assert inversion_config.time.dt == SAMPLE_CONFIG["time"]["dt"]
    assert inversion_config.optimizer.name == SAMPLE_CONFIG["optimizer"]["name"]


@pytest.mark.parametrize(
    "missing_field",
    [
        "mesh",
        "time",
        "geometry",
        "observed_data",
        "optimizer",
        "checkpoint",
        "output",
        "mesh.mesh_file",
        "time.initial_time",
        "time.final_time",
        "time.dt",
        "geometry.sources",
        "geometry.receivers",
        "observed_data.path",
        "optimizer.name",
        "optimizer.max_iterations",
        "optimizer.tolerance",
        "checkpoint.directory",
        "checkpoint.every",
        "output.directory",
    ],
)
def test_missing_required_field_reports_field_name(missing_field):
    bad_config = deepcopy(SAMPLE_CONFIG)
    _remove_field(bad_config, missing_field)

    with pytest.raises(ConfigValidationError, match=re.escape(missing_field)):
        load_inversion_config(bad_config)
