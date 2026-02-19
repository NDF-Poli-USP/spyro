from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
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
load_inversion_config_file = inversion_config.load_inversion_config_file
run_acoustic_forward_modeling_from_config = (
    inversion_config.run_acoustic_forward_modeling_from_config
)
load_observed_data_into_internal_trace_structure = (
    inversion_config.load_observed_data_into_internal_trace_structure
)
compute_residual_traces = inversion_config.compute_residual_traces
compute_scalar_objective_from_residual_traces = (
    inversion_config.compute_scalar_objective_from_residual_traces
)
run_initial_forward_objective_evaluation = (
    inversion_config.run_initial_forward_objective_evaluation
)
define_velocity_model_as_primary_control_variable = (
    inversion_config.define_velocity_model_as_primary_control_variable
)
tape_misfit_functional_for_automatic_differentiation = (
    inversion_config.tape_misfit_functional_for_automatic_differentiation
)
compute_gradient_norm = inversion_config.compute_gradient_norm
compute_and_expose_gradient_norm_each_iteration = (
    inversion_config.compute_and_expose_gradient_norm_each_iteration
)
build_optimizer_factory = inversion_config.build_optimizer_factory


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


def test_load_yaml_and_json_configs_normalize_identically(tmp_path):
    json_path = tmp_path / "inversion.json"
    yaml_path = tmp_path / "inversion.yaml"

    json_path.write_text(json.dumps(SAMPLE_CONFIG), encoding="utf-8")
    yaml_path.write_text(
        """
mesh:
  mesh_file: meshes/model.msh
time:
  initial_time: 0.0
  final_time: 2.0
  dt: 0.001
geometry:
  sources:
    - [-0.1, 0.5]
  receivers:
    - [-0.1, 0.1]
    - [-0.1, 0.2]
observed_data:
  path: shots/observed_shots.npy
optimizer:
  name: L-BFGS-B
  max_iterations: 25
  tolerance: 1.0e-6
checkpoint:
  directory: checkpoints
  every: 5
output:
  directory: results
""".strip(),
        encoding="utf-8",
    )

    assert load_inversion_config_file(json_path) == load_inversion_config_file(yaml_path)


@pytest.mark.parametrize(
    "missing_field",
    [
        "mesh",
        "time",
        "geometry",
        "observed_data",
        "checkpoint",
        "output",
        "mesh.mesh_file",
        "time.initial_time",
        "time.final_time",
        "time.dt",
        "geometry.sources",
        "geometry.receivers",
        "observed_data.path",
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


def test_forward_modeling_from_config_outputs_traces_for_all_shots_and_receivers(
    tmp_path,
):
    config = deepcopy(SAMPLE_CONFIG)
    config["geometry"]["sources"] = [(-0.1, 0.4), (-0.1, 0.8), (-0.1, 1.2)]
    config["geometry"]["receivers"] = [(-0.1, 0.2), (-0.1, 0.5), (-0.1, 0.9)]

    mesh_path = tmp_path / config["mesh"]["mesh_file"]
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.write_text("mesh", encoding="utf-8")
    config_path = tmp_path / "inversion.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    normalized_config = load_inversion_config(config)

    class FakeWave:
        def __init__(self, dictionary):
            self.dictionary = dictionary
            self.forward_solution_receivers = None
            self._shot_traces = []

        def set_initial_velocity_model(self, constant):
            self.velocity_constant = constant

        def forward_solve(self):
            receiver_count = len(self.dictionary["acquisition"]["receiver_locations"])
            shot_count = len(self.dictionary["acquisition"]["source_locations"])
            self._shot_traces = [
                np.full((5, receiver_count), shot_index + 1.0)
                for shot_index in range(shot_count)
            ]
            self.forward_solution_receivers = self._shot_traces[-1]

    def fake_switch_shot(wave, source_index):
        wave.forward_solution_receivers = wave._shot_traces[source_index]

    traces = run_acoustic_forward_modeling_from_config(
        normalized_config,
        config_path,
        wave_factory=FakeWave,
        switch_shot=fake_switch_shot,
        cleanup_tmp_files=lambda wave: None,
    )

    assert traces.shape == (3, 5, 3)
    assert np.allclose(traces[0], 1.0)
    assert np.allclose(traces[1], 2.0)
    assert np.allclose(traces[2], 3.0)


def test_load_observed_data_into_internal_trace_structure_single_shot(tmp_path):
    config = deepcopy(SAMPLE_CONFIG)
    observed_data = np.arange(10, dtype=float).reshape(5, 2)

    observed_data_path = tmp_path / config["observed_data"]["path"]
    observed_data_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(observed_data_path, observed_data)

    config_path = tmp_path / "inversion.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    normalized_config = load_inversion_config(config)

    loaded_observed_data = load_observed_data_into_internal_trace_structure(
        normalized_config,
        config_path,
        expected_shape=(1, 5, 2),
    )

    assert loaded_observed_data.shape == (1, 5, 2)
    assert np.allclose(loaded_observed_data[0], observed_data)


@pytest.mark.parametrize(
    ("axis_name", "expected_shape", "actual_shape", "expected_size", "actual_size"),
    [
        ("shot", (3, 5, 2), (2, 5, 2), 3, 2),
        ("time", (2, 5, 2), (2, 4, 2), 5, 4),
        ("receiver", (2, 5, 3), (2, 5, 2), 3, 2),
    ],
)
def test_load_observed_data_into_internal_trace_structure_dimension_mismatch(
    tmp_path,
    axis_name,
    expected_shape,
    actual_shape,
    expected_size,
    actual_size,
):
    config = deepcopy(SAMPLE_CONFIG)
    observed_data = np.zeros(actual_shape, dtype=float)

    observed_data_path = tmp_path / config["observed_data"]["path"]
    observed_data_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(observed_data_path, observed_data)

    config_path = tmp_path / "inversion.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    normalized_config = load_inversion_config(config)

    expected_error = (
        f"Observed data {axis_name} dimension mismatch: "
        f"expected {expected_size}, actual {actual_size}."
    )
    with pytest.raises(ConfigValidationError, match=re.escape(expected_error)):
        load_observed_data_into_internal_trace_structure(
            normalized_config,
            config_path,
            expected_shape=expected_shape,
        )


def test_compute_residual_traces_matches_synthetic_and_observed_indexing():
    synthetic_traces = np.arange(24, dtype=float).reshape(2, 4, 3)
    observed_traces = synthetic_traces + np.array(
        [
            [
                [0.5, 1.0, 1.5],
                [2.0, 2.5, 3.0],
                [3.5, 4.0, 4.5],
                [5.0, 5.5, 6.0],
            ],
            [
                [6.5, 7.0, 7.5],
                [8.0, 8.5, 9.0],
                [9.5, 10.0, 10.5],
                [11.0, 11.5, 12.0],
            ],
        ],
        dtype=float,
    )

    residual_traces = compute_residual_traces(synthetic_traces, observed_traces)

    assert residual_traces.shape == synthetic_traces.shape
    assert residual_traces.shape == observed_traces.shape
    assert np.allclose(residual_traces, observed_traces - synthetic_traces)


@pytest.mark.parametrize(
    ("axis_name", "synthetic_shape", "observed_shape", "synthetic_size", "observed_size"),
    [
        ("shot", (3, 5, 2), (2, 5, 2), 3, 2),
        ("time", (2, 5, 2), (2, 4, 2), 5, 4),
        ("receiver", (2, 5, 3), (2, 5, 2), 3, 2),
    ],
)
def test_compute_residual_traces_dimension_mismatch(
    axis_name,
    synthetic_shape,
    observed_shape,
    synthetic_size,
    observed_size,
):
    synthetic_traces = np.zeros(synthetic_shape, dtype=float)
    observed_traces = np.zeros(observed_shape, dtype=float)
    expected_error = (
        f"Residual trace {axis_name} dimension mismatch: "
        f"synthetic {synthetic_size}, observed {observed_size}."
    )

    with pytest.raises(ConfigValidationError, match=re.escape(expected_error)):
        compute_residual_traces(synthetic_traces, observed_traces)


def test_compute_scalar_objective_from_residual_traces_matches_fixture_value():
    residual_traces = np.array(
        [
            [
                [1.0, 0.0],
                [2.0, 1.0],
                [3.0, 0.0],
            ],
            [
                [2.0, 1.0],
                [2.0, 0.0],
                [2.0, 1.0],
            ],
        ],
        dtype=float,
    )

    objective_value = compute_scalar_objective_from_residual_traces(
        residual_traces, dt=0.25
    )

    assert np.isfinite(objective_value)
    assert np.isclose(objective_value, 2.375, atol=1.0e-12, rtol=0.0)


def test_compute_scalar_objective_from_residual_traces_rejects_non_finite_values():
    residual_traces = np.array([[[1.0], [np.inf]]], dtype=float)

    with pytest.raises(
        ConfigValidationError,
        match=re.escape(
            "Residual traces must contain only finite values to compute objective J."
        ),
    ):
        compute_scalar_objective_from_residual_traces(residual_traces, dt=0.25)


def test_run_initial_forward_objective_evaluation_persists_iteration_zero_objective(
    tmp_path,
):
    config = deepcopy(SAMPLE_CONFIG)
    config["time"]["final_time"] = 0.3
    config["time"]["dt"] = 0.1

    observed_traces = np.array(
        [[[2.0, 1.0], [3.0, 2.0], [4.0, 3.0], [5.0, 4.0]]],
        dtype=float,
    )
    synthetic_traces = np.array(
        [[[1.5, 0.5], [2.5, 1.5], [3.5, 2.5], [4.5, 3.5]]],
        dtype=float,
    )

    mesh_path = tmp_path / config["mesh"]["mesh_file"]
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.write_text("mesh", encoding="utf-8")
    observed_data_path = tmp_path / config["observed_data"]["path"]
    observed_data_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(observed_data_path, observed_traces)
    config_path = tmp_path / "inversion.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    normalized_config = load_inversion_config(config)

    pipeline_result = run_initial_forward_objective_evaluation(
        normalized_config,
        config_path,
        forward_modeling_runner=lambda *_args, **_kwargs: synthetic_traces,
    )

    expected_objective = compute_scalar_objective_from_residual_traces(
        observed_traces - synthetic_traces,
        dt=config["time"]["dt"],
    )
    assert pipeline_result["iteration"] == 0
    assert np.isclose(pipeline_result["objective"], expected_objective)
    objective_path = Path(pipeline_result["objective_path"])
    assert objective_path.exists()

    persisted = json.loads(objective_path.read_text(encoding="utf-8"))
    assert persisted["iteration"] == 0
    assert np.isclose(persisted["objective"], expected_objective)


def test_load_inversion_config_defaults_initial_velocity_model():
    normalized_config = load_inversion_config(deepcopy(SAMPLE_CONFIG))

    assert normalized_config.initial_model.velocity_km_s == 1.5


def test_load_inversion_config_defaults_optimizer_when_section_missing():
    config = deepcopy(SAMPLE_CONFIG)
    del config["optimizer"]

    normalized_config = load_inversion_config(config)

    assert normalized_config.optimizer.name == "L-BFGS-B"
    assert normalized_config.optimizer.max_iterations == 25
    assert normalized_config.optimizer.tolerance == 1.0e-6


def test_load_inversion_config_preserves_optimizer_step_control_settings():
    config = deepcopy(SAMPLE_CONFIG)
    config["optimizer"]["step_control"] = {
        "initial_step_size": 0.25,
        "backtracking_factor": 0.75,
    }

    normalized_config = load_inversion_config(config)

    assert normalized_config.optimizer.step_control["initial_step_size"] == 0.25
    assert normalized_config.optimizer.step_control["backtracking_factor"] == 0.75


def test_build_optimizer_factory_defaults_to_lbfgsb_when_optimizer_key_missing():
    call_kwargs = {}

    def fake_minimize(_objective, _x0, **kwargs):
        call_kwargs.update(kwargs)
        return kwargs

    optimizer = build_optimizer_factory({}, minimize_callable=fake_minimize)
    optimizer(lambda _x: 0.0, np.array([1.0], dtype=float))

    assert optimizer.optimizer_name == "L-BFGS-B"
    assert call_kwargs["method"] == "L-BFGS-B"


def test_build_optimizer_factory_uses_configured_optimizer_override():
    call_kwargs = {}

    def fake_minimize(_objective, _x0, **kwargs):
        call_kwargs.update(kwargs)
        return kwargs

    optimizer = build_optimizer_factory(
        {"optimizer": {"name": "CG"}},
        minimize_callable=fake_minimize,
    )
    optimizer(lambda _x: 0.0, np.array([1.0], dtype=float))

    assert optimizer.optimizer_name == "CG"
    assert call_kwargs["method"] == "CG"


def test_build_optimizer_factory_applies_configured_stop_limits_to_backend():
    call_kwargs = {}

    def fake_minimize(_objective, _x0, **kwargs):
        call_kwargs.update(kwargs)
        return kwargs

    optimizer = build_optimizer_factory(
        {
            "optimizer": {
                "name": "CG",
                "max_iterations": 7,
                "tolerance": 1.0e-4,
            }
        },
        minimize_callable=fake_minimize,
    )
    optimizer(lambda _x: 0.0, np.array([1.0], dtype=float))

    assert call_kwargs["options"]["maxiter"] == 7
    assert call_kwargs["tol"] == pytest.approx(1.0e-4)


def test_build_optimizer_factory_step_control_changes_accepted_step_sizes():
    aggressive_config = deepcopy(SAMPLE_CONFIG)
    aggressive_config["optimizer"]["step_control"] = {
        "initial_step_size": 1.0,
        "backtracking_factor": 0.5,
        "armijo_constant": 1.0e-4,
    }
    conservative_config = deepcopy(SAMPLE_CONFIG)
    conservative_config["optimizer"]["step_control"] = {
        "initial_step_size": 0.2,
        "backtracking_factor": 0.5,
        "armijo_constant": 1.0e-4,
    }

    def quadratic_objective(model_vector):
        model_vector = np.asarray(model_vector, dtype=float)
        objective_value = float(np.dot(model_vector, model_vector))
        gradient_vector = 2.0 * model_vector
        return objective_value, gradient_vector

    aggressive_optimizer = build_optimizer_factory(
        load_inversion_config(aggressive_config),
        minimize_callable=lambda *_args, **_kwargs: None,
    )
    conservative_optimizer = build_optimizer_factory(
        load_inversion_config(conservative_config),
        minimize_callable=lambda *_args, **_kwargs: None,
    )

    aggressive_result = aggressive_optimizer(
        quadratic_objective,
        np.array([1.0], dtype=float),
        maxiter=1,
    )
    conservative_result = conservative_optimizer(
        quadratic_objective,
        np.array([1.0], dtype=float),
        maxiter=1,
    )

    assert aggressive_result["accepted_step_sizes"]
    assert conservative_result["accepted_step_sizes"]
    assert (
        aggressive_result["accepted_step_sizes"][0]
        > conservative_result["accepted_step_sizes"][0]
    )


def test_build_optimizer_factory_step_control_stops_on_tolerance_first(caplog):
    config = deepcopy(SAMPLE_CONFIG)
    config["optimizer"]["max_iterations"] = 1
    config["optimizer"]["tolerance"] = 3.0
    config["optimizer"]["step_control"] = {
        "initial_step_size": 1.0,
        "backtracking_factor": 0.5,
        "armijo_constant": 1.0e-4,
    }

    def quadratic_objective(model_vector):
        model_vector = np.asarray(model_vector, dtype=float)
        objective_value = float(np.dot(model_vector, model_vector))
        gradient_vector = 2.0 * model_vector
        return objective_value, gradient_vector

    optimizer = build_optimizer_factory(
        load_inversion_config(config),
        minimize_callable=lambda *_args, **_kwargs: None,
    )

    with caplog.at_level("INFO", logger=inversion_config.__name__):
        result = optimizer(quadratic_objective, np.array([1.0], dtype=float))

    assert result["success"] is True
    assert result["nit"] == 0
    assert result["stop_reason"] == "tolerance"
    assert result["message"] == "Gradient tolerance reached."
    assert "Optimizer stop reason: tolerance" in caplog.text


def test_build_optimizer_factory_step_control_stops_on_max_iterations(caplog):
    config = deepcopy(SAMPLE_CONFIG)
    config["optimizer"]["max_iterations"] = 1
    config["optimizer"]["tolerance"] = 1.0e-12
    config["optimizer"]["step_control"] = {
        "initial_step_size": 1.0,
        "backtracking_factor": 0.5,
        "armijo_constant": 1.0e-4,
    }

    def quadratic_objective(model_vector):
        model_vector = np.asarray(model_vector, dtype=float)
        objective_value = float(np.dot(model_vector, model_vector))
        gradient_vector = 2.0 * model_vector
        return objective_value, gradient_vector

    optimizer = build_optimizer_factory(
        load_inversion_config(config),
        minimize_callable=lambda *_args, **_kwargs: None,
    )

    with caplog.at_level("INFO", logger=inversion_config.__name__):
        result = optimizer(quadratic_objective, np.array([1.0], dtype=float))

    assert result["success"] is False
    assert result["stop_reason"] == "max_iterations"
    assert result["message"] == "Maximum iterations reached before convergence."
    assert "Optimizer stop reason: max_iterations" in caplog.text


def test_build_optimizer_factory_step_control_records_iteration_metrics_in_memory():
    config = deepcopy(SAMPLE_CONFIG)
    config["optimizer"]["max_iterations"] = 3
    config["optimizer"]["tolerance"] = 1.0e-20
    config["optimizer"]["step_control"] = {
        "initial_step_size": 0.25,
        "backtracking_factor": 0.5,
        "armijo_constant": 1.0e-4,
    }

    def quadratic_objective(model_vector):
        model_vector = np.asarray(model_vector, dtype=float)
        objective_value = float(np.dot(model_vector, model_vector))
        gradient_vector = 2.0 * model_vector
        return objective_value, gradient_vector

    optimizer = build_optimizer_factory(
        load_inversion_config(config),
        minimize_callable=lambda *_args, **_kwargs: None,
    )
    result = optimizer(quadratic_objective, np.array([1.0], dtype=float))

    iteration_metrics = result["iteration_metrics"]
    assert len(iteration_metrics) == result["nit"] == 3
    required_fields = {"iteration", "objective", "gradient_norm", "step_size", "elapsed_time"}
    for expected_iteration, metrics_record in enumerate(iteration_metrics):
        assert set(metrics_record) == required_fields
        assert metrics_record["iteration"] == expected_iteration
        assert np.isfinite(metrics_record["objective"])
        assert np.isfinite(metrics_record["gradient_norm"])
        assert metrics_record["step_size"] > 0.0
        assert metrics_record["elapsed_time"] >= 0.0


def test_reference_synthetic_case_objective_reduces_by_thirty_percent_in_twenty_iterations():
    config = deepcopy(SAMPLE_CONFIG)
    config["optimizer"]["max_iterations"] = 20
    config["optimizer"]["tolerance"] = 1.0e-20
    config["optimizer"]["step_control"] = {
        "initial_step_size": 0.25,
        "backtracking_factor": 0.5,
        "armijo_constant": 1.0e-4,
    }

    def reference_synthetic_objective(model_vector):
        model_vector = np.asarray(model_vector, dtype=float)
        objective_value = 0.5 * float(np.dot(model_vector, model_vector))
        gradient_vector = model_vector
        return objective_value, gradient_vector

    initial_model = np.array([2.0, -1.0, 0.5], dtype=float)
    initial_objective, _ = reference_synthetic_objective(initial_model)

    optimizer = build_optimizer_factory(
        load_inversion_config(config),
        minimize_callable=lambda *_args, **_kwargs: None,
    )
    result = optimizer(reference_synthetic_objective, initial_model)

    objective_at_iteration_20 = float(result["fun"])
    objective_reduction = (
        initial_objective - objective_at_iteration_20
    ) / initial_objective

    assert result["stop_reason"] == "max_iterations"
    assert result["nit"] == 20
    assert objective_reduction >= 0.30


def test_define_velocity_model_as_primary_control_variable_uses_configured_initial_model(
    tmp_path,
):
    config = deepcopy(SAMPLE_CONFIG)
    config["initial_model"] = {"velocity_km_s": 2.75}
    config_path = tmp_path / "inversion.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    normalized_config = load_inversion_config(config)

    velocity_model = object()

    class FakeWave:
        def __init__(self, dictionary):
            self.dictionary = dictionary
            self.initial_velocity_model = None
            self.loaded_velocity = None

        def set_initial_velocity_model(self, constant):
            self.loaded_velocity = constant
            self.initial_velocity_model = velocity_model

    class FakeControl:
        def __init__(self, variable):
            self.variable = variable

    taped_functional = object()
    reduced_functional_calls = {}

    def fake_reduced_functional_factory(functional, control):
        reduced_functional_calls["functional"] = functional
        reduced_functional_calls["control"] = control
        return {"functional": functional, "control": control}

    result = define_velocity_model_as_primary_control_variable(
        normalized_config,
        config_path,
        wave_factory=FakeWave,
        control_factory=FakeControl,
        taped_functional=taped_functional,
        reduced_functional_factory=fake_reduced_functional_factory,
    )

    assert result["wave"].loaded_velocity == 2.75
    assert result["control"].variable is velocity_model
    assert result["taped_functional"] is taped_functional
    assert reduced_functional_calls["functional"] is taped_functional
    assert reduced_functional_calls["control"] is result["control"]


def test_tape_misfit_functional_for_automatic_differentiation_returns_gradient_field(
    tmp_path,
):
    config = deepcopy(SAMPLE_CONFIG)
    config["initial_model"] = {"velocity_km_s": 2.2}
    config_path = tmp_path / "inversion.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    normalized_config = load_inversion_config(config)

    velocity_model = object()
    taped_functional = object()
    gradient_field = np.array([3.0, 4.0], dtype=float)
    taping_callback_calls = {}

    class FakeWave:
        def __init__(self, dictionary):
            self.dictionary = dictionary
            self.initial_velocity_model = None

        def set_initial_velocity_model(self, constant):
            self.loaded_velocity = constant
            self.initial_velocity_model = velocity_model

    class FakeControl:
        def __init__(self, variable):
            self.variable = variable

    class FakeReducedFunctional:
        def __init__(self, functional, control):
            self.functional = functional
            self.control = control
            self.derivative_calls = 0

        def derivative(self):
            self.derivative_calls += 1
            return gradient_field

    def fake_functional_taping_callback(wave):
        taping_callback_calls["wave"] = wave
        return taped_functional

    result = tape_misfit_functional_for_automatic_differentiation(
        normalized_config,
        config_path,
        wave_factory=FakeWave,
        control_factory=FakeControl,
        functional_taping_callback=fake_functional_taping_callback,
        reduced_functional_factory=FakeReducedFunctional,
    )

    assert taping_callback_calls["wave"] is result["wave"]
    assert result["control"].variable is velocity_model
    assert result["taped_functional"] is taped_functional
    assert result["reduced_functional"].functional is taped_functional
    assert result["reduced_functional"].control is result["control"]
    assert result["reduced_functional"].derivative_calls == 1
    assert result["gradient"] is gradient_field
    assert np.isclose(result["gradient_norm"], 5.0)
    assert result["gradient_norm_history"] == [
        {"iteration": 0, "gradient_norm": result["gradient_norm"]}
    ]


def test_tape_misfit_functional_for_automatic_differentiation_requires_taping_input(
    tmp_path,
):
    config_path = tmp_path / "inversion.json"
    config_path.write_text(json.dumps(SAMPLE_CONFIG), encoding="utf-8")
    normalized_config = load_inversion_config(SAMPLE_CONFIG)

    with pytest.raises(
        ConfigValidationError, match=re.escape("functional_taping_callback")
    ):
        tape_misfit_functional_for_automatic_differentiation(
            normalized_config,
            config_path,
            wave_factory=lambda dictionary: dictionary,
            control_factory=lambda variable: variable,
        )


def test_compute_and_expose_gradient_norm_each_iteration_records_finite_values():
    gradient_norm_history = []
    gradients = [
        np.array([3.0, 4.0], dtype=float),
        np.array([5.0, 12.0], dtype=float),
        np.array([8.0, 15.0], dtype=float),
    ]

    for expected_iteration, gradient_field in enumerate(gradients):
        record = compute_and_expose_gradient_norm_each_iteration(
            gradient_field,
            gradient_norm_history=gradient_norm_history,
        )
        assert record["iteration"] == expected_iteration
        assert np.isfinite(record["gradient_norm"])
        assert record == gradient_norm_history[expected_iteration]

    assert [record["iteration"] for record in gradient_norm_history] == [0, 1, 2]
    assert np.all(np.isfinite([record["gradient_norm"] for record in gradient_norm_history]))
    assert np.allclose(
        [record["gradient_norm"] for record in gradient_norm_history],
        [5.0, 13.0, 17.0],
    )


def test_compute_gradient_norm_rejects_non_finite_values():
    with pytest.raises(ConfigValidationError, match=re.escape("non-finite")):
        compute_gradient_norm(np.array([1.0, np.nan], dtype=float))
