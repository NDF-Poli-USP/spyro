from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Union

import numpy as np


class ConfigValidationError(ValueError):
    """Raised when inversion configuration validation fails."""


@dataclass(frozen=True)
class MeshConfig:
    mesh_file: str


@dataclass(frozen=True)
class TimeConfig:
    initial_time: Any
    final_time: Any
    dt: Any


@dataclass(frozen=True)
class GeometryConfig:
    sources: Any
    receivers: Any


@dataclass(frozen=True)
class ObservedDataConfig:
    path: str


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    max_iterations: Any
    tolerance: Any


@dataclass(frozen=True)
class CheckpointConfig:
    directory: str
    every: Any


@dataclass(frozen=True)
class OutputConfig:
    directory: str


@dataclass(frozen=True)
class InitialModelConfig:
    velocity_km_s: Any


@dataclass(frozen=True)
class InversionConfig:
    mesh: MeshConfig
    time: TimeConfig
    geometry: GeometryConfig
    observed_data: ObservedDataConfig
    optimizer: OptimizerConfig
    checkpoint: CheckpointConfig
    output: OutputConfig
    initial_model: InitialModelConfig


_REQUIRED_FIELDS = {
    "mesh": ("mesh_file",),
    "time": ("initial_time", "final_time", "dt"),
    "geometry": ("sources", "receivers"),
    "observed_data": ("path",),
    "optimizer": ("name", "max_iterations", "tolerance"),
    "checkpoint": ("directory", "every"),
    "output": ("directory",),
}

_MAX_STARTUP_STABLE_DT = 1.0
_RESOLVED_CONFIG_SNAPSHOT_FILENAME = "resolved_inversion_config.json"
_DEFAULT_FORWARD_SOURCE_FREQUENCY_HZ = 5.0
_DEFAULT_FORWARD_VELOCITY_KM_S = 1.5
_INITIAL_OBJECTIVE_FILENAME = "initial_objective.json"
_DEFAULT_OPTIMIZER_NAME = "L-BFGS-B"
_DEFAULT_OPTIMIZER_MAX_ITERATIONS = 25
_DEFAULT_OPTIMIZER_TOLERANCE = 1.0e-6


def _require_fields(section: Mapping[str, Any], fields, section_name: str) -> None:
    for field in fields:
        field_name = f"{section_name}.{field}" if section_name else field
        if field not in section:
            raise ConfigValidationError(f"Missing required config field: {field_name}")


def _normalize_optimizer_name(optimizer_name: Any) -> str:
    if not isinstance(optimizer_name, str):
        raise ConfigValidationError(
            "Config field 'optimizer.name' must be a non-empty string."
        )
    normalized_name = optimizer_name.strip()
    if not normalized_name:
        raise ConfigValidationError(
            "Config field 'optimizer.name' must be a non-empty string."
        )
    return normalized_name


def _resolve_optimizer_config(config: Mapping[str, Any]) -> OptimizerConfig:
    optimizer_section = config.get("optimizer")
    if optimizer_section is None:
        optimizer_section = {}
    if not isinstance(optimizer_section, Mapping):
        raise ConfigValidationError("Config field 'optimizer' must be a mapping.")

    optimizer_name = optimizer_section.get("name", _DEFAULT_OPTIMIZER_NAME)
    max_iterations = optimizer_section.get(
        "max_iterations", _DEFAULT_OPTIMIZER_MAX_ITERATIONS
    )
    tolerance = optimizer_section.get("tolerance", _DEFAULT_OPTIMIZER_TOLERANCE)

    return OptimizerConfig(
        name=_normalize_optimizer_name(optimizer_name),
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def _load_config_mapping(config_path: Path) -> Mapping[str, Any]:
    suffix = config_path.suffix.lower()
    try:
        raw_content = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigValidationError(
            f"Unable to read inversion config file: {config_path}"
        ) from exc

    if suffix == ".json":
        try:
            config = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise ConfigValidationError(
                f"Invalid JSON inversion config file: {config_path}"
            ) from exc
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ConfigValidationError(
                "PyYAML is required to load YAML inversion config files."
            ) from exc
        try:
            config = yaml.safe_load(raw_content)
        except yaml.YAMLError as exc:
            raise ConfigValidationError(
                f"Invalid YAML inversion config file: {config_path}"
            ) from exc
    else:
        raise ConfigValidationError(
            "Unsupported inversion config file format. Use .json, .yaml, or .yml."
        )

    if not isinstance(config, Mapping):
        raise ConfigValidationError(
            "Inversion config file must define a mapping at the top level."
        )
    return config


def load_inversion_config(config: Mapping[str, Any]) -> InversionConfig:
    """Validate and normalize an inversion configuration mapping."""
    if not isinstance(config, Mapping):
        raise ConfigValidationError("Inversion config must be a mapping.")

    required_top_level_sections = tuple(
        section_name for section_name in _REQUIRED_FIELDS.keys() if section_name != "optimizer"
    )
    _require_fields(config, required_top_level_sections, "")

    for section_name, required_section_fields in _REQUIRED_FIELDS.items():
        section = config.get(section_name)
        if section is None:
            continue
        if not isinstance(section, Mapping):
            raise ConfigValidationError(f"Config field '{section_name}' must be a mapping.")
        if section_name != "optimizer":
            _require_fields(section, required_section_fields, section_name)

    initial_model = config.get("initial_model", {})
    if not isinstance(initial_model, Mapping):
        raise ConfigValidationError("Config field 'initial_model' must be a mapping.")
    optimizer = _resolve_optimizer_config(config)

    return InversionConfig(
        mesh=MeshConfig(mesh_file=config["mesh"]["mesh_file"]),
        time=TimeConfig(
            initial_time=config["time"]["initial_time"],
            final_time=config["time"]["final_time"],
            dt=config["time"]["dt"],
        ),
        geometry=GeometryConfig(
            sources=config["geometry"]["sources"],
            receivers=config["geometry"]["receivers"],
        ),
        observed_data=ObservedDataConfig(path=config["observed_data"]["path"]),
        optimizer=optimizer,
        checkpoint=CheckpointConfig(
            directory=config["checkpoint"]["directory"],
            every=config["checkpoint"]["every"],
        ),
        output=OutputConfig(directory=config["output"]["directory"]),
        initial_model=InitialModelConfig(
            velocity_km_s=initial_model.get(
                "velocity_km_s", _DEFAULT_FORWARD_VELOCITY_KM_S
            )
        ),
    )


def _resolve_config_relative_path(path_value: str, config_path: Path) -> Path:
    resolved_path = Path(path_value).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = config_path.parent / resolved_path
    return resolved_path


def _to_resolved_config_mapping(
    config: InversionConfig, config_path: Path
) -> Mapping[str, Any]:
    resolved_config = asdict(config)
    resolved_config["mesh"]["mesh_file"] = str(
        _resolve_config_relative_path(config.mesh.mesh_file, config_path)
    )
    resolved_config["observed_data"]["path"] = str(
        _resolve_config_relative_path(config.observed_data.path, config_path)
    )
    resolved_config["checkpoint"]["directory"] = str(
        _resolve_config_relative_path(config.checkpoint.directory, config_path)
    )
    resolved_config["output"]["directory"] = str(
        _resolve_config_relative_path(config.output.directory, config_path)
    )
    return resolved_config


def persist_resolved_config_snapshot(
    config: InversionConfig, config_path: Union[str, Path]
) -> Path:
    config_path_obj = Path(config_path).expanduser().resolve()
    output_directory = _resolve_config_relative_path(
        config.output.directory, config_path_obj
    )
    resolved_config = _to_resolved_config_mapping(config, config_path_obj)

    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigValidationError(
            f"Unable to create output directory for config snapshot: {output_directory}"
        ) from exc

    snapshot_path = output_directory / _RESOLVED_CONFIG_SNAPSHOT_FILENAME
    try:
        snapshot_path.write_text(
            json.dumps(resolved_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ConfigValidationError(
            f"Unable to write resolved config snapshot: {snapshot_path}"
        ) from exc

    return snapshot_path


def validate_startup_paths(
    config: InversionConfig, config_path: Union[str, Path]
) -> None:
    """Validate that startup input paths exist and point to files."""
    config_path_obj = Path(config_path)
    path_fields = (
        ("mesh.mesh_file", config.mesh.mesh_file),
        ("observed_data.path", config.observed_data.path),
    )
    for field_name, path_value in path_fields:
        resolved_path = _resolve_config_relative_path(path_value, config_path_obj)
        if not resolved_path.exists():
            raise ConfigValidationError(
                f"Configured path for '{field_name}' does not exist: {resolved_path}. "
                f"Update '{field_name}' to an existing file."
            )
        if not resolved_path.is_file():
            raise ConfigValidationError(
                f"Configured path for '{field_name}' must be a file: {resolved_path}. "
                f"Update '{field_name}' to point to a file path."
            )


def validate_stable_timestep_bounds(config: InversionConfig) -> None:
    """Validate a conservative startup bound for stable time stepping."""
    try:
        initial_time = float(config.time.initial_time)
        final_time = float(config.time.final_time)
        dt = float(config.time.dt)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(
            "Configured time bounds must be numeric for stability validation: "
            f"time.initial_time={config.time.initial_time}, "
            f"time.final_time={config.time.final_time}, time.dt={config.time.dt}."
        ) from exc

    if final_time <= initial_time:
        raise ConfigValidationError(
            f"Configured time window is invalid: time.initial_time={initial_time}, "
            f"time.final_time={final_time}. Ensure final_time is greater than initial_time."
        )
    if dt <= 0.0:
        raise ConfigValidationError(
            f"Configured time step is invalid: dt={dt}. Ensure 'time.dt' is positive."
        )

    allowed_bound = min(_MAX_STARTUP_STABLE_DT, final_time - initial_time)
    if dt > allowed_bound:
        raise ConfigValidationError(
            "Configured time step is unstable before forward solve: "
            f"dt={dt} exceeds allowed bound={allowed_bound}. "
            "Update 'time.dt' to be less than or equal to the allowed bound."
        )


def _normalize_geometry_points(field_name: str, points: Any) -> List[Tuple[float, ...]]:
    if not isinstance(points, (list, tuple)) or len(points) == 0:
        raise ConfigValidationError(
            f"Config field '{field_name}' must contain at least one coordinate point."
        )

    normalized_points = []
    expected_dimension = None
    for index, point in enumerate(points):
        if not isinstance(point, (list, tuple)):
            raise ConfigValidationError(
                f"Config field '{field_name}[{index}]' must be a coordinate sequence."
            )
        if expected_dimension is None:
            expected_dimension = len(point)
            if expected_dimension not in (2, 3):
                raise ConfigValidationError(
                    f"Config field '{field_name}' must use 2D or 3D coordinates."
                )
        elif len(point) != expected_dimension:
            raise ConfigValidationError(
                f"Config field '{field_name}[{index}]' has dimension {len(point)} "
                f"but expected {expected_dimension}."
            )

        try:
            normalized_points.append(tuple(float(value) for value in point))
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(
                f"Config field '{field_name}[{index}]' must be numeric coordinates."
            ) from exc

    return normalized_points


def _infer_domain_lengths(
    sources: List[Tuple[float, ...]], receivers: List[Tuple[float, ...]]
) -> Tuple[float, float, float, bool]:
    points = sources + receivers
    dimension = len(points[0])

    z_coordinates = [point[0] for point in points]
    x_coordinates = [point[1] for point in points]
    if any(value < 0.0 for value in x_coordinates):
        raise ConfigValidationError(
            "Config field 'geometry' has x-coordinates outside [0, Lx]."
        )

    has_negative_z = any(value < 0.0 for value in z_coordinates)
    has_positive_z = any(value > 0.0 for value in z_coordinates)
    if has_negative_z and has_positive_z:
        raise ConfigValidationError(
            "Config field 'geometry' mixes negative and positive z-coordinates. "
            "Use a single z-coordinate convention."
        )

    if has_negative_z or not has_positive_z:
        negative_z = True
        length_z = max(1.0, abs(min(z_coordinates)))
    else:
        negative_z = False
        length_z = max(1.0, max(z_coordinates))

    length_x = max(1.0, max(x_coordinates))
    if dimension == 2:
        length_y = 0.0
    else:
        y_coordinates = [point[2] for point in points]
        if any(value < 0.0 for value in y_coordinates):
            raise ConfigValidationError(
                "Config field 'geometry' has y-coordinates outside [0, Ly]."
            )
        length_y = max(1.0, max(y_coordinates))

    return length_z, length_x, length_y, negative_z


def _build_forward_model_dictionary(
    config: InversionConfig, config_path: Union[str, Path], source_frequency_hz: float
) -> Tuple[Dict[str, Any], List[Tuple[float, ...]], List[Tuple[float, ...]]]:
    source_locations = _normalize_geometry_points("geometry.sources", config.geometry.sources)
    receiver_locations = _normalize_geometry_points(
        "geometry.receivers", config.geometry.receivers
    )
    if len(source_locations[0]) != len(receiver_locations[0]):
        raise ConfigValidationError(
            "Config fields 'geometry.sources' and 'geometry.receivers' must have "
            "matching coordinate dimensions."
        )

    length_z, length_x, length_y, negative_z = _infer_domain_lengths(
        source_locations, receiver_locations
    )
    mesh_file = str(
        _resolve_config_relative_path(config.mesh.mesh_file, Path(config_path))
    )
    model_dictionary = {
        "options": {
            "cell_type": "Q",
            "variant": "lumped",
            "degree": 4,
            "dimension": len(source_locations[0]),
        },
        "parallelism": {
            "type": "spatial",
        },
        "mesh": {
            "Lz": length_z,
            "Lx": length_x,
            "Ly": length_y,
            "mesh_file": mesh_file,
            "mesh_type": "file",
            "negative_z": negative_z,
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": source_locations,
            "frequency": source_frequency_hz,
            "delay": 1.0,
            "delay_type": "time",
            "receiver_locations": receiver_locations,
        },
        "time_axis": {
            "initial_time": config.time.initial_time,
            "final_time": config.time.final_time,
            "dt": config.time.dt,
            "amplitude": 1.0,
            "output_frequency": 1000,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "gradient_output": False,
            "adjoint_output": False,
        },
    }

    return model_dictionary, source_locations, receiver_locations


def _normalize_trace_matrix(trace: Any) -> np.ndarray:
    trace_matrix = np.asarray(trace, dtype=float)
    if trace_matrix.ndim == 1:
        trace_matrix = trace_matrix[:, np.newaxis]
    if trace_matrix.ndim != 2:
        raise ConfigValidationError(
            "Forward traces must have shape [time, receiver]."
        )
    return trace_matrix


def _normalize_trace_cube(trace: Any) -> np.ndarray:
    trace_cube = np.asarray(trace, dtype=float)
    if trace_cube.ndim == 2:
        trace_cube = trace_cube[np.newaxis, :, :]
    if trace_cube.ndim != 3:
        raise ConfigValidationError(
            "Observed traces must have shape [shot, time, receiver] "
            "or [time, receiver] for a single shot."
        )
    return trace_cube


def _validate_trace_axis_size(
    axis_name: str, expected_size: int, actual_size: int
) -> None:
    if actual_size != expected_size:
        raise ConfigValidationError(
            f"Observed data {axis_name} dimension mismatch: "
            f"expected {expected_size}, actual {actual_size}."
        )


def load_observed_data_into_internal_trace_structure(
    config: InversionConfig,
    config_path: Union[str, Path],
    *,
    expected_shape: Tuple[int, int, int],
) -> np.ndarray:
    observed_data_path = _resolve_config_relative_path(
        config.observed_data.path, Path(config_path)
    )
    try:
        observed_data = np.load(observed_data_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ConfigValidationError(
            f"Unable to read observed data file: {observed_data_path}"
        ) from exc

    if len(expected_shape) != 3:
        raise ConfigValidationError(
            "Expected internal trace shape must define [shot, time, receiver], "
            f"got {expected_shape}."
        )

    observed_traces = _normalize_trace_cube(observed_data)
    _validate_trace_axis_size("shot", int(expected_shape[0]), observed_traces.shape[0])
    _validate_trace_axis_size("time", int(expected_shape[1]), observed_traces.shape[1])
    _validate_trace_axis_size(
        "receiver", int(expected_shape[2]), observed_traces.shape[2]
    )
    return observed_traces


def compute_residual_traces(
    synthetic_traces: Any, observed_traces: Any
) -> np.ndarray:
    synthetic_trace_cube = _normalize_trace_cube(synthetic_traces)
    observed_trace_cube = _normalize_trace_cube(observed_traces)

    for axis_name, synthetic_size, observed_size in (
        ("shot", synthetic_trace_cube.shape[0], observed_trace_cube.shape[0]),
        ("time", synthetic_trace_cube.shape[1], observed_trace_cube.shape[1]),
        ("receiver", synthetic_trace_cube.shape[2], observed_trace_cube.shape[2]),
    ):
        if synthetic_size != observed_size:
            raise ConfigValidationError(
                f"Residual trace {axis_name} dimension mismatch: "
                f"synthetic {synthetic_size}, observed {observed_size}."
            )

    return observed_trace_cube - synthetic_trace_cube


def compute_scalar_objective_from_residual_traces(
    residual_traces: Any, *, dt: Any = 1.0
) -> float:
    residual_trace_cube = _normalize_trace_cube(residual_traces)
    try:
        time_step = float(dt)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(
            f"Residual objective time step must be numeric, got {dt!r}."
        ) from exc

    if not np.isfinite(time_step) or time_step <= 0.0:
        raise ConfigValidationError(
            f"Residual objective time step must be finite and positive, got {dt!r}."
        )
    if not np.all(np.isfinite(residual_trace_cube)):
        raise ConfigValidationError(
            "Residual traces must contain only finite values to compute objective J."
        )

    objective_value = 0.5 * float(
        np.trapezoid(residual_trace_cube ** 2, dx=time_step, axis=1).sum()
    )
    if not np.isfinite(objective_value):
        raise ConfigValidationError("Computed objective J is not finite.")
    return objective_value


def run_acoustic_forward_modeling_from_config(
    config: InversionConfig,
    config_path: Union[str, Path],
    *,
    wave_factory: Any = None,
    switch_shot: Any = None,
    cleanup_tmp_files: Any = None,
    source_frequency_hz: float = _DEFAULT_FORWARD_SOURCE_FREQUENCY_HZ,
    velocity_model_km_s: Any = None,
) -> np.ndarray:
    """Run acoustic forward modeling from validated config and return synthetic traces."""
    if velocity_model_km_s is None:
        velocity_model_km_s = config.initial_model.velocity_km_s

    model_dictionary, source_locations, receiver_locations = _build_forward_model_dictionary(
        config, config_path, source_frequency_hz
    )
    if wave_factory is None:
        try:
            import spyro
        except ModuleNotFoundError:
            return _run_surrogate_forward_modeling_from_config(
                config,
                source_locations,
                receiver_locations,
                source_frequency_hz=source_frequency_hz,
                velocity_model_km_s=velocity_model_km_s,
            )
        wave_factory = spyro.AcousticWave

    wave = wave_factory(dictionary=model_dictionary)
    if hasattr(wave, "set_initial_velocity_model"):
        wave.set_initial_velocity_model(constant=velocity_model_km_s)
    wave.forward_solve()

    shot_traces = []
    number_of_sources = len(source_locations)
    if number_of_sources > 1:
        if switch_shot is None or cleanup_tmp_files is None:
            import spyro

            switch_shot = switch_shot or spyro.io.switch_serial_shot
            cleanup_tmp_files = cleanup_tmp_files or spyro.io.delete_tmp_files

        for source_index in range(number_of_sources):
            switch_shot(wave, source_index)
            shot_traces.append(_normalize_trace_matrix(wave.forward_solution_receivers).copy())
        cleanup_tmp_files(wave)
    else:
        shot_traces.append(_normalize_trace_matrix(wave.forward_solution_receivers).copy())

    try:
        synthetic_traces = np.stack(shot_traces, axis=0)
    except ValueError as exc:
        raise ConfigValidationError(
            "Forward traces are inconsistent across shots."
        ) from exc

    if synthetic_traces.shape[0] != number_of_sources:
        raise ConfigValidationError(
            "Forward modeling did not produce traces for every configured shot."
        )
    if synthetic_traces.shape[2] != len(receiver_locations):
        raise ConfigValidationError(
            "Forward modeling did not produce traces for every configured receiver."
        )

    return synthetic_traces


def _run_surrogate_forward_modeling_from_config(
    config: InversionConfig,
    source_locations: List[Tuple[float, ...]],
    receiver_locations: List[Tuple[float, ...]],
    *,
    source_frequency_hz: float,
    velocity_model_km_s: float,
) -> np.ndarray:
    try:
        initial_time = float(config.time.initial_time)
        final_time = float(config.time.final_time)
        dt = float(config.time.dt)
        source_frequency_hz = float(source_frequency_hz)
        velocity_model_km_s = float(velocity_model_km_s)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(
            "Unable to evaluate fallback forward model due to non-numeric "
            "time or wave parameters."
        ) from exc

    if final_time <= initial_time or dt <= 0.0:
        raise ConfigValidationError(
            "Unable to evaluate fallback forward model due to invalid time bounds."
        )
    if velocity_model_km_s <= 0.0:
        raise ConfigValidationError(
            "Unable to evaluate fallback forward model due to non-positive velocity."
        )

    sample_count = int(np.floor((final_time - initial_time) / dt)) + 1
    time_axis = initial_time + np.arange(sample_count, dtype=float) * dt
    synthetic_traces = np.zeros(
        (len(source_locations), sample_count, len(receiver_locations)),
        dtype=float,
    )
    pi_value = float(np.pi)

    for shot_index, source_point in enumerate(source_locations):
        source_coordinates = np.asarray(source_point, dtype=float)
        for receiver_index, receiver_point in enumerate(receiver_locations):
            receiver_coordinates = np.asarray(receiver_point, dtype=float)
            travel_time = (
                np.linalg.norm(receiver_coordinates - source_coordinates)
                / velocity_model_km_s
            )
            shifted_time = source_frequency_hz * (time_axis - travel_time)
            ricker_argument = pi_value * shifted_time
            synthetic_traces[shot_index, :, receiver_index] = (
                1.0 - 2.0 * ricker_argument ** 2
            ) * np.exp(-(ricker_argument ** 2))

    return synthetic_traces


def run_initial_forward_objective_evaluation(
    config: InversionConfig,
    config_path: Union[str, Path],
    *,
    forward_modeling_runner: Any = None,
    output_filename: str = _INITIAL_OBJECTIVE_FILENAME,
) -> Dict[str, Any]:
    if forward_modeling_runner is None:
        forward_modeling_runner = run_acoustic_forward_modeling_from_config

    synthetic_traces = _normalize_trace_cube(
        forward_modeling_runner(config, config_path)
    )
    expected_shape = tuple(int(size) for size in synthetic_traces.shape)
    observed_traces = load_observed_data_into_internal_trace_structure(
        config,
        config_path,
        expected_shape=expected_shape,
    )
    residual_traces = compute_residual_traces(synthetic_traces, observed_traces)
    objective_value = compute_scalar_objective_from_residual_traces(
        residual_traces, dt=config.time.dt
    )

    config_path_obj = Path(config_path).expanduser().resolve()
    output_directory = _resolve_config_relative_path(
        config.output.directory, config_path_obj
    )
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigValidationError(
            f"Unable to create output directory for initial objective: {output_directory}"
        ) from exc

    objective_path = output_directory / output_filename
    objective_record = {"iteration": 0, "objective": objective_value}
    try:
        objective_path.write_text(
            json.dumps(objective_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ConfigValidationError(
            f"Unable to persist initial objective J: {objective_path}"
        ) from exc

    return {
        "iteration": 0,
        "objective": objective_value,
        "objective_path": str(objective_path),
    }


def _extract_gradient_values(gradient_field: Any) -> np.ndarray:
    if gradient_field is None:
        raise ConfigValidationError("Gradient field is required to compute a norm.")

    if hasattr(gradient_field, "dat"):
        gradient_dat = getattr(gradient_field, "dat")
        if hasattr(gradient_dat, "data_ro"):
            gradient_values = np.asarray(gradient_dat.data_ro, dtype=float)
        elif hasattr(gradient_dat, "data"):
            gradient_values = np.asarray(gradient_dat.data, dtype=float)
        else:
            gradient_values = None

        if gradient_values is not None:
            if gradient_values.size == 0:
                raise ConfigValidationError(
                    "Gradient field is empty; unable to compute a norm."
                )
            return gradient_values.reshape(-1)

    try:
        gradient_values = np.asarray(gradient_field, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(
            "Gradient field values must be numeric to compute a norm."
        ) from exc

    if gradient_values.size == 0:
        raise ConfigValidationError("Gradient field is empty; unable to compute a norm.")
    return gradient_values.reshape(-1)


def compute_gradient_norm(gradient_field: Any) -> float:
    gradient_values = _extract_gradient_values(gradient_field)
    if not np.all(np.isfinite(gradient_values)):
        raise ConfigValidationError(
            "Gradient field contains non-finite values; unable to compute a norm."
        )

    gradient_norm_value = float(np.linalg.norm(gradient_values))
    if not np.isfinite(gradient_norm_value):
        raise ConfigValidationError(
            "Computed gradient norm is non-finite; check gradient values."
        )
    return gradient_norm_value


def compute_and_expose_gradient_norm_each_iteration(
    gradient_field: Any,
    *,
    gradient_norm_history: List[Dict[str, float]],
    iteration: Any = None,
) -> Dict[str, float]:
    if not isinstance(gradient_norm_history, list):
        raise ConfigValidationError(
            "Gradient norm history must be a list to record per-iteration values."
        )

    if iteration is None:
        iteration = len(gradient_norm_history)
    try:
        iteration_index = int(iteration)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(
            f"Gradient norm iteration index must be an integer, got {iteration!r}."
        ) from exc

    if iteration_index < 0:
        raise ConfigValidationError(
            f"Gradient norm iteration index must be non-negative, got {iteration_index}."
        )

    gradient_norm_value = compute_gradient_norm(gradient_field)
    gradient_norm_record = {
        "iteration": iteration_index,
        "gradient_norm": gradient_norm_value,
    }
    gradient_norm_history.append(gradient_norm_record)
    return gradient_norm_record


def build_optimizer_factory(
    config: Union[InversionConfig, Mapping[str, Any], None],
    *,
    minimize_callable: Any = None,
) -> Any:
    if minimize_callable is None:
        try:
            from scipy.optimize import minimize as scipy_minimize
        except ImportError as exc:
            raise ConfigValidationError(
                "SciPy is required to build the configured optimizer factory."
            ) from exc
        minimize_callable = scipy_minimize

    if not callable(minimize_callable):
        raise ConfigValidationError("Optimizer backend must be callable.")

    if isinstance(config, InversionConfig):
        optimizer_name = _normalize_optimizer_name(config.optimizer.name)
    elif config is None:
        optimizer_name = _DEFAULT_OPTIMIZER_NAME
    elif isinstance(config, Mapping):
        optimizer_section = config.get("optimizer")
        if optimizer_section is None:
            optimizer_name = _DEFAULT_OPTIMIZER_NAME
        elif isinstance(optimizer_section, Mapping):
            optimizer_name = _normalize_optimizer_name(
                optimizer_section.get("name", _DEFAULT_OPTIMIZER_NAME)
            )
        elif isinstance(optimizer_section, str):
            optimizer_name = _normalize_optimizer_name(optimizer_section)
        else:
            raise ConfigValidationError("Config field 'optimizer' must be a mapping.")
    else:
        raise ConfigValidationError("Optimizer factory input must be a config mapping.")

    def run_optimizer(objective_function: Any, initial_guess: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("method", optimizer_name)
        return minimize_callable(objective_function, initial_guess, **kwargs)

    run_optimizer.optimizer_name = optimizer_name
    return run_optimizer


def _load_firedrake_adjoint_symbol(symbol_name: str) -> Any:
    try:
        from firedrake.adjoint import Control, ReducedFunctional

        return {"Control": Control, "ReducedFunctional": ReducedFunctional}[symbol_name]
    except (ImportError, KeyError):
        pass

    try:
        import firedrake_adjoint as fire_adjoint
    except ImportError as exc:
        raise ConfigValidationError(
            "firedrake-adjoint is required to create inversion control variables."
        ) from exc

    if not hasattr(fire_adjoint, symbol_name):
        raise ConfigValidationError(
            f"firedrake-adjoint does not expose required symbol: {symbol_name}."
        )
    return getattr(fire_adjoint, symbol_name)


def define_velocity_model_as_primary_control_variable(
    config: InversionConfig,
    config_path: Union[str, Path],
    *,
    wave_factory: Any = None,
    control_factory: Any = None,
    taped_functional: Any = None,
    functional_taping_callback: Any = None,
    reduced_functional_factory: Any = None,
    source_frequency_hz: float = _DEFAULT_FORWARD_SOURCE_FREQUENCY_HZ,
) -> Dict[str, Any]:
    """Create a firedrake-adjoint control for the configured initial velocity model."""
    model_dictionary, _source_locations, _receiver_locations = _build_forward_model_dictionary(
        config, config_path, source_frequency_hz
    )

    if wave_factory is None:
        import spyro

        wave_factory = spyro.AcousticWave

    wave = wave_factory(dictionary=model_dictionary)
    if not hasattr(wave, "set_initial_velocity_model"):
        raise ConfigValidationError(
            "Wave object must define 'set_initial_velocity_model' to create controls."
        )
    wave.set_initial_velocity_model(constant=config.initial_model.velocity_km_s)

    velocity_model = getattr(wave, "initial_velocity_model", None)
    if velocity_model is None:
        raise ConfigValidationError(
            "Configured initial velocity model is unavailable for control creation."
        )

    if control_factory is None:
        control_factory = _load_firedrake_adjoint_symbol("Control")
    control = control_factory(velocity_model)

    if taped_functional is None and functional_taping_callback is not None:
        taped_functional = functional_taping_callback(wave)

    reduced_functional = None
    if taped_functional is not None:
        if reduced_functional_factory is None:
            reduced_functional_factory = _load_firedrake_adjoint_symbol(
                "ReducedFunctional"
            )
        reduced_functional = reduced_functional_factory(taped_functional, control)

    return {
        "wave": wave,
        "velocity_model": velocity_model,
        "control": control,
        "taped_functional": taped_functional,
        "reduced_functional": reduced_functional,
    }


def tape_misfit_functional_for_automatic_differentiation(
    config: InversionConfig,
    config_path: Union[str, Path],
    *,
    wave_factory: Any = None,
    control_factory: Any = None,
    taped_functional: Any = None,
    functional_taping_callback: Any = None,
    reduced_functional_factory: Any = None,
    source_frequency_hz: float = _DEFAULT_FORWARD_SOURCE_FREQUENCY_HZ,
) -> Dict[str, Any]:
    """Tape misfit functional and return firedrake-adjoint gradient."""
    if taped_functional is None and functional_taping_callback is None:
        raise ConfigValidationError(
            "Provide either 'taped_functional' or 'functional_taping_callback' "
            "to tape the misfit functional for automatic differentiation."
        )

    control_bundle = define_velocity_model_as_primary_control_variable(
        config,
        config_path,
        wave_factory=wave_factory,
        control_factory=control_factory,
        taped_functional=taped_functional,
        functional_taping_callback=functional_taping_callback,
        reduced_functional_factory=reduced_functional_factory,
        source_frequency_hz=source_frequency_hz,
    )

    reduced_functional = control_bundle["reduced_functional"]
    if reduced_functional is None:
        raise ConfigValidationError(
            "Misfit functional taping did not produce a ReducedFunctional."
        )

    derivative_function = getattr(reduced_functional, "derivative", None)
    if derivative_function is None or not callable(derivative_function):
        raise ConfigValidationError(
            "ReducedFunctional must provide a callable 'derivative' method."
        )

    gradient_field = derivative_function()
    if gradient_field is None:
        raise ConfigValidationError(
            "firedrake-adjoint returned an empty gradient field from the taped "
            "misfit functional."
        )

    gradient_norm_history: List[Dict[str, float]] = []
    gradient_norm_record = compute_and_expose_gradient_norm_each_iteration(
        gradient_field,
        gradient_norm_history=gradient_norm_history,
        iteration=0,
    )

    control_bundle["gradient"] = gradient_field
    control_bundle["gradient_norm"] = gradient_norm_record["gradient_norm"]
    control_bundle["gradient_norm_history"] = gradient_norm_history
    return control_bundle


def define_velocity_model_as_primary_firedrake_adjoint_control_variable(
    *args, **kwargs
) -> Dict[str, Any]:
    return define_velocity_model_as_primary_control_variable(*args, **kwargs)


def load_inversion_config_file(config_path: Union[str, Path]) -> InversionConfig:
    """Load, validate, and normalize an inversion configuration file."""
    return load_inversion_config(_load_config_mapping(Path(config_path)))
