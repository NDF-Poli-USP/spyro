from dataclasses import dataclass
from typing import Any, Mapping


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
class InversionConfig:
    mesh: MeshConfig
    time: TimeConfig
    geometry: GeometryConfig
    observed_data: ObservedDataConfig
    optimizer: OptimizerConfig
    checkpoint: CheckpointConfig
    output: OutputConfig


_REQUIRED_FIELDS = {
    "mesh": ("mesh_file",),
    "time": ("initial_time", "final_time", "dt"),
    "geometry": ("sources", "receivers"),
    "observed_data": ("path",),
    "optimizer": ("name", "max_iterations", "tolerance"),
    "checkpoint": ("directory", "every"),
    "output": ("directory",),
}


def _require_fields(section: Mapping[str, Any], fields, section_name: str) -> None:
    for field in fields:
        field_name = f"{section_name}.{field}" if section_name else field
        if field not in section:
            raise ConfigValidationError(f"Missing required config field: {field_name}")


def load_inversion_config(config: Mapping[str, Any]) -> InversionConfig:
    """Validate and normalize an inversion configuration mapping."""
    if not isinstance(config, Mapping):
        raise ConfigValidationError("Inversion config must be a mapping.")

    _require_fields(config, _REQUIRED_FIELDS.keys(), "")

    for section_name, required_section_fields in _REQUIRED_FIELDS.items():
        section = config[section_name]
        if not isinstance(section, Mapping):
            raise ConfigValidationError(f"Config field '{section_name}' must be a mapping.")
        _require_fields(section, required_section_fields, section_name)

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
        optimizer=OptimizerConfig(
            name=config["optimizer"]["name"],
            max_iterations=config["optimizer"]["max_iterations"],
            tolerance=config["optimizer"]["tolerance"],
        ),
        checkpoint=CheckpointConfig(
            directory=config["checkpoint"]["directory"],
            every=config["checkpoint"]["every"],
        ),
        output=OutputConfig(directory=config["output"]["directory"]),
    )
