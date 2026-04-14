from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    frozen_values = {key: _freeze_value(value) for key, value in dict(values or {}).items()}
    return MappingProxyType(frozen_values)


def _freeze_tuple(values: tuple[Any, ...] | list[Any]) -> tuple[Any, ...]:
    return tuple(_freeze_value(value) for value in values)


@dataclass(frozen=True)
class SegmentPlan:
    """A stage-local layer span using inclusive bounds [start, end]."""

    start: int
    end: int
    strategy: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy", _freeze_mapping(self.strategy))


@dataclass(frozen=True)
class StagePlan:
    stage_id: int
    segments: tuple[SegmentPlan, ...]
    device_group: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", _freeze_tuple(self.segments))
        object.__setattr__(self, "device_group", _freeze_tuple(self.device_group))


@dataclass(frozen=True)
class TransitionPlan:
    source_stage_id: int
    target_stage_id: int
    kind: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class ExecutionContract:
    world_size: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    global_batch_size: int | None = None


@dataclass(frozen=True)
class ModelPlan:
    stages: tuple[StagePlan, ...]
    transitions: tuple[TransitionPlan, ...] = ()
    contract: ExecutionContract | None = None
    total_layers: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", _freeze_tuple(self.stages))
        object.__setattr__(self, "transitions", _freeze_tuple(self.transitions))
