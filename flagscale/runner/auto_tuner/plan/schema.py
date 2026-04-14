from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def _freeze_mapping(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(values or {}))


def _freeze_tuple(values: tuple[Any, ...] | list[Any]) -> tuple[Any, ...]:
    return tuple(values)


@dataclass(frozen=True)
class SegmentPlan:
    start: int
    end: int
    strategy: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy", _freeze_mapping(self.strategy))


@dataclass(frozen=True)
class StagePlan:
    stage_id: int
    segments: tuple[SegmentPlan, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", _freeze_tuple(self.segments))


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


@dataclass(frozen=True)
class ModelPlan:
    stages: tuple[StagePlan, ...]
    transitions: tuple[TransitionPlan, ...] = ()
    contract: ExecutionContract | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", _freeze_tuple(self.stages))
        object.__setattr__(self, "transitions", _freeze_tuple(self.transitions))
