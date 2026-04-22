from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_value(item) for key, item in dict(value).items()})
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze_value(item) for key, item in dict(values or {}).items()})


@dataclass(frozen=True)
class SegmentRuntimeMeshSpec:
    tensor_model_parallel_size: int
    context_parallel_size: int
    expert_model_parallel_size: int
    data_parallel_size: int
    pp_local: int

    def __post_init__(self) -> None:
        if self.pp_local != 1:
            raise ValueError("segment-local pipeline size must be pp_local=1")

    def to_runtime_mesh(self) -> list[int]:
        return [
            self.tensor_model_parallel_size,
            self.context_parallel_size,
            self.expert_model_parallel_size,
            self.data_parallel_size,
            self.pp_local,
        ]


@dataclass(frozen=True)
class SegmentRuntimeTransitionSpec:
    source_stage_id: int
    target_stage_id: int
    kind: str
    source_segment_index: int
    target_segment_index: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_runtime_dict(self) -> dict[str, Any]:
        return {
            "source_stage_id": self.source_stage_id,
            "target_stage_id": self.target_stage_id,
            "kind": self.kind,
            "source_segment_index": self.source_segment_index,
            "target_segment_index": self.target_segment_index,
            "metadata": _freeze_to_plain_dict(self.metadata),
        }


@dataclass(frozen=True)
class SegmentRuntimeStageSpec:
    segment_splits: tuple[int, ...]
    segment_meshes: tuple[SegmentRuntimeMeshSpec, ...]
    transitions: tuple[SegmentRuntimeTransitionSpec, ...]


@dataclass(frozen=True)
class SegmentRuntimeSpec:
    stages: tuple[SegmentRuntimeStageSpec, ...]

    def to_runtime_dict(self) -> dict[str, Any]:
        return {
            "hetero_stage_segment_splits": [list(stage.segment_splits) for stage in self.stages],
            "hetero_stage_segment_meshes": [
                [mesh.to_runtime_mesh() for mesh in stage.segment_meshes]
                for stage in self.stages
            ],
            "hetero_stage_segment_transitions": [
                transition.to_runtime_dict()
                for stage in self.stages
                for transition in stage.transitions
            ],
        }


def _freeze_to_plain_dict(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _freeze_to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_freeze_to_plain_dict(item) for item in value]
    return value


__all__ = [
    "SegmentRuntimeMeshSpec",
    "SegmentRuntimeSpec",
    "SegmentRuntimeStageSpec",
    "SegmentRuntimeTransitionSpec",
]
