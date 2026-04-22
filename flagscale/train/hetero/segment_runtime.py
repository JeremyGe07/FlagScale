from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

TRANSITION_MESH_KEYS = ("tp", "cp", "ep", "dp", "pp")
REDISTRIBUTION_KINDS = frozenset({"tp-only", "dp-only", "tp-dp"})


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


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive non-bool int")
    return value


def _freeze_transition_mesh(mesh: Mapping[str, Any], label: str) -> Mapping[str, int]:
    frozen = _freeze_mapping(mesh)
    if set(frozen.keys()) != set(TRANSITION_MESH_KEYS):
        raise ValueError(f"{label} has unexpected or missing keys")
    values = {key: _positive_int(frozen[key], f"{label}.{key}") for key in TRANSITION_MESH_KEYS}
    if values["pp"] != 1:
        raise ValueError(f"{label}.pp must be 1")
    return MappingProxyType(values)


@dataclass(frozen=True)
class SegmentRuntimeMeshSpec:
    tensor_model_parallel_size: int
    context_parallel_size: int
    expert_model_parallel_size: int
    data_parallel_size: int
    pp_local: int

    def __post_init__(self) -> None:
        _positive_int(self.tensor_model_parallel_size, "tensor_model_parallel_size")
        _positive_int(self.context_parallel_size, "context_parallel_size")
        _positive_int(self.expert_model_parallel_size, "expert_model_parallel_size")
        _positive_int(self.data_parallel_size, "data_parallel_size")
        if isinstance(self.pp_local, bool) or not isinstance(self.pp_local, int) or self.pp_local != 1:
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
    source_mesh: Mapping[str, int]
    target_mesh: Mapping[str, int]
    batch_unit: int
    redistribution_kind: str
    requires_sequence_parallel: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_mesh", _freeze_transition_mesh(self.source_mesh, "source_mesh"))
        object.__setattr__(self, "target_mesh", _freeze_transition_mesh(self.target_mesh, "target_mesh"))
        _positive_int(self.batch_unit, "batch_unit")
        _validate_batch_unit(self.batch_unit, self.source_mesh, self.target_mesh)
        _validate_redistribution_kind(
            self.redistribution_kind,
            self.source_mesh,
            self.target_mesh,
        )
        if not isinstance(self.requires_sequence_parallel, bool):
            raise ValueError("requires_sequence_parallel must be bool")
        if self.requires_sequence_parallel != _tp_changed(self.source_mesh, self.target_mesh):
            raise ValueError("requires_sequence_parallel must match tensor parallel changes")

    def to_runtime_dict(self) -> dict[str, Any]:
        return {
            "source_stage_id": self.source_stage_id,
            "target_stage_id": self.target_stage_id,
            "kind": self.kind,
            "source_segment_index": self.source_segment_index,
            "target_segment_index": self.target_segment_index,
            "source_mesh": _freeze_to_plain_dict(self.source_mesh),
            "target_mesh": _freeze_to_plain_dict(self.target_mesh),
            "batch_unit": self.batch_unit,
            "redistribution_kind": self.redistribution_kind,
            "requires_sequence_parallel": self.requires_sequence_parallel,
        }


@dataclass(frozen=True)
class SegmentRuntimeStageSpec:
    segment_splits: tuple[int, ...]
    segment_meshes: tuple[SegmentRuntimeMeshSpec, ...]
    transitions: tuple[SegmentRuntimeTransitionSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "segment_splits", tuple(self.segment_splits))
        object.__setattr__(self, "segment_meshes", tuple(self.segment_meshes))
        object.__setattr__(self, "transitions", tuple(self.transitions))


@dataclass(frozen=True)
class SegmentRuntimeSpec:
    stages: tuple[SegmentRuntimeStageSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", tuple(self.stages))

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


def _validate_batch_unit(
    batch_unit: int,
    source_mesh: Mapping[str, int],
    target_mesh: Mapping[str, int],
) -> None:
    if batch_unit % source_mesh["dp"] != 0 or batch_unit % target_mesh["dp"] != 0:
        raise ValueError("batch_unit must be divisible by source and target dp")


def _validate_redistribution_kind(
    redistribution_kind: str,
    source_mesh: Mapping[str, int],
    target_mesh: Mapping[str, int],
) -> None:
    if redistribution_kind not in REDISTRIBUTION_KINDS:
        raise ValueError("redistribution_kind must be one of tp-only, dp-only, tp-dp")
    if redistribution_kind != _expected_redistribution_kind(source_mesh, target_mesh):
        raise ValueError("redistribution_kind must match source and target mesh changes")


def _expected_redistribution_kind(
    source_mesh: Mapping[str, int],
    target_mesh: Mapping[str, int],
) -> str:
    tp_changed = _tp_changed(source_mesh, target_mesh)
    dp_changed = source_mesh["dp"] != target_mesh["dp"]
    if tp_changed and dp_changed:
        return "tp-dp"
    if tp_changed:
        return "tp-only"
    if dp_changed:
        return "dp-only"
    raise ValueError("redistribution_kind requires tp or dp changes across the boundary")


def _tp_changed(source_mesh: Mapping[str, int], target_mesh: Mapping[str, int]) -> bool:
    return source_mesh["tp"] != target_mesh["tp"]


__all__ = [
    "SegmentRuntimeMeshSpec",
    "SegmentRuntimeSpec",
    "SegmentRuntimeStageSpec",
    "SegmentRuntimeTransitionSpec",
]
