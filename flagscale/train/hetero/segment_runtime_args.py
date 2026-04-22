from collections.abc import Mapping, Sequence
from typing import Any

from flagscale.train.hetero.segment_runtime import (
    SegmentRuntimeMeshSpec,
    SegmentRuntimeSpec,
    SegmentRuntimeStageSpec,
    SegmentRuntimeTransitionSpec,
)

_RUNTIME_KEYS = (
    "hetero_stage_segment_splits",
    "hetero_stage_segment_meshes",
    "hetero_stage_segment_transitions",
)
_TRANSITION_KEYS = (
    "source_stage_id",
    "target_stage_id",
    "kind",
    "source_segment_index",
    "target_segment_index",
    "source_mesh",
    "target_mesh",
    "batch_unit",
    "redistribution_kind",
    "requires_sequence_parallel",
)
_TRANSITION_MESH_KEYS = ("tp", "cp", "ep", "dp", "pp")
_MISSING = object()


def parse_segment_runtime_args(source) -> SegmentRuntimeSpec | None:
    runtime = _resolve_segment_runtime(source)
    if runtime is None:
        return None
    return _parse_segment_runtime(_as_mapping(runtime))


def _resolve_segment_runtime(source):
    nested = _resolve_path(source, ("train", "system", "hetero", "segment_runtime"))
    if nested is not _MISSING:
        return nested
    explicit = _resolve_path(source, ("segment_runtime",))
    if explicit is not _MISSING:
        return explicit
    present = [key for key in _RUNTIME_KEYS if _resolve_path(source, (key,)) is not _MISSING]
    if not present:
        return None
    if len(present) != len(_RUNTIME_KEYS):
        raise ValueError("segment_runtime fields must be provided together")
    return {key: _resolve_path(source, (key,)) for key in _RUNTIME_KEYS}


def _parse_segment_runtime(runtime: Mapping[str, Any]) -> SegmentRuntimeSpec:
    _require_exact_keys(runtime, _RUNTIME_KEYS, "segment_runtime")
    stage_splits = _parse_stage_splits(runtime["hetero_stage_segment_splits"])
    stage_meshes = _parse_stage_meshes(runtime["hetero_stage_segment_meshes"])
    transitions = _parse_transitions(runtime["hetero_stage_segment_transitions"])
    return SegmentRuntimeSpec(_build_stages(stage_splits, stage_meshes, transitions))


def _build_stages(
    stage_splits,
    stage_meshes,
    transitions,
) -> tuple[SegmentRuntimeStageSpec, ...]:
    if len(stage_splits) != len(stage_meshes):
        raise ValueError("segment counts must match between splits and meshes")
    grouped = _group_stage_transitions(transitions)
    stages = []
    for stage_id, (splits, meshes) in enumerate(zip(stage_splits, stage_meshes, strict=True)):
        stage_transitions = grouped.pop(stage_id, ())
        _validate_stage(stage_id, splits, meshes, stage_transitions)
        stages.append(
            SegmentRuntimeStageSpec(
                segment_splits=splits,
                segment_meshes=meshes,
                transitions=stage_transitions,
            )
        )
    if grouped:
        raise ValueError("transition stage ids must refer to existing stages")
    return tuple(stages)


def _validate_stage(stage_id, splits, meshes, transitions) -> None:
    if not splits:
        raise ValueError("segment_runtime stage must not be empty")
    if len(splits) != len(meshes):
        raise ValueError("segment counts must match between splits and meshes")
    _validate_stage_fixed_mesh_dimensions(meshes)
    expected_count = len(splits) - 1
    if len(transitions) != expected_count:
        raise ValueError("transition indexes must match per-stage segment counts")
    for segment_index, split in enumerate(splits):
        if split <= 0:
            raise ValueError(f"stage {stage_id} segment {segment_index} splits must be positive")
    for segment_index, transition in enumerate(transitions):
        _validate_transition_position(stage_id, segment_index, transition)
        _validate_transition_mesh_alignment(
            transition,
            meshes[segment_index],
            meshes[segment_index + 1],
        )


def _validate_transition_position(stage_id, segment_index, transition) -> None:
    if transition.source_stage_id != stage_id or transition.target_stage_id != stage_id:
        raise ValueError("transition indexes must stay within each stage")
    if transition.source_segment_index != segment_index:
        raise ValueError("transition indexes must match per-stage segment counts")
    if transition.target_segment_index != segment_index + 1:
        raise ValueError("transition indexes must match per-stage segment counts")
    if transition.kind != "segment-redistribution":
        raise ValueError("unsupported segment_runtime transition kind")


def _validate_transition_mesh_alignment(
    transition: SegmentRuntimeTransitionSpec,
    source_mesh: SegmentRuntimeMeshSpec,
    target_mesh: SegmentRuntimeMeshSpec,
) -> None:
    if dict(transition.source_mesh) != _stage_mesh_record(source_mesh):
        raise ValueError("source_mesh must match source segment mesh")
    if dict(transition.target_mesh) != _stage_mesh_record(target_mesh):
        raise ValueError("target_mesh must match target segment mesh")


def _validate_stage_fixed_mesh_dimensions(meshes) -> None:
    reference = meshes[0]
    for mesh in meshes[1:]:
        if mesh.context_parallel_size != reference.context_parallel_size:
            raise ValueError("context_parallel_size must stay consistent within each stage")
        if mesh.expert_model_parallel_size != reference.expert_model_parallel_size:
            raise ValueError("expert_model_parallel_size must stay consistent within each stage")


def _parse_stage_splits(value) -> tuple[tuple[int, ...], ...]:
    return tuple(
        _parse_positive_int_tuple(stage, "hetero_stage_segment_splits")
        for stage in _require_sequence(value, "hetero_stage_segment_splits")
    )


def _parse_stage_meshes(value) -> tuple[tuple[SegmentRuntimeMeshSpec, ...], ...]:
    stages = []
    for stage_id, stage_meshes in enumerate(_require_sequence(value, "hetero_stage_segment_meshes")):
        meshes = tuple(
            _parse_mesh(mesh, stage_id, segment_id)
            for segment_id, mesh in enumerate(
                _require_sequence(stage_meshes, "hetero_stage_segment_meshes")
            )
        )
        stages.append(meshes)
    return tuple(stages)


def _parse_transitions(value) -> tuple[SegmentRuntimeTransitionSpec, ...]:
    transitions = []
    for item in _require_sequence(value, "hetero_stage_segment_transitions"):
        mapping = _as_mapping(item)
        _require_exact_keys(mapping, _TRANSITION_KEYS, "segment transition")
        transitions.append(
            SegmentRuntimeTransitionSpec(
                source_stage_id=_strict_int(mapping["source_stage_id"], "source_stage_id"),
                target_stage_id=_strict_int(mapping["target_stage_id"], "target_stage_id"),
                kind=_strict_str(mapping["kind"], "kind"),
                source_segment_index=_strict_int(
                    mapping["source_segment_index"],
                    "source_segment_index",
                ),
                target_segment_index=_strict_int(
                    mapping["target_segment_index"],
                    "target_segment_index",
                ),
                source_mesh=_parse_transition_mesh(mapping["source_mesh"], "source_mesh"),
                target_mesh=_parse_transition_mesh(mapping["target_mesh"], "target_mesh"),
                batch_unit=_strict_positive_int(mapping["batch_unit"], "batch_unit"),
                redistribution_kind=_strict_str(
                    mapping["redistribution_kind"],
                    "redistribution_kind",
                ),
                requires_sequence_parallel=_strict_bool(
                    mapping["requires_sequence_parallel"],
                    "requires_sequence_parallel",
                ),
            )
        )
    return tuple(transitions)


def _parse_mesh(mesh, stage_id: int, segment_id: int) -> SegmentRuntimeMeshSpec:
    values = _require_sequence(mesh, f"stage {stage_id} segment {segment_id} mesh")
    if len(values) != 5:
        raise ValueError("segment mesh must contain exactly five dimensions")
    return SegmentRuntimeMeshSpec(
        tensor_model_parallel_size=_strict_positive_int(
            values[0],
            "tensor_model_parallel_size",
        ),
        context_parallel_size=_strict_positive_int(values[1], "context_parallel_size"),
        expert_model_parallel_size=_strict_positive_int(values[2], "expert_model_parallel_size"),
        data_parallel_size=_strict_positive_int(values[3], "data_parallel_size"),
        pp_local=_strict_unit_int(values[4], "pp_local"),
    )


def _stage_mesh_record(mesh: SegmentRuntimeMeshSpec) -> dict[str, int]:
    return {
        "tp": mesh.tensor_model_parallel_size,
        "cp": mesh.context_parallel_size,
        "ep": mesh.expert_model_parallel_size,
        "dp": mesh.data_parallel_size,
        "pp": mesh.pp_local,
    }


def _parse_transition_mesh(value, label: str) -> dict[str, int]:
    mapping = _as_mapping(value)
    _require_exact_keys(mapping, _TRANSITION_MESH_KEYS, label)
    mesh = {
        key: _strict_positive_int(mapping[key], f"{label}.{key}")
        for key in _TRANSITION_MESH_KEYS
    }
    if mesh["pp"] != 1:
        raise ValueError(f"{label}.pp must be a non-bool int equal to 1")
    return mesh


def _group_stage_transitions(
    transitions: tuple[SegmentRuntimeTransitionSpec, ...],
) -> dict[int, tuple[SegmentRuntimeTransitionSpec, ...]]:
    grouped: dict[int, dict[tuple[int, int], SegmentRuntimeTransitionSpec]] = {}
    for transition in transitions:
        stage = grouped.setdefault(transition.source_stage_id, {})
        key = (transition.source_segment_index, transition.target_segment_index)
        if key in stage:
            raise ValueError("segment transitions must be unique per stage boundary")
        stage[key] = transition
    return {stage_id: tuple(stage[key] for key in sorted(stage)) for stage_id, stage in grouped.items()}


def _parse_positive_int_tuple(value, label: str) -> tuple[int, ...]:
    items = _require_sequence(value, label)
    return tuple(_strict_positive_int(item, label) for item in items)


def _require_sequence(value, label: str):
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    return tuple(value)


def _strict_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be a non-bool int")
    return value


def _strict_positive_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive non-bool int")
    return value


def _strict_unit_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise ValueError(f"{label} must be a non-bool int equal to 1")
    return value


def _strict_str(value, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _strict_bool(value, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be bool")
    return value


def _require_exact_keys(mapping: Mapping[str, Any], expected: tuple[str, ...], label: str) -> None:
    if set(mapping.keys()) != set(expected):
        raise ValueError(f"{label} has unexpected or missing keys")


def _as_mapping(value) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError("segment_runtime values must be mappings")


def _resolve_path(source, keys):
    current = source
    for key in keys:
        current = _get_value(current, key)
        if current is _MISSING:
            return _MISSING
    return current


def _get_value(source, key):
    if source is None:
        return _MISSING
    if isinstance(source, Mapping) and key in source:
        return source[key]
    try:
        return source[key]
    except Exception:
        pass
    try:
        return getattr(source, key)
    except AttributeError:
        return _MISSING


__all__ = ["parse_segment_runtime_args"]
