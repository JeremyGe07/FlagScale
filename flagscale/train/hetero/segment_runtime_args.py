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
    flat_present = [key for key in _RUNTIME_KEYS if _resolve_path(source, (key,)) is not _MISSING]
    if not flat_present:
        return None
    if len(flat_present) != len(_RUNTIME_KEYS):
        raise ValueError("segment_runtime fields must be provided together")
    return {key: _resolve_path(source, (key,)) for key in _RUNTIME_KEYS}


def _parse_segment_runtime(runtime: Mapping[str, Any]) -> SegmentRuntimeSpec:
    _require_exact_keys(runtime, _RUNTIME_KEYS, "segment_runtime")
    stage_splits = _parse_stage_splits(runtime["hetero_stage_segment_splits"])
    stage_meshes = _parse_stage_meshes(runtime["hetero_stage_segment_meshes"])
    transitions = _parse_transitions(runtime["hetero_stage_segment_transitions"])
    return SegmentRuntimeSpec(_build_stages(stage_splits, stage_meshes, transitions))


def _build_stages(stage_splits, stage_meshes, transitions) -> tuple[SegmentRuntimeStageSpec, ...]:
    if len(stage_splits) != len(stage_meshes):
        raise ValueError("segment counts must match between splits and meshes")
    stages = []
    offset = 0
    for stage_id, (splits, meshes) in enumerate(zip(stage_splits, stage_meshes, strict=True)):
        expected = max(len(splits) - 1, 0)
        stage_transitions = transitions[offset : offset + expected]
        _validate_stage(stage_id, splits, meshes, stage_transitions)
        stages.append(
            SegmentRuntimeStageSpec(
                segment_splits=splits,
                segment_meshes=meshes,
                transitions=tuple(stage_transitions),
            )
        )
        offset += expected
    if offset != len(transitions):
        raise ValueError("transition indexes must match per-stage segment counts")
    return tuple(stages)


def _validate_stage(stage_id, splits, meshes, transitions) -> None:
    if len(splits) != len(meshes):
        raise ValueError("segment counts must match between splits and meshes")
    expected_transition_count = max(len(splits) - 1, 0)
    if len(transitions) != expected_transition_count:
        raise ValueError("transition indexes must match per-stage segment counts")
    for segment_id, (split, mesh) in enumerate(zip(splits, meshes, strict=True)):
        if split <= 0:
            raise ValueError("segment splits must be positive integers")
        if mesh.pp_local != 1:
            raise ValueError("segment-local pipeline size must be pp_local=1")
    for segment_index, transition in enumerate(transitions):
        if transition.source_stage_id != stage_id or transition.target_stage_id != stage_id:
            raise ValueError("transition indexes must stay within each stage")
        if transition.source_segment_index != segment_index:
            raise ValueError("transition indexes must match per-stage segment counts")
        if transition.target_segment_index != segment_index + 1:
            raise ValueError("transition indexes must match per-stage segment counts")
        if transition.kind != "segment-redistribution":
            raise ValueError("unsupported segment_runtime transition kind")


def _parse_stage_splits(value) -> tuple[tuple[int, ...], ...]:
    return tuple(_parse_int_tuple(stage, "hetero_stage_segment_splits") for stage in _require_sequence(value, "hetero_stage_segment_splits"))


def _parse_stage_meshes(value) -> tuple[tuple[SegmentRuntimeMeshSpec, ...], ...]:
    stages = []
    for stage_id, stage_meshes in enumerate(_require_sequence(value, "hetero_stage_segment_meshes")):
        meshes = []
        for segment_id, mesh in enumerate(_require_sequence(stage_meshes, "hetero_stage_segment_meshes")):
            meshes.append(_parse_mesh(mesh, stage_id, segment_id))
        stages.append(tuple(meshes))
    return tuple(stages)


def _parse_transitions(value) -> tuple[SegmentRuntimeTransitionSpec, ...]:
    transitions = []
    for item in _require_sequence(value, "hetero_stage_segment_transitions"):
        mapping = _as_mapping(item)
        _require_exact_keys(
            mapping,
            (
                "source_stage_id",
                "target_stage_id",
                "kind",
                "source_segment_index",
                "target_segment_index",
                "metadata",
            ),
            "segment transition",
        )
        transitions.append(
            SegmentRuntimeTransitionSpec(
                source_stage_id=_strict_int(mapping["source_stage_id"], "source_stage_id"),
                target_stage_id=_strict_int(mapping["target_stage_id"], "target_stage_id"),
                kind=_strict_str(mapping["kind"], "kind"),
                source_segment_index=_strict_int(
                    mapping["source_segment_index"], "source_segment_index"
                ),
                target_segment_index=_strict_int(
                    mapping["target_segment_index"], "target_segment_index"
                ),
                metadata=_as_mapping(mapping["metadata"]),
            )
        )
    return tuple(transitions)


def _parse_mesh(mesh, stage_id: int, segment_id: int) -> SegmentRuntimeMeshSpec:
    values = _require_sequence(mesh, f"stage {stage_id} segment {segment_id} mesh")
    if len(values) != 5:
        raise ValueError("segment mesh must contain exactly five dimensions")
    tensor = _strict_int(values[0], "tensor_model_parallel_size")
    context = _strict_int(values[1], "context_parallel_size")
    expert = _strict_int(values[2], "expert_model_parallel_size")
    data = _strict_int(values[3], "data_parallel_size")
    pp_local = _strict_unit_int(values[4], "pp_local")
    return SegmentRuntimeMeshSpec(
        tensor_model_parallel_size=tensor,
        context_parallel_size=context,
        expert_model_parallel_size=expert,
        data_parallel_size=data,
        pp_local=pp_local,
    )


def _parse_int_tuple(value, label: str) -> tuple[int, ...]:
    items = _require_sequence(value, label)
    return tuple(_strict_int(item, label) for item in items)


def _require_sequence(value, label: str):
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    return tuple(value)


def _strict_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be a non-bool int")
    return value


def _strict_unit_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise ValueError(f"{label} must be a non-bool int equal to 1")
    return value


def _strict_str(value, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _require_exact_keys(mapping: Mapping[str, Any], expected: tuple[str, ...], label: str) -> None:
    keys = set(mapping.keys())
    expected_keys = set(expected)
    if keys != expected_keys:
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
