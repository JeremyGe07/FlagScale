from collections.abc import Mapping, Sequence, Set

from flagscale.runner.auto_tuner.plan.schema import ModelPlan

HOMOGENEOUS_PLAN = "homogeneous"
HOMOGENEOUS_VPP_PLAN = "homogeneous-vpp"
STAGE_HETEROGENEOUS_PLAN = "stage-heterogeneous"
SEGMENT_HETEROGENEOUS_PLAN = "segment-heterogeneous"


def to_json_safe(value):
    if isinstance(value, Mapping):
        return {key: to_json_safe(item) for key, item in value.items()}
    if isinstance(value, Set):
        return [to_json_safe(item) for item in sorted(value, key=repr)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_json_safe(item) for item in value]
    return value


def summarize_plan(plan: ModelPlan) -> dict[str, object]:
    return {
        "total_layers": plan.total_layers,
        "stage_count": len(plan.stages),
        "vpp_stage_segment_counts": [len(stage.segments) for stage in plan.stages],
        "contract": {
            "world_size": None if plan.contract is None else plan.contract.world_size,
            "micro_batch_size": None if plan.contract is None else plan.contract.micro_batch_size,
            "gradient_accumulation_steps": (
                None if plan.contract is None else plan.contract.gradient_accumulation_steps
            ),
            "global_batch_size": None if plan.contract is None else plan.contract.global_batch_size,
        },
        "stages": [
            {
                "stage_id": stage.stage_id,
                "device_group": list(stage.device_group),
                "segments": [
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "strategy": to_json_safe(dict(segment.strategy)),
                    }
                    for segment in stage.segments
                ],
            }
            for stage in plan.stages
        ],
        "transitions": [
            {
                "source_stage_id": transition.source_stage_id,
                "target_stage_id": transition.target_stage_id,
                "kind": transition.kind,
                "source_segment_index": transition.source_segment_index,
                "target_segment_index": transition.target_segment_index,
                "metadata": to_json_safe(dict(transition.metadata)),
            }
            for transition in plan.transitions
        ],
    }


def summarize_execution_contract(plan: ModelPlan) -> dict[str, object]:
    return {
        "world_size": None if plan.contract is None else plan.contract.world_size,
        "micro_batch_size": None if plan.contract is None else plan.contract.micro_batch_size,
        "gradient_accumulation_steps": (
            None if plan.contract is None else plan.contract.gradient_accumulation_steps
        ),
        "global_batch_size": None if plan.contract is None else plan.contract.global_batch_size,
    }


def summarize_segment_runtime_contract(plan: ModelPlan) -> dict[str, object]:
    segment_transitions = [
        transition
        for transition in plan.transitions
        if transition.kind == "segment-redistribution"
    ]
    batch_unit = _segment_batch_unit(plan)
    stage_map = {stage.stage_id: stage for stage in plan.stages}
    return {
        "hetero_stage_segment_splits": [
            [segment.end - segment.start + 1 for segment in stage.segments]
            for stage in plan.stages
        ],
        "hetero_stage_segment_meshes": [
            [_segment_mesh_record(segment.strategy) for segment in stage.segments]
            for stage in plan.stages
        ],
        "hetero_stage_segment_transitions": [
            _segment_transition_record(transition, batch_unit, stage_map)
            for transition in segment_transitions
        ],
    }


def segment_count(plan: ModelPlan) -> int:
    return sum(len(stage.segments) for stage in plan.stages)


def plan_kind(plan: ModelPlan) -> str:
    homogeneous_strategy = extract_homogeneous_strategy(plan)
    has_multi_segment_stage = any(len(stage.segments) > 1 for stage in plan.stages)
    if homogeneous_strategy is not None:
        return HOMOGENEOUS_VPP_PLAN if has_multi_segment_stage else HOMOGENEOUS_PLAN
    if all(len(stage.segments) == 1 for stage in plan.stages):
        return STAGE_HETEROGENEOUS_PLAN
    return SEGMENT_HETEROGENEOUS_PLAN


def is_runtime_executable_plan(plan: ModelPlan) -> bool:
    return plan_kind(plan) in {
        HOMOGENEOUS_PLAN,
        HOMOGENEOUS_VPP_PLAN,
        STAGE_HETEROGENEOUS_PLAN,
        SEGMENT_HETEROGENEOUS_PLAN,
    }


def extract_homogeneous_strategy(plan: ModelPlan) -> dict[str, object] | None:
    if not plan.stages or plan.transitions:
        return None
    stage_segments = [segment for stage in plan.stages for segment in stage.segments]
    if not stage_segments:
        return None
    strategy = dict(stage_segments[0].strategy)
    for segment in stage_segments[1:]:
        if dict(segment.strategy) != strategy:
            return None
    return strategy


def _segment_mesh_record(strategy: Mapping[str, object]) -> list[int]:
    return [
        _required_strategy_int(strategy, ("tensor_model_parallel_size", "tp")),
        _required_strategy_int(strategy, ("context_parallel_size", "cp")),
        _required_strategy_int(strategy, ("expert_model_parallel_size", "ep")),
        _required_strategy_int(strategy, ("data_parallel_size", "dp")),
        _segment_local_pipeline_size(strategy),
    ]


def _segment_local_pipeline_size(strategy: Mapping[str, object]) -> int:
    value = strategy.get("pp_local")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    value = strategy.get("pipeline_model_parallel_size")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 1


def _segment_batch_unit(plan: ModelPlan) -> int:
    contract = plan.contract
    if contract is None or contract.global_batch_size is None:
        raise ValueError("segment runtime contract requires global_batch_size")
    if contract.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if contract.global_batch_size % contract.gradient_accumulation_steps != 0:
        raise ValueError("global_batch_size must be divisible by gradient_accumulation_steps")
    return contract.global_batch_size // contract.gradient_accumulation_steps


def _segment_transition_record(transition, batch_unit: int, stage_map) -> dict[str, object]:
    source_mesh = _stage_segment_transition_mesh(
        stage_map,
        transition.source_stage_id,
        transition.source_segment_index,
        "source",
    )
    target_mesh = _stage_segment_transition_mesh(
        stage_map,
        transition.target_stage_id,
        transition.target_segment_index,
        "target",
    )
    _validate_segment_runtime_mesh(source_mesh, "source_mesh")
    _validate_segment_runtime_mesh(target_mesh, "target_mesh")
    _validate_batch_unit(batch_unit, source_mesh, target_mesh)
    return {
        "source_stage_id": transition.source_stage_id,
        "target_stage_id": transition.target_stage_id,
        "kind": transition.kind,
        "source_segment_index": transition.source_segment_index,
        "target_segment_index": transition.target_segment_index,
        "source_mesh": source_mesh,
        "target_mesh": target_mesh,
        "batch_unit": batch_unit,
        "redistribution_kind": _redistribution_kind(source_mesh, target_mesh),
        "requires_sequence_parallel": source_mesh["tp"] != target_mesh["tp"],
    }


def _stage_segment_transition_mesh(
    stage_map,
    stage_id: int | None,
    segment_index: int | None,
    label: str,
) -> dict[str, int]:
    if stage_id is None or segment_index is None:
        raise ValueError(f"{label} segment transition must define stage and segment indexes")
    stage = stage_map.get(stage_id)
    if stage is None:
        raise ValueError(f"{label} stage {stage_id} is missing from segment runtime plan")
    if segment_index < 0 or segment_index >= len(stage.segments):
        raise ValueError(f"{label} segment index {segment_index} is out of range")
    strategy = stage.segments[segment_index].strategy
    return {
        "tp": _required_strategy_int(strategy, ("tensor_model_parallel_size", "tp")),
        "cp": _required_strategy_int(strategy, ("context_parallel_size", "cp")),
        "ep": _required_strategy_int(strategy, ("expert_model_parallel_size", "ep")),
        "dp": _required_strategy_int(strategy, ("data_parallel_size", "dp")),
        "pp": _segment_local_pipeline_size(strategy),
    }


def _redistribution_kind(source_mesh: Mapping[str, int], target_mesh: Mapping[str, int]) -> str:
    tp_changed = source_mesh["tp"] != target_mesh["tp"]
    dp_changed = source_mesh["dp"] != target_mesh["dp"]
    if tp_changed and dp_changed:
        return "tp-dp"
    if tp_changed:
        return "tp-only"
    if dp_changed:
        return "dp-only"
    raise ValueError("redistribution_kind requires tp or dp changes across the boundary")


def _validate_segment_runtime_mesh(mesh: Mapping[str, int], label: str) -> None:
    if mesh["pp"] != 1:
        raise ValueError(f"{label}.pp must be 1")


def _validate_batch_unit(
    batch_unit: int,
    source_mesh: Mapping[str, int],
    target_mesh: Mapping[str, int],
) -> None:
    if batch_unit % source_mesh["dp"] != 0 or batch_unit % target_mesh["dp"] != 0:
        raise ValueError("batch_unit must be divisible by source and target dp")


def _required_strategy_int(strategy: Mapping[str, object], keys: tuple[str, ...]) -> int:
    for key in keys:
        value = strategy.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    raise ValueError(f"missing required strategy field {keys[0]}")


__all__ = [
    "extract_homogeneous_strategy",
    "is_runtime_executable_plan",
    "plan_kind",
    "segment_count",
    "summarize_execution_contract",
    "summarize_segment_runtime_contract",
    "summarize_plan",
    "to_json_safe",
]
