from dataclasses import dataclass
from typing import Literal

from flagscale.runner.auto_tuner.plan.schema import ModelPlan, SegmentPlan, StagePlan

ANALYSIS_ONLY = "analysis-only"
STAGE_EXECUTABLE = "stage-executable"
DATA_PARALLEL_KEYS = ("data_parallel_size", "dp")
PARALLELISM_KEYS = (
    ("data_parallel_size", "dp"),
    ("tensor_model_parallel_size", "tp"),
    ("pipeline_model_parallel_size", "pp"),
    ("context_parallel_size", "cp"),
    ("expert_model_parallel_size", "ep"),
)


@dataclass(frozen=True)
class PlanValidationResult:
    runtime_mode: Literal["stage-executable", "analysis-only"]


def validate_model_plan(plan: ModelPlan) -> PlanValidationResult:
    runtime_mode = _runtime_mode(plan)
    _validate_stage_ids(plan)
    _validate_stage_segments(plan)
    _validate_global_coverage(plan)
    _validate_parallelism(plan)
    _validate_global_batch_size(plan)
    _validate_explicit_device_groups(plan)
    if runtime_mode == STAGE_EXECUTABLE:
        _validate_stage_executable_constraints(plan)
    return PlanValidationResult(runtime_mode=runtime_mode)


def _runtime_mode(plan: ModelPlan) -> Literal["stage-executable", "analysis-only"]:
    if plan.stages and (
        all(len(stage.segments) == 1 for stage in plan.stages) or _is_homogeneous_vpp_plan(plan)
    ):
        return STAGE_EXECUTABLE
    return ANALYSIS_ONLY


def _validate_stage_ids(plan: ModelPlan) -> None:
    stage_ids = [stage.stage_id for stage in plan.stages]
    if len(set(stage_ids)) != len(stage_ids):
        raise ValueError("stage_id values must be unique")
    for index, stage_id in enumerate(stage_ids):
        if index == 0 and stage_id != 0:
            raise ValueError("stage_id values must start from 0")
        if index > 0 and stage_id != index:
            raise ValueError("stage_id values must be continuous in plan order")


def _validate_stage_segments(plan: ModelPlan) -> None:
    for stage in plan.stages:
        _ensure_stage_segments_are_valid(stage, plan.total_layers)


def _ensure_stage_segments_are_valid(stage: StagePlan, total_layers: int | None) -> None:
    previous_end: int | None = None
    previous_start: int | None = None
    allow_stage_gaps = _allows_stage_segment_gaps(stage)
    for segment in stage.segments:
        _ensure_segment_bounds(stage.stage_id, segment, total_layers)
        if previous_start is not None and segment.start < previous_start:
            raise ValueError(f"stage {stage.stage_id} segment order is invalid")
        if previous_end is not None:
            expected_start = previous_end + 1
            if segment.start < expected_start:
                raise ValueError(f"stage {stage.stage_id} segment overlap detected")
            if segment.start > expected_start and not allow_stage_gaps:
                raise ValueError(f"stage {stage.stage_id} segment gap detected")
        previous_start = segment.start
        previous_end = segment.end


def _ensure_segment_bounds(stage_id: int, segment: SegmentPlan, total_layers: int | None) -> None:
    if segment.start < 0 or segment.end < segment.start:
        raise ValueError(f"stage {stage_id} segment [{segment.start}, {segment.end}] is out of bounds")
    if total_layers is not None and segment.end >= total_layers:
        raise ValueError(f"stage {stage_id} segment [{segment.start}, {segment.end}] is out of bounds")


def _validate_global_coverage(plan: ModelPlan) -> None:
    flattened_segments = sorted(
        (segment for stage in plan.stages for segment in stage.segments),
        key=lambda segment: (segment.start, segment.end),
    )
    if not flattened_segments:
        return
    previous_end: int | None = None
    for segment in flattened_segments:
        if previous_end is None:
            if segment.start != 0:
                raise ValueError(f"global layer coverage gap detected before layer {segment.start}")
        else:
            expected_start = previous_end + 1
            if segment.start < expected_start:
                raise ValueError(f"global layer coverage overlap detected at layer {segment.start}")
            if segment.start > expected_start:
                raise ValueError(f"global layer coverage gap detected before layer {segment.start}")
        previous_end = segment.end
    if plan.total_layers is not None and previous_end != plan.total_layers - 1:
        raise ValueError(f"global layer coverage gap detected before layer {previous_end + 1}")


def _validate_parallelism(plan: ModelPlan) -> None:
    for stage in plan.stages:
        for index, segment in enumerate(stage.segments):
            for keys in PARALLELISM_KEYS:
                value = _strategy_int(segment.strategy, keys)
                if value is None:
                    continue
                if value <= 0:
                    raise ValueError(
                        f"stage {stage.stage_id} segment {index} parallelism {keys[0]} must be positive"
                    )


def _validate_global_batch_size(plan: ModelPlan) -> None:
    contract = plan.contract
    if contract is None or contract.global_batch_size is None or not plan.stages:
        return
    if _is_stage_runtime_bridge_plan(plan):
        _validate_stage_heterogeneous_global_batch_size(plan, contract)
        return
    for stage in plan.stages:
        for index, segment in enumerate(stage.segments):
            data_parallel_size = _required_strategy_int(segment.strategy, DATA_PARALLEL_KEYS)
            if data_parallel_size <= 0:
                raise ValueError(
                    f"stage {stage.stage_id} segment {index} parallelism data_parallel_size must be positive"
                )
            expected = (
                data_parallel_size
                * contract.micro_batch_size
                * contract.gradient_accumulation_steps
            )
            if expected != contract.global_batch_size:
                raise ValueError(
                    f"stage {stage.stage_id} segment {index} global batch size mismatch: "
                    f"expected {expected}, got {contract.global_batch_size}"
                )


def _validate_stage_executable_constraints(plan: ModelPlan) -> None:
    used_ranks = _validate_required_stage_device_groups(plan)
    if _is_stage_runtime_bridge_plan(plan):
        _validate_stage_runtime_fields(plan)
        _validate_stage_runtime_mesh_sizes(plan)
    else:
        _validate_homogeneous_stage_parallelism(plan)
        _validate_stage_device_group_capacity(plan)
    if plan.contract is not None and len(used_ranks) != plan.contract.world_size:
        raise ValueError("device_group union size must equal contract world_size")


def _validate_homogeneous_stage_parallelism(plan: ModelPlan) -> None:
    expected_dp_size: int | None = None
    for stage in plan.stages:
        segment = stage.segments[0]
        dp_size = _required_strategy_int(segment.strategy, DATA_PARALLEL_KEYS)
        if expected_dp_size is None:
            expected_dp_size = dp_size
        elif dp_size != expected_dp_size:
            raise ValueError("data_parallel_size must be consistent across stages")
        pp_size = _strategy_int(segment.strategy, ("pipeline_model_parallel_size", "pp"))
        if pp_size is not None and pp_size != len(plan.stages):
            raise ValueError("pipeline_model_parallel_size must equal len(plan.stages)")


def _validate_stage_device_group_capacity(plan: ModelPlan) -> None:
    for stage in plan.stages:
        required_ranks = _required_parallel_ranks(stage.segments[0].strategy)
        if required_ranks > len(stage.device_group):
            raise ValueError(
                f"stage {stage.stage_id} device_group is too small for tensor/context/expert parallelism"
            )


def _validate_stage_runtime_mesh_sizes(plan: ModelPlan) -> None:
    for stage in plan.stages:
        required_ranks = _required_parallel_ranks(stage.segments[0].strategy)
        if required_ranks != len(stage.device_group):
            raise ValueError(f"stage {stage.stage_id} device_group must match stage mesh size")


def _validate_explicit_device_groups(plan: ModelPlan) -> None:
    used_ranks: set[int] = set()
    for stage in plan.stages:
        if not stage.device_group:
            continue
        _validate_device_group_ranks(stage.device_group, plan.contract)
        if len(set(stage.device_group)) != len(stage.device_group):
            raise ValueError(f"stage {stage.stage_id} device_group contains duplicate ranks")
        overlap = used_ranks.intersection(stage.device_group)
        if overlap:
            raise ValueError(f"stage {stage.stage_id} device_group overlap across stages: {sorted(overlap)}")
        used_ranks.update(stage.device_group)


def _validate_required_stage_device_groups(plan: ModelPlan) -> set[int]:
    used_ranks: set[int] = set()
    for stage in plan.stages:
        if not stage.device_group:
            raise ValueError(f"stage {stage.stage_id} device_group must be non-empty")
        used_ranks.update(stage.device_group)
    return used_ranks


def _validate_stage_heterogeneous_global_batch_size(plan: ModelPlan, contract) -> None:
    first_dp = _required_strategy_int(plan.stages[0].segments[0].strategy, DATA_PARALLEL_KEYS)
    expected = first_dp * contract.micro_batch_size * contract.gradient_accumulation_steps
    if expected != contract.global_batch_size:
        raise ValueError(
            "stage-heterogeneous global batch size mismatch: "
            f"expected {expected}, got {contract.global_batch_size}"
        )
    base_unit = first_dp * contract.micro_batch_size
    for stage in plan.stages[1:]:
        stage_dp = _required_strategy_int(stage.segments[0].strategy, DATA_PARALLEL_KEYS)
        if base_unit % stage_dp != 0:
            raise ValueError(
                "stage-heterogeneous data_parallel_size must divide "
                "first_stage_dp * micro_batch_size"
            )


def _validate_stage_runtime_fields(plan: ModelPlan) -> None:
    stage_strategies = [dict(stage.segments[0].strategy) for stage in plan.stages]
    if _has_inconsistent_field(stage_strategies, "use_distributed_optimizer"):
        raise ValueError("use_distributed_optimizer must be consistent across stages")
    if _has_inconsistent_field(stage_strategies, "context_parallel_size"):
        raise ValueError("context_parallel_size must be consistent across stages")
    if _has_inconsistent_field(stage_strategies, "recompute_signature"):
        raise ValueError("recompute configuration must be consistent across stages")
    tp_values = {_required_strategy_int(strategy, ("tensor_model_parallel_size", "tp")) for strategy in stage_strategies}
    if len(tp_values) > 1 and not all(strategy.get("sequence_parallel") is True for strategy in stage_strategies):
        raise ValueError("sequence_parallel must be enabled when tensor parallelism differs across stages")


def _required_parallel_ranks(strategy: dict[str, object]) -> int:
    dp_size = _required_strategy_int(strategy, ("data_parallel_size", "dp"))
    tp_size = _required_strategy_int(strategy, ("tensor_model_parallel_size", "tp"))
    cp_size = _required_strategy_int(strategy, ("context_parallel_size", "cp"))
    return dp_size * tp_size * cp_size


def _has_inconsistent_field(stage_strategies, field: str) -> bool:
    values = {_field_signature(strategy, field) for strategy in stage_strategies}
    return len(values) > 1


def _field_signature(strategy: dict[str, object], field: str):
    if field == "recompute_signature":
        return (
            strategy.get("use_recompute"),
            strategy.get("recompute_method"),
            strategy.get("recompute_granularity"),
            strategy.get("recompute_num_layers"),
        )
    return strategy.get(field)


def _validate_device_group_ranks(
    device_group: tuple[object, ...],
    contract: object | None,
) -> None:
    world_size = getattr(contract, "world_size", None)
    for rank in device_group:
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError("device_group ranks must be int")
        if rank < 0:
            raise ValueError("device_group ranks must be >= 0")
        if world_size is not None and rank >= world_size:
            raise ValueError("device_group ranks must be < contract.world_size")


def _required_strategy_int(strategy: dict[str, object], keys: tuple[str, ...]) -> int:
    value = _strategy_int(strategy, keys)
    if value is None:
        raise ValueError(f"missing required strategy field {keys[0]}")
    return value


def _strategy_int(strategy: dict[str, object], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = strategy.get(key)
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
    return None


def _allows_stage_segment_gaps(stage: StagePlan) -> bool:
    if not stage.segments:
        return False
    value = stage.segments[0].strategy.get("num_layers_per_virtual_pipeline_stage")
    return isinstance(value, int) and value > 0


def _is_homogeneous_vpp_plan(plan: ModelPlan) -> bool:
    stage_segments = [segment for stage in plan.stages for segment in stage.segments]
    if not stage_segments:
        return False
    baseline = dict(stage_segments[0].strategy)
    chunk_layers = baseline.get("num_layers_per_virtual_pipeline_stage")
    if not isinstance(chunk_layers, int) or chunk_layers <= 0:
        return False
    return all(dict(segment.strategy) == baseline for segment in stage_segments[1:])


def _is_stage_heterogeneous_plan(plan: ModelPlan) -> bool:
    if _is_homogeneous_vpp_plan(plan):
        return False
    baseline = dict(plan.stages[0].segments[0].strategy)
    return any(dict(stage.segments[0].strategy) != baseline for stage in plan.stages[1:])


def _is_stage_runtime_bridge_plan(plan: ModelPlan) -> bool:
    return any(stage.segments[0].strategy.get("stage_runtime_bridge") is True for stage in plan.stages)


__all__ = ["PlanValidationResult", "validate_model_plan"]
