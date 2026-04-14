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
    _validate_stage_segments(plan)
    _validate_global_coverage(plan)
    _validate_parallelism(plan)
    _validate_global_batch_size(plan)
    if runtime_mode == STAGE_EXECUTABLE:
        _validate_stage_executable_constraints(plan)
    return PlanValidationResult(runtime_mode=runtime_mode)


def _runtime_mode(plan: ModelPlan) -> Literal["stage-executable", "analysis-only"]:
    if plan.stages and all(len(stage.segments) == 1 for stage in plan.stages):
        return STAGE_EXECUTABLE
    return ANALYSIS_ONLY


def _validate_stage_segments(plan: ModelPlan) -> None:
    for stage in plan.stages:
        _ensure_stage_segments_are_valid(stage, plan.total_layers)


def _ensure_stage_segments_are_valid(stage: StagePlan, total_layers: int | None) -> None:
    previous_end: int | None = None
    previous_start: int | None = None
    for segment in stage.segments:
        _ensure_segment_bounds(stage.stage_id, segment, total_layers)
        if previous_start is not None and segment.start < previous_start:
            raise ValueError(f"stage {stage.stage_id} segment order is invalid")
        if previous_end is not None:
            expected_start = previous_end + 1
            if segment.start < expected_start:
                raise ValueError(f"stage {stage.stage_id} segment overlap detected")
            if segment.start > expected_start:
                raise ValueError(f"stage {stage.stage_id} segment gap detected")
        previous_start = segment.start
        previous_end = segment.end


def _ensure_segment_bounds(stage_id: int, segment: SegmentPlan, total_layers: int | None) -> None:
    if segment.start < 0 or segment.end < segment.start:
        raise ValueError(f"stage {stage_id} segment [{segment.start}, {segment.end}] is out of bounds")
    if total_layers is not None and segment.end >= total_layers:
        raise ValueError(f"stage {stage_id} segment [{segment.start}, {segment.end}] is out of bounds")


def _validate_global_coverage(plan: ModelPlan) -> None:
    flattened_segments = [segment for stage in plan.stages for segment in stage.segments]
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
    if contract is None or contract.global_batch_size is None:
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
    _validate_stage_device_groups(plan)
    for stage in plan.stages:
        segment = stage.segments[0]
        pp_size = _strategy_int(segment.strategy, ("pipeline_model_parallel_size", "pp"))
        if pp_size is not None and pp_size != len(plan.stages):
            raise ValueError("pipeline_model_parallel_size must equal len(plan.stages)")
        required_ranks = _required_parallel_ranks(segment.strategy)
        if required_ranks > len(stage.device_group):
            raise ValueError(
                f"stage {stage.stage_id} device_group is too small for tensor/context/expert parallelism"
            )


def _validate_stage_device_groups(plan: ModelPlan) -> None:
    used_ranks: set[int] = set()
    for stage in plan.stages:
        if not stage.device_group:
            raise ValueError(f"stage {stage.stage_id} device_group must be non-empty")
        if len(set(stage.device_group)) != len(stage.device_group):
            raise ValueError(f"stage {stage.stage_id} device_group contains duplicate ranks")
        overlap = used_ranks.intersection(stage.device_group)
        if overlap:
            raise ValueError(f"stage {stage.stage_id} device_group overlap across stages: {sorted(overlap)}")
        used_ranks.update(stage.device_group)
    if plan.contract is not None and len(used_ranks) != plan.contract.world_size:
        raise ValueError("device_group union size must equal contract world_size")


def _required_parallel_ranks(strategy: dict[str, object]) -> int:
    tp_size = _required_strategy_int(strategy, ("tensor_model_parallel_size", "tp"))
    cp_size = _required_strategy_int(strategy, ("context_parallel_size", "cp"))
    ep_size = _required_strategy_int(strategy, ("expert_model_parallel_size", "ep"))
    return tp_size * cp_size * ep_size


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


__all__ = ["PlanValidationResult", "validate_model_plan"]
