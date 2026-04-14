from dataclasses import dataclass
from typing import Literal

from flagscale.runner.auto_tuner.plan.schema import ModelPlan

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
    is_valid: bool
    runtime_mode: Literal["stage-executable", "analysis-only"]
    errors: tuple[str, ...] = ()


def validate_model_plan(plan: ModelPlan) -> PlanValidationResult:
    runtime_mode = _runtime_mode(plan)
    errors = []
    errors.extend(_validate_layer_spans(plan))
    errors.extend(_validate_parallelism(plan))
    errors.extend(_validate_global_batch_size(plan))
    if runtime_mode == STAGE_EXECUTABLE:
        errors.extend(_validate_stage_device_groups(plan))
    return PlanValidationResult(
        is_valid=not errors,
        runtime_mode=runtime_mode,
        errors=tuple(errors),
    )


def _runtime_mode(plan: ModelPlan) -> Literal["stage-executable", "analysis-only"]:
    if plan.stages and all(len(stage.segments) == 1 for stage in plan.stages):
        return STAGE_EXECUTABLE
    return ANALYSIS_ONLY


def _validate_layer_spans(plan: ModelPlan) -> list[str]:
    errors = _validate_segment_bounds(plan)
    if plan.total_layers is None or errors:
        return errors
    ordered_segments = sorted(
        (segment for stage in plan.stages for segment in stage.segments),
        key=lambda segment: (segment.start, segment.end),
    )
    if not ordered_segments:
        return errors
    expected_start = 0
    for segment in ordered_segments:
        if segment.start > expected_start:
            errors.append(f"gap detected before layer {segment.start}")
        if segment.start < expected_start:
            errors.append(f"overlap detected at layer {segment.start}")
        expected_start = max(expected_start, segment.end + 1)
    if expected_start < plan.total_layers:
        errors.append(f"gap detected before layer {expected_start}")
    return errors


def _validate_segment_bounds(plan: ModelPlan) -> list[str]:
    errors = []
    for stage in plan.stages:
        for segment in stage.segments:
            if segment.start < 0 or segment.end < segment.start:
                errors.append(
                    f"stage {stage.stage_id} segment [{segment.start}, {segment.end}] is out of bounds"
                )
                continue
            if plan.total_layers is None:
                continue
            if segment.end >= plan.total_layers:
                errors.append(
                    f"stage {stage.stage_id} segment [{segment.start}, {segment.end}] is out of bounds"
                )
    return errors


def _validate_parallelism(plan: ModelPlan) -> list[str]:
    errors = []
    for stage in plan.stages:
        for index, segment in enumerate(stage.segments):
            for keys in PARALLELISM_KEYS:
                value = _strategy_int(segment.strategy, keys)
                if value is None:
                    continue
                if value <= 0:
                    errors.append(
                        f"stage {stage.stage_id} segment {index} parallelism {keys[0]} must be positive"
                    )
    return errors


def _validate_global_batch_size(plan: ModelPlan) -> list[str]:
    contract = plan.contract
    if contract is None or contract.global_batch_size is None:
        return []
    errors = []
    for stage in plan.stages:
        for index, segment in enumerate(stage.segments):
            data_parallel_size = _strategy_int(segment.strategy, DATA_PARALLEL_KEYS)
            if data_parallel_size is None or data_parallel_size <= 0:
                continue
            expected = (
                contract.micro_batch_size
                * contract.gradient_accumulation_steps
                * data_parallel_size
            )
            if expected != contract.global_batch_size:
                errors.append(
                    f"stage {stage.stage_id} segment {index} global batch size mismatch: "
                    f"expected {expected}, got {contract.global_batch_size}"
                )
    return errors


def _validate_stage_device_groups(plan: ModelPlan) -> list[str]:
    errors = []
    used_ranks: set[int] = set()
    for stage in plan.stages:
        if not stage.device_group:
            errors.append(f"stage {stage.stage_id} device_group must be non-empty")
            continue
        if len(set(stage.device_group)) != len(stage.device_group):
            errors.append(f"stage {stage.stage_id} device_group contains duplicate ranks")
        overlap = used_ranks.intersection(stage.device_group)
        if overlap:
            errors.append(
                f"stage {stage.stage_id} device_group overlap across stages: {sorted(overlap)}"
            )
        used_ranks.update(stage.device_group)
    if plan.contract is not None and len(used_ranks) != plan.contract.world_size:
        errors.append(
            "device_group union size must equal contract world_size"
        )
    return errors


def _strategy_int(strategy: dict[str, object], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = strategy.get(key)
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
    return None


__all__ = ["PlanValidationResult", "validate_model_plan"]
