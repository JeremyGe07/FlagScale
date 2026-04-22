from collections.abc import Mapping

from flagscale.runner.auto_tuner.plan.schema import ModelPlan, SegmentPlan, StagePlan, TransitionPlan

SEGMENT_EXECUTABLE = "segment-executable"
SEGMENT_REDISRIBUTION = "segment-redistribution"
MESH_KEYS = (
    ("tensor_model_parallel_size", "tp"),
    ("context_parallel_size", "cp"),
    ("expert_model_parallel_size", "ep"),
    ("data_parallel_size", "dp"),
)
LOCAL_PIPELINE_KEYS = ("pp_local",)
GLOBAL_PIPELINE_KEYS = ("pipeline_model_parallel_size", "pp")


def validate_segment_executable_plan(plan: ModelPlan) -> None:
    for stage in plan.stages:
        _validate_stage_segments(stage)
    _validate_segment_transitions(plan)
    _validate_segment_global_batch_size(plan)


def _validate_stage_segments(stage: StagePlan) -> None:
    if not stage.segments:
        return
    _validate_stage_contiguity(stage)
    _validate_stage_mesh_consistency(stage)


def _validate_stage_contiguity(stage: StagePlan) -> None:
    previous_end = None
    previous_start = None
    for segment in stage.segments:
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


def _validate_stage_mesh_consistency(stage: StagePlan) -> None:
    reference = dict(stage.segments[0].strategy)
    tp_values: set[int] = set()
    for index, segment in enumerate(stage.segments):
        _validate_segment_mesh(stage.stage_id, index, segment, len(stage.device_group))
        _validate_fixed_fields(stage.stage_id, index, reference, segment)
        tp_values.add(_required_strategy_int(segment.strategy, ("tensor_model_parallel_size", "tp")))
    if len(tp_values) > 1 and not all(segment.strategy.get("sequence_parallel") is True for segment in stage.segments):
        raise ValueError("sequence_parallel must be enabled when tensor parallelism differs across segments")


def _validate_segment_mesh(stage_id: int, index: int, segment: SegmentPlan, device_count: int) -> None:
    pp_local = _segment_local_pipeline_size(segment.strategy)
    if pp_local != 1:
        raise ValueError(f"stage {stage_id} segment {index} segment-local pipeline size must be 1")
    mesh_size = pp_local
    for keys in MESH_KEYS:
        mesh_size *= _required_strategy_int(segment.strategy, keys)
    if mesh_size != device_count:
        raise ValueError(f"stage {stage_id} segment {index} mesh size must match device_group size")


def _validate_fixed_fields(
    stage_id: int,
    index: int,
    reference: Mapping[str, object],
    segment: SegmentPlan,
) -> None:
    if _strategy_value(segment.strategy, ("context_parallel_size", "cp")) != _strategy_value(
        reference, ("context_parallel_size", "cp")
    ):
        raise ValueError(f"stage {stage_id} segment {index} context_parallel_size must be consistent")
    if _strategy_value(segment.strategy, ("expert_model_parallel_size", "ep")) != _strategy_value(
        reference, ("expert_model_parallel_size", "ep")
    ):
        raise ValueError(f"stage {stage_id} segment {index} expert_model_parallel_size must be consistent")
    if _strategy_value(segment.strategy, ("use_distributed_optimizer",)) != _strategy_value(
        reference, ("use_distributed_optimizer",)
    ):
        raise ValueError(f"stage {stage_id} segment {index} use_distributed_optimizer must be consistent")
    if _strategy_value(segment.strategy, ("device_type",)) != _strategy_value(reference, ("device_type",)):
        raise ValueError(f"stage {stage_id} segment {index} device_type must be consistent")


def _validate_segment_transitions(plan: ModelPlan) -> None:
    expected = {
        (stage.stage_id, stage.stage_id, index, index + 1)
        for stage in plan.stages
        for index in range(len(stage.segments) - 1)
    }
    seen: set[tuple[int, int, int, int]] = set()
    for transition in plan.transitions:
        if transition.kind != SEGMENT_REDISRIBUTION:
            continue
        _validate_transition_metadata(transition)
        key = (
            transition.source_stage_id,
            transition.target_stage_id,
            transition.source_segment_index,
            transition.target_segment_index,
        )
        if key not in expected:
            raise ValueError("segment-redistribution transition required for segment-executable plans")
        seen.add(key)
    missing = sorted(expected - seen)
    if missing:
        stage_id, target_stage_id, source_index, target_index = missing[0]
        raise ValueError(
            "missing segment-redistribution transition for "
            f"stage {stage_id}->{target_stage_id} segments {source_index}->{target_index}"
        )


def _validate_transition_metadata(transition: TransitionPlan) -> None:
    if not isinstance(transition.metadata, Mapping):
        raise ValueError("segment-redistribution transition metadata must be a mapping")
    source_mesh = transition.metadata.get("source_mesh")
    target_mesh = transition.metadata.get("target_mesh")
    if not isinstance(source_mesh, Mapping) or not isinstance(target_mesh, Mapping):
        raise ValueError("segment-redistribution transition metadata must include source_mesh and target_mesh")
    _validate_mesh_metadata(source_mesh, "source_mesh")
    _validate_mesh_metadata(target_mesh, "target_mesh")


def _validate_mesh_metadata(mesh: Mapping[str, object], label: str) -> None:
    if _segment_local_pipeline_size(mesh) != 1:
        raise ValueError(f"{label} segment-local pipeline size must be 1")
    for keys in MESH_KEYS:
        _required_strategy_int(mesh, keys)


def _validate_segment_global_batch_size(plan: ModelPlan) -> None:
    contract = plan.contract
    if contract is None or contract.global_batch_size is None or not plan.stages:
        return
    if contract.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if contract.global_batch_size % contract.gradient_accumulation_steps != 0:
        raise ValueError("global_batch_size must be divisible by gradient_accumulation_steps")
    base = contract.global_batch_size // contract.gradient_accumulation_steps
    for stage in plan.stages:
        for index, segment in enumerate(stage.segments):
            dp_size = _required_strategy_int(segment.strategy, ("data_parallel_size", "dp"))
            if base % dp_size != 0:
                raise ValueError(
                    f"stage {stage.stage_id} segment {index} data_parallel_size must divide "
                    "global_batch_size / acc_step"
                )


def _required_strategy_int(strategy: Mapping[str, object], keys: tuple[str, ...]) -> int:
    value = _strategy_int(strategy, keys)
    if value is None:
        raise ValueError(f"missing required strategy field {keys[0]}")
    return value


def _strategy_int(strategy: Mapping[str, object], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = strategy.get(key)
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
    return None


def _strategy_value(strategy: Mapping[str, object], keys: tuple[str, ...]):
    for key in keys:
        value = strategy.get(key)
        if value is not None:
            return value
    return None


def _segment_local_pipeline_size(strategy: Mapping[str, object]) -> int:
    explicit = _strategy_int(strategy, LOCAL_PIPELINE_KEYS)
    if explicit is not None:
        return explicit
    return _required_strategy_int(strategy, GLOBAL_PIPELINE_KEYS)
