from flagscale.runner.auto_tuner.plan.schema import SegmentPlan, TransitionPlan


def has_explicit_segment_metadata(stage_strategy) -> bool:
    return "segment_partition_ranges" in stage_strategy and "segment_strategies" in stage_strategy


def build_explicit_stage_segments(stage_id: int, stage_range, stage_strategy, stage_device_type=None):
    if has_explicit_segment_metadata(stage_strategy):
        return build_segmented_stage_segments(
            stage_id,
            stage_range,
            stage_strategy,
            stage_device_type,
        )
    if "segment_partition_ranges" in stage_strategy or "segment_strategies" in stage_strategy:
        raise ValueError(
            "segment_partition_ranges and segment_strategies must be provided together"
        )
    return build_stage_runtime_bridge_segments(stage_range, stage_strategy, stage_device_type)


def build_segmented_stage_segments(
    stage_id: int,
    stage_range,
    stage_strategy,
    stage_device_type=None,
):
    segment_ranges = tuple(tuple(item) for item in stage_strategy["segment_partition_ranges"])
    segment_strategies = tuple(dict(item) for item in stage_strategy["segment_strategies"])
    if len(segment_ranges) != len(segment_strategies):
        raise ValueError("segment_partition_ranges and segment_strategies must have the same length")
    if not segment_ranges:
        raise ValueError("segment_partition_ranges must not be empty")
    _validate_segment_ranges(stage_id, stage_range, segment_ranges)
    start_offset = int(stage_range[0])
    segments = []
    for segment_index, (segment_range, segment_strategy) in enumerate(
        zip(segment_ranges, segment_strategies, strict=True)
    ):
        if len(segment_range) != 2:
            raise ValueError("segment_partition_ranges must contain [start, end] pairs")
        if stage_device_type is not None and "device_type" not in segment_strategy:
            segment_strategy["device_type"] = stage_device_type
        _set_segment_local_pipeline_size(stage_id, segment_index, segment_strategy)
        segments.append(
            SegmentPlan(
                start=start_offset + int(segment_range[0]),
                end=start_offset + int(segment_range[1]),
                strategy=segment_strategy,
            )
        )
    transitions = []
    for source_index, (source_segment, target_segment) in enumerate(zip(segments, segments[1:])):
        transitions.append(
            TransitionPlan(
                source_stage_id=stage_id,
                target_stage_id=stage_id,
                kind="segment-redistribution",
                source_segment_index=source_index,
                target_segment_index=source_index + 1,
                metadata={
                    "source_mesh": segment_mesh_metadata(source_segment.strategy),
                    "target_mesh": segment_mesh_metadata(target_segment.strategy),
                },
            )
        )
    return tuple(segments), tuple(transitions)


def build_stage_runtime_bridge_segments(stage_range, stage_strategy, stage_device_type=None):
    segment_strategy = dict(stage_strategy)
    if stage_device_type is not None:
        segment_strategy["device_type"] = stage_device_type
    segment_strategy["stage_runtime_bridge"] = True
    segments = (
        SegmentPlan(
            start=int(stage_range[0]),
            end=int(stage_range[1]),
            strategy=segment_strategy,
        ),
    )
    return segments, ()


def segment_mesh_metadata(strategy) -> dict[str, object]:
    return {
        "tensor_model_parallel_size": required_strategy_int(
            strategy,
            ("tensor_model_parallel_size", "tp"),
        ),
        "context_parallel_size": required_strategy_int(strategy, ("context_parallel_size", "cp")),
        "expert_model_parallel_size": required_strategy_int(
            strategy,
            ("expert_model_parallel_size", "ep"),
        ),
        "data_parallel_size": required_strategy_int(strategy, ("data_parallel_size", "dp")),
        "pp_local": segment_local_pipeline_size(strategy),
    }


def segment_local_pipeline_size(strategy) -> int:
    value = strategy.get("pp_local")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    value = strategy.get("pipeline_model_parallel_size")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 1


def required_strategy_int(strategy, keys) -> int:
    for key in keys:
        value = strategy.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    raise ValueError(f"missing required strategy field {keys[0]}")


def required_stage_world_size(strategy) -> int:
    if "segment_strategies" in strategy:
        return required_nested_stage_world_size(strategy)
    return required_parallel_world_size(strategy)


def required_nested_stage_world_size(strategy) -> int:
    segment_strategies = strategy.get("segment_strategies")
    if not segment_strategies:
        raise ValueError("segment_strategies must not be empty")
    counts = {required_parallel_world_size(segment) for segment in segment_strategies}
    if len(counts) != 1:
        raise ValueError("segment_strategies must have consistent world size")
    return counts.pop()


def required_parallel_world_size(strategy) -> int:
    return (
        required_strategy_int(strategy, ("data_parallel_size", "dp"))
        * required_strategy_int(strategy, ("tensor_model_parallel_size", "tp"))
        * required_strategy_int(strategy, ("context_parallel_size", "cp"))
    )


def _validate_segment_ranges(stage_id, stage_range, segment_ranges):
    stage_start = int(stage_range[0])
    stage_end = int(stage_range[1])
    if stage_end < stage_start:
        raise ValueError(f"stage {stage_id} range is invalid")
    stage_length = stage_end - stage_start + 1
    expected_start = 0
    for segment_index, segment_range in enumerate(segment_ranges):
        rel_start = int(segment_range[0])
        rel_end = int(segment_range[1])
        if rel_start < 0 or rel_end < 0:
            raise ValueError("segment_partition_ranges must be non-negative")
        if rel_start > rel_end:
            raise ValueError("segment_partition_ranges must be ordered start <= end")
        if rel_start != expected_start:
            raise ValueError("segment_partition_ranges must be contiguous")
        if rel_end >= stage_length:
            raise ValueError("segment_partition_ranges must stay within stage range")
        expected_start = rel_end + 1
    if expected_start != stage_length:
        raise ValueError("segment_partition_ranges must exactly cover stage range")


def _set_segment_local_pipeline_size(stage_id, segment_index, segment_strategy):
    pp_local = segment_strategy.get("pp_local")
    pipeline_model_parallel_size = segment_strategy.get("pipeline_model_parallel_size")
    if pp_local is not None:
        if pp_local != 1:
            raise ValueError(
                f"stage {stage_id} segment {segment_index} segment-local pipeline size must be 1"
            )
        if pipeline_model_parallel_size is not None and pipeline_model_parallel_size != 1:
            raise ValueError(
                f"stage {stage_id} segment {segment_index} segment-local pipeline size must be 1"
            )
        return
    if pipeline_model_parallel_size is not None and pipeline_model_parallel_size != 1:
        raise ValueError(
            f"stage {stage_id} segment {segment_index} segment-local pipeline size must be 1"
        )
    segment_strategy["pp_local"] = 1
