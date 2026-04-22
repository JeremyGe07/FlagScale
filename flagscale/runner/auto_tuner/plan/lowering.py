from flagscale.runner.auto_tuner.plan.schema import (
    ExecutionContract,
    ModelPlan,
    SegmentPlan,
    StagePlan,
    TransitionPlan,
)
from flagscale.runner.auto_tuner.plan.summary import summarize_plan
from flagscale.runner.auto_tuner.plan.validator import validate_model_plan


def build_execution_contract(strategy, config) -> ExecutionContract:
    return ExecutionContract(
        world_size=_resolve_world_size(config),
        micro_batch_size=strategy["micro_batch_size"],
        gradient_accumulation_steps=strategy["acc_step"],
        global_batch_size=config.train.model.global_batch_size,
    )


def lower_strategy_to_plan(strategy, config, validate=True) -> ModelPlan:
    num_layers = strategy.get("num_layers", config.train.model.num_layers)
    stage_segments, stage_device_groups, transitions = _build_plan_stages(
        num_layers,
        strategy,
        config,
    )

    plan = ModelPlan(
        stages=tuple(
            StagePlan(stage_id=stage_id, segments=segments, device_group=device_group)
            for stage_id, (segments, device_group) in enumerate(
                zip(stage_segments, stage_device_groups, strict=True)
            )
        ),
        transitions=transitions,
        contract=build_execution_contract(strategy, config),
        total_layers=num_layers,
    )
    if validate:
        validate_model_plan(plan)
    return plan


def _resolve_world_size(config) -> int:
    auto_tuner = config.experiment.auto_tuner
    if "cards" in auto_tuner:
        return int(auto_tuner.cards)
    runner = config.experiment.runner
    return int(runner.nnodes) * int(runner.nproc_per_node)


def _build_plan_stages(num_layers: int, strategy, config):
    if _has_explicit_stage_metadata(strategy):
        return _build_explicit_stage_plan(num_layers, strategy, config)
    pp_size = strategy["pipeline_model_parallel_size"]
    stage_device_groups = _contiguous_stage_groups(_resolve_world_size(config), pp_size)
    stage_segments = _build_stage_segments(num_layers, strategy)
    return stage_segments, stage_device_groups, ()


def _build_stage_segments(num_layers: int, strategy) -> tuple[tuple[SegmentPlan, ...], ...]:
    if _uses_vpp(strategy):
        return _build_interleaved_stage_segments(num_layers, strategy)
    return _build_stage_segments_without_vpp(num_layers, strategy)


def _build_stage_segments_without_vpp(
    num_layers: int,
    strategy,
) -> tuple[tuple[SegmentPlan, ...], ...]:
    stage_segments = []
    layer_start = 0
    frozen_strategy = dict(strategy)
    for layer_count in _stage_layer_counts(num_layers, strategy):
        layer_end = layer_start + layer_count - 1
        stage_segments.append(
            (
                SegmentPlan(
                    start=layer_start,
                    end=layer_end,
                    strategy=frozen_strategy,
                ),
            )
        )
        layer_start = layer_end + 1
    return tuple(stage_segments)


def _build_interleaved_stage_segments(
    num_layers: int,
    strategy,
) -> tuple[tuple[SegmentPlan, ...], ...]:
    pp_size = strategy["pipeline_model_parallel_size"]
    chunk_layers = strategy["num_layers_per_virtual_pipeline_stage"]
    if num_layers % pp_size != 0:
        raise ValueError(
            "VPP lowering requires num_layers divisible by pipeline_model_parallel_size"
        )
    layers_per_stage = num_layers // pp_size
    if layers_per_stage % chunk_layers != 0:
        raise ValueError(
            "VPP lowering requires stage layers divisible by num_layers_per_virtual_pipeline_stage"
        )
    total_chunks = num_layers // chunk_layers
    frozen_strategy = dict(strategy)
    stage_segments = []
    for stage_id in range(pp_size):
        segments = []
        for chunk_index in range(stage_id, total_chunks, pp_size):
            layer_start = chunk_index * chunk_layers
            layer_end = layer_start + chunk_layers - 1
            segments.append(
                SegmentPlan(
                    start=layer_start,
                    end=layer_end,
                    strategy=frozen_strategy,
                )
            )
        stage_segments.append(tuple(segments))
    return tuple(stage_segments)


def _build_explicit_stage_plan(num_layers: int, strategy, config):
    stage_ranges = tuple(tuple(item) for item in strategy["stage_partition_ranges"])
    stage_strategies = tuple(dict(item) for item in strategy["stage_strategies"])
    if len(stage_ranges) != len(stage_strategies):
        raise ValueError(
            "stage_partition_ranges and stage_strategies must have the same length"
        )
    stage_device_groups = strategy.get("stage_device_groups")
    if stage_device_groups is None:
        device_groups = _contiguous_stage_groups_for_stage_strategies(stage_strategies, config)
    else:
        device_groups = tuple(tuple(group) for group in stage_device_groups)
    if len(device_groups) != len(stage_ranges):
        raise ValueError("stage_device_groups must match stage count")
    stage_device_types = strategy.get("stage_device_types")
    stage_segments = []
    transitions = []
    for index, (stage_range, stage_strategy) in enumerate(
        zip(stage_ranges, stage_strategies, strict=True)
    ):
        if len(stage_range) != 2:
            raise ValueError("stage_partition_ranges must contain [start, end] pairs")
        segments, stage_transitions = _build_explicit_stage_segments(
            stage_id=index,
            stage_range=stage_range,
            stage_strategy=stage_strategy,
            stage_device_type=None if stage_device_types is None else stage_device_types[index],
        )
        stage_segments.append(segments)
        transitions.extend(stage_transitions)
    return tuple(stage_segments), device_groups, tuple(transitions)


def _build_explicit_stage_segments(
    stage_id: int,
    stage_range,
    stage_strategy,
    stage_device_type=None,
):
    if _has_explicit_segment_metadata(stage_strategy):
        return _build_segmented_stage_segments(
            stage_id,
            stage_range,
            stage_strategy,
            stage_device_type,
        )
    return _build_stage_runtime_bridge_segments(stage_range, stage_strategy, stage_device_type)


def _build_segmented_stage_segments(
    stage_id: int,
    stage_range,
    stage_strategy,
    stage_device_type=None,
):
    segment_ranges = tuple(tuple(item) for item in stage_strategy["segment_partition_ranges"])
    segment_strategies = tuple(dict(item) for item in stage_strategy["segment_strategies"])
    if len(segment_ranges) != len(segment_strategies):
        raise ValueError(
            "segment_partition_ranges and segment_strategies must have the same length"
        )
    if not segment_ranges:
        raise ValueError("segment_partition_ranges must not be empty")
    start_offset = int(stage_range[0])
    segments = []
    for segment_range, segment_strategy in zip(segment_ranges, segment_strategies, strict=True):
        if len(segment_range) != 2:
            raise ValueError("segment_partition_ranges must contain [start, end] pairs")
        if stage_device_type is not None and "device_type" not in segment_strategy:
            segment_strategy["device_type"] = stage_device_type
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
                    "source_mesh": _segment_mesh_metadata(source_segment.strategy),
                    "target_mesh": _segment_mesh_metadata(target_segment.strategy),
                },
            )
        )
    return tuple(segments), tuple(transitions)


def _build_stage_runtime_bridge_segments(stage_range, stage_strategy, stage_device_type=None):
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


def _stage_layer_counts(num_layers: int, strategy) -> tuple[int, ...]:
    pp_size = strategy["pipeline_model_parallel_size"]
    if pp_size == 1:
        return (num_layers,)
    first_layers = strategy.get("decoder_first_pipeline_num_layers")
    last_layers = strategy.get("decoder_last_pipeline_num_layers")
    if first_layers is None and last_layers is None and num_layers % pp_size == 0:
        return tuple([num_layers // pp_size] * pp_size)
    if pp_size == 2:
        if first_layers is None and last_layers is None:
            first_layers, last_layers = _default_edge_layer_counts(num_layers, pp_size)
        first_layers = num_layers - last_layers if first_layers is None else first_layers
        last_layers = num_layers - first_layers
        return (first_layers, last_layers)
    default_first, default_last = _default_edge_layer_counts(num_layers, pp_size)
    first_layers = default_first if first_layers is None else first_layers
    last_layers = default_last if last_layers is None else last_layers
    middle_stages = pp_size - 2
    middle_total_layers = num_layers - first_layers - last_layers
    if middle_total_layers <= 0 or middle_total_layers % middle_stages != 0:
        raise ValueError("cannot derive homogeneous stage layer counts from strategy")
    middle_layers = middle_total_layers // middle_stages
    return (first_layers,) + tuple([middle_layers] * middle_stages) + (last_layers,)


def _default_edge_layer_counts(num_layers: int, pp_size: int) -> tuple[int, int]:
    average_layers = num_layers / pp_size
    middle_layers = round(average_layers)
    remaining_layers = num_layers - middle_layers * (pp_size - 2)
    if remaining_layers < 2:
        middle_layers -= 1
        remaining_layers = num_layers - middle_layers * (pp_size - 2)
    first_layers = remaining_layers // 2
    last_layers = remaining_layers - first_layers
    return first_layers, last_layers


def _contiguous_stage_groups_for_stage_strategies(stage_strategies, config):
    world_size = _resolve_world_size(config)
    counts = tuple(_required_stage_world_size(strategy) for strategy in stage_strategies)
    if sum(counts) != world_size:
        raise ValueError("stage strategy world sizes must sum to contract world_size")
    offset = 0
    groups = []
    for count in counts:
        groups.append(tuple(range(offset, offset + count)))
        offset += count
    return tuple(groups)


def _contiguous_stage_groups(world_size: int, pp_size: int) -> tuple[tuple[int, ...], ...]:
    if world_size % pp_size != 0:
        raise ValueError("world_size must be divisible by pipeline_model_parallel_size")
    stage_group_size = world_size // pp_size
    return tuple(
        tuple(range(offset, offset + stage_group_size))
        for offset in range(0, world_size, stage_group_size)
    )


def _has_explicit_stage_metadata(strategy) -> bool:
    return "stage_partition_ranges" in strategy and "stage_strategies" in strategy


def _has_explicit_segment_metadata(stage_strategy) -> bool:
    return "segment_partition_ranges" in stage_strategy and "segment_strategies" in stage_strategy


def _segment_mesh_metadata(strategy) -> dict[str, object]:
    return {
        "tensor_model_parallel_size": _required_strategy_int(
            strategy,
            ("tensor_model_parallel_size", "tp"),
        ),
        "context_parallel_size": _required_strategy_int(strategy, ("context_parallel_size", "cp")),
        "expert_model_parallel_size": _required_strategy_int(
            strategy,
            ("expert_model_parallel_size", "ep"),
        ),
        "data_parallel_size": _required_strategy_int(strategy, ("data_parallel_size", "dp")),
        "pp_local": _segment_local_pipeline_size(strategy),
    }


def _segment_local_pipeline_size(strategy) -> int:
    value = strategy.get("pp_local")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    value = strategy.get("pipeline_model_parallel_size")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 1


def _required_strategy_int(strategy, keys):
    for key in keys:
        value = strategy.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    raise ValueError(f"missing required strategy field {keys[0]}")


def _required_stage_world_size(strategy) -> int:
    return (
        int(strategy["data_parallel_size"])
        * int(strategy["tensor_model_parallel_size"])
        * int(strategy["context_parallel_size"])
    )


def _uses_vpp(strategy) -> bool:
    chunk_layers = strategy.get("num_layers_per_virtual_pipeline_stage")
    return isinstance(chunk_layers, int) and chunk_layers > 0


__all__ = ["build_execution_contract", "lower_strategy_to_plan", "summarize_plan"]
