from flagscale.runner.auto_tuner.plan.schema import (
    ExecutionContract,
    ModelPlan,
    SegmentPlan,
    StagePlan,
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
    pp_size = strategy["pipeline_model_parallel_size"]
    stage_device_groups = _contiguous_stage_groups(_resolve_world_size(config), pp_size)
    stage_segments = _build_stage_segments(num_layers, strategy)

    plan = ModelPlan(
        stages=tuple(
            StagePlan(stage_id=stage_id, segments=segments, device_group=device_group)
            for stage_id, (segments, device_group) in enumerate(
                zip(stage_segments, stage_device_groups, strict=True)
            )
        ),
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
        raise ValueError("VPP lowering requires num_layers divisible by pipeline_model_parallel_size")
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


def _contiguous_stage_groups(world_size: int, pp_size: int) -> tuple[tuple[int, ...], ...]:
    if world_size % pp_size != 0:
        raise ValueError("world_size must be divisible by pipeline_model_parallel_size")
    stage_group_size = world_size // pp_size
    return tuple(
        tuple(range(offset, offset + stage_group_size))
        for offset in range(0, world_size, stage_group_size)
    )


def _uses_vpp(strategy) -> bool:
    chunk_layers = strategy.get("num_layers_per_virtual_pipeline_stage")
    return isinstance(chunk_layers, int) and chunk_layers > 0


__all__ = ["build_execution_contract", "lower_strategy_to_plan", "summarize_plan"]
