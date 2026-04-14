from flagscale.runner.auto_tuner.plan.schema import (
    ExecutionContract,
    ModelPlan,
    SegmentPlan,
    StagePlan,
)
from flagscale.runner.auto_tuner.plan.validator import validate_model_plan


def build_execution_contract(strategy, config) -> ExecutionContract:
    world_size = _resolve_world_size(config)
    return ExecutionContract(
        world_size=world_size,
        micro_batch_size=strategy["micro_batch_size"],
        gradient_accumulation_steps=strategy["acc_step"],
        global_batch_size=config.train.model.global_batch_size,
    )


def lower_strategy_to_plan(strategy, config) -> ModelPlan:
    num_layers = strategy.get("num_layers", config.train.model.num_layers)
    pp_size = strategy["pipeline_model_parallel_size"]
    stage_layer_counts = _stage_layer_counts(num_layers, strategy)
    stage_device_groups = _contiguous_stage_groups(_resolve_world_size(config), pp_size)

    stages = []
    layer_start = 0
    frozen_strategy = dict(strategy)
    for stage_id, (layer_count, device_group) in enumerate(
        zip(stage_layer_counts, stage_device_groups, strict=True)
    ):
        layer_end = layer_start + layer_count - 1
        stages.append(
            StagePlan(
                stage_id=stage_id,
                segments=(
                    SegmentPlan(
                        start=layer_start,
                        end=layer_end,
                        strategy=frozen_strategy,
                    ),
                ),
                device_group=device_group,
            )
        )
        layer_start = layer_end + 1

    plan = ModelPlan(
        stages=tuple(stages),
        contract=build_execution_contract(strategy, config),
        total_layers=num_layers,
    )
    validation = validate_model_plan(plan)
    if validation.runtime_mode != "stage-executable":
        raise ValueError("homogeneous lowering must produce a stage-executable plan")
    return plan


def _resolve_world_size(config) -> int:
    auto_tuner = config.experiment.auto_tuner
    if "cards" in auto_tuner:
        return int(auto_tuner.cards)
    runner = config.experiment.runner
    return int(runner.nnodes) * int(runner.nproc_per_node)


def _stage_layer_counts(num_layers: int, strategy) -> tuple[int, ...]:
    pp_size = strategy["pipeline_model_parallel_size"]
    if pp_size == 1:
        return (num_layers,)
    first_layers = strategy.get("decoder_first_pipeline_num_layers")
    last_layers = strategy.get("decoder_last_pipeline_num_layers")
    if first_layers is None and last_layers is None and num_layers % pp_size == 0:
        return tuple([num_layers // pp_size] * pp_size)
    if pp_size == 2:
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
    stage_groups = []
    rank_start = 0
    for _ in range(pp_size):
        rank_end = rank_start + stage_group_size
        stage_groups.append(tuple(range(rank_start, rank_end)))
        rank_start = rank_end
    return tuple(stage_groups)


__all__ = ["build_execution_contract", "lower_strategy_to_plan"]
