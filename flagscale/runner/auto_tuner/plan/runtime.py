from flagscale.runner.auto_tuner.plan.lowering import lower_strategy_to_plan
from flagscale.runner.auto_tuner.plan.segment_runtime_contract import build_segment_runtime_contract
from flagscale.runner.auto_tuner.plan.summary import (
    SEGMENT_HETEROGENEOUS_PLAN,
    STAGE_HETEROGENEOUS_PLAN,
    plan_kind,
)
from flagscale.runner.auto_tuner.plan.validator import validate_model_plan

MESH_PP_SIZE = 1
RECOMPUTE_KEYS = (
    "use_recompute",
    "recompute_method",
    "recompute_granularity",
    "recompute_num_layers",
)


def build_stage_hetero_runtime_overrides(strategy, config):
    if "stage_partition_ranges" not in strategy or "stage_strategies" not in strategy:
        return None
    plan = lower_strategy_to_plan(strategy, config)
    if plan_kind(plan) not in {STAGE_HETEROGENEOUS_PLAN, SEGMENT_HETEROGENEOUS_PLAN}:
        return None
    if plan_kind(plan) == STAGE_HETEROGENEOUS_PLAN:
        stage_strategies = [dict(stage.segments[0].strategy) for stage in plan.stages]
        layer_split = _layer_split(plan)
    else:
        stage_strategies = _segment_stage_strategies(strategy)
        layer_split = _stage_layer_split(strategy)
    return _build_stage_shell_overrides(stage_strategies, layer_split, config)


def build_segment_hetero_runtime_overrides(strategy, config):
    if "stage_partition_ranges" not in strategy or "stage_strategies" not in strategy:
        return None
    plan = lower_strategy_to_plan(strategy, config)
    if plan_kind(plan) != SEGMENT_HETEROGENEOUS_PLAN:
        return None
    validate_model_plan(plan)
    return {"hetero": {"segment_runtime": build_segment_runtime_contract(plan)}}


def _validate_stage_runtime_support(stage_strategies):
    signatures = {_recompute_signature(strategy) for strategy in stage_strategies}
    if len(signatures) > 1:
        raise ValueError("stage-heterogeneous recompute configuration must be consistent")


def _resolve_device_types(stage_strategies, config):
    explicit = [strategy.get("device_type") for strategy in stage_strategies]
    if all(device_type is not None for device_type in explicit):
        return explicit
    configured = config.train.system.get("hetero", {}).get("hetero_device_types")
    if configured is not None:
        if len(configured) != len(stage_strategies):
            raise ValueError("hetero_device_types must match stage count")
        return list(configured)
    profile = config.experiment.auto_tuner.get("chip_profile", {}).get("profile")
    if profile is None:
        raise ValueError(
            "stage-heterogeneous runtime lowering requires hetero_device_types or chip profile"
        )
    device_type = profile["identity"]["name"]
    return [device_type] * len(stage_strategies)


def _resolve_current_device_type(config, device_types):
    current = config.train.system.get("hetero", {}).get("hetero_current_device_type")
    if current is not None:
        return current
    unique = set(device_types)
    if len(unique) != 1:
        raise ValueError("hetero_current_device_type is required when stage device types differ")
    return device_types[0]


def _layer_split(plan):
    return [stage.segments[0].end - stage.segments[0].start + 1 for stage in plan.stages]


def _stage_layer_split(strategy):
    return [end - start + 1 for start, end in strategy["stage_partition_ranges"]]


def _segment_stage_strategies(strategy):
    return [
        dict(stage_strategy["segment_strategies"][0])
        for stage_strategy in strategy["stage_strategies"]
    ]


def _build_stage_shell_overrides(stage_strategies, layer_split, config):
    _validate_stage_runtime_support(stage_strategies)
    device_types = _resolve_device_types(stage_strategies, config)
    return {
        "hetero": {
            "enable_hetero": True,
            "hetero_pipeline_layer_split": layer_split,
            "hetero_process_meshes": _flatten_meshes(stage_strategies),
            "hetero_device_types": device_types,
            "hetero_current_device_type": _resolve_current_device_type(config, device_types),
        },
        "system": {
            "pipeline_model_parallel_size": len(stage_strategies),
            "tensor_model_parallel_size": _required_int(
                stage_strategies[0],
                "tensor_model_parallel_size",
            ),
            "context_parallel_size": _required_int(stage_strategies[0], "context_parallel_size"),
            "expert_model_parallel_size": _required_int(
                stage_strategies[0],
                "expert_model_parallel_size",
            ),
            "sequence_parallel": any(
                item.get("sequence_parallel") is True for item in stage_strategies
            ),
        },
    }


def _flatten_meshes(stage_strategies):
    values = []
    for strategy in stage_strategies:
        values.extend(
            [
                _required_int(strategy, "tensor_model_parallel_size"),
                _required_int(strategy, "context_parallel_size"),
                _required_int(strategy, "expert_model_parallel_size"),
                _required_int(strategy, "data_parallel_size"),
                MESH_PP_SIZE,
            ]
        )
    return values


def _recompute_signature(strategy):
    return tuple(strategy.get(key) for key in RECOMPUTE_KEYS)


def _required_int(strategy, key):
    value = strategy.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"stage-heterogeneous runtime lowering requires positive int {key}")
    return value


__all__ = ["build_stage_hetero_runtime_overrides", "build_segment_hetero_runtime_overrides"]
