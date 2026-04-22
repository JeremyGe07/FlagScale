from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.plan.runtime import (
    build_segment_hetero_runtime_overrides,
    build_stage_hetero_runtime_overrides,
)


def apply_hetero_runtime_overrides(strategy, config, runtime_mode):
    if runtime_mode == "stage-executable":
        overrides = build_stage_hetero_runtime_overrides(strategy, config)
    elif runtime_mode == "segment-executable":
        _require_segment_runtime_metadata(strategy)
        overrides = build_segment_hetero_runtime_overrides(strategy, config)
    else:
        raise ValueError(f"unsupported runtime_mode for hetero overrides: {runtime_mode}")
    if overrides is None:
        raise ValueError("segment runtime bridge did not materialize segment runtime overrides")
    config.train.system.hetero = OmegaConf.merge(
        config.train.system.get("hetero", {}),
        overrides["hetero"],
    )
    if "system" in overrides:
        config.train.system = OmegaConf.merge(config.train.system, overrides["system"])


def _require_segment_runtime_metadata(strategy):
    if "stage_partition_ranges" not in strategy or "stage_strategies" not in strategy:
        raise ValueError(
            "segment runtime bridge requires stage_partition_ranges and stage_strategies"
        )
