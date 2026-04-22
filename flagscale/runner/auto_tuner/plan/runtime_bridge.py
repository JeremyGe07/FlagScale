from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.plan.runtime import (
    build_segment_hetero_runtime_overrides,
    build_stage_hetero_runtime_overrides,
)


def apply_hetero_runtime_overrides(strategy, config, runtime_mode):
    if runtime_mode == "stage-executable":
        overrides = build_stage_hetero_runtime_overrides(strategy, config)
    elif runtime_mode == "segment-executable":
        overrides = build_segment_hetero_runtime_overrides(strategy, config)
    else:
        raise ValueError(f"unsupported runtime_mode for hetero overrides: {runtime_mode}")
    if overrides is None:
        return
    config.train.system.hetero = OmegaConf.merge(
        config.train.system.get("hetero", {}),
        overrides["hetero"],
    )
    if "system" in overrides:
        config.train.system = OmegaConf.merge(config.train.system, overrides["system"])
