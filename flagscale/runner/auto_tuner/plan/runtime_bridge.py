from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.plan.lowering import lower_strategy_to_plan
from flagscale.runner.auto_tuner.plan.runtime import (
    build_segment_hetero_runtime_overrides,
    build_stage_hetero_runtime_overrides,
)
from flagscale.runner.auto_tuner.plan.summary import (
    plan_kind,
    segment_count,
    summarize_execution_contract,
    summarize_plan,
)
from flagscale.runner.auto_tuner.plan.validator import validate_model_plan


def apply_hetero_runtime_overrides(strategy, config, runtime_mode):
    if runtime_mode == "stage-executable":
        overrides = build_stage_hetero_runtime_overrides(strategy, config)
    elif runtime_mode == "segment-executable":
        _require_segment_runtime_metadata(strategy)
        segment_overrides = build_segment_hetero_runtime_overrides(strategy, config)
        if segment_overrides is None:
            raise ValueError(
                "segment runtime bridge did not materialize segment runtime overrides"
            )
        stage_overrides = build_stage_hetero_runtime_overrides(strategy, config)
        if stage_overrides is None:
            raise ValueError("segment runtime bridge did not materialize stage overrides")
        overrides = OmegaConf.merge(stage_overrides, segment_overrides)
    else:
        raise ValueError(f"unsupported runtime_mode for hetero overrides: {runtime_mode}")
    if overrides is None:
        if runtime_mode == "segment-executable":
            raise ValueError(
                "segment runtime bridge did not materialize segment runtime overrides"
            )
        return
    config.train.system.hetero = OmegaConf.merge(
        config.train.system.get("hetero", {}),
        overrides["hetero"],
    )
    if "segment_runtime" in overrides["hetero"]:
        config.train.system.hetero.segment_runtime = overrides["hetero"]["segment_runtime"]
    if "system" in overrides:
        config.train.system = OmegaConf.merge(config.train.system, overrides["system"])


def validate_prefilled_plan_metadata(strategy, config, metadata):
    if strategy.get("plan_summary") is None:
        return
    if "stage_partition_ranges" not in strategy or "stage_strategies" not in strategy:
        return

    expected = _build_plan_metadata(strategy, config)
    mismatches = [
        key
        for key in (
            "plan_kind",
            "stage_count",
            "segment_count",
            "runtime_mode",
            "runtime_executable",
            "execution_contract",
            "plan_summary",
        )
        if metadata.get(key) != expected.get(key)
    ]
    if mismatches:
        raise ValueError(
            "prefilled plan metadata does not match raw segment strategy: {}".format(
                ", ".join(mismatches)
            )
        )


def _require_segment_runtime_metadata(strategy):
    if "stage_partition_ranges" not in strategy or "stage_strategies" not in strategy:
        raise ValueError(
            "segment runtime bridge requires stage_partition_ranges and stage_strategies"
        )


def _build_plan_metadata(strategy, config):
    plan = lower_strategy_to_plan(strategy, config)
    validation = validate_model_plan(plan)
    return {
        "plan_kind": plan_kind(plan),
        "stage_count": len(plan.stages),
        "segment_count": segment_count(plan),
        "runtime_mode": validation.runtime_mode,
        "runtime_executable": validation.runtime_mode == "stage-executable",
        "execution_contract": summarize_execution_contract(plan),
        "plan_summary": summarize_plan(plan),
    }
