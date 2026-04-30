from collections.abc import Mapping

from flagscale.runner.auto_tuner.plan.schema import StagePlan


def select_segment_stage_shell_strategy(stage: StagePlan) -> Mapping[str, object]:
    if not stage.segments:
        raise ValueError("segment stage shell requires at least one segment")
    return max(stage.segments, key=lambda segment: _tensor_parallel_size(segment.strategy)).strategy


def _tensor_parallel_size(strategy: Mapping[str, object]) -> int:
    return _strategy_int(strategy, ("tensor_model_parallel_size", "tp"))


def _strategy_int(strategy: Mapping[str, object], keys: tuple[str, ...]) -> int:
    for key in keys:
        value = strategy.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    raise ValueError(f"missing required strategy field {keys[0]}")


__all__ = ["select_segment_stage_shell_strategy"]
