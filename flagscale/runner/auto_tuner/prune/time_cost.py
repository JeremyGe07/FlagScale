import logging

from flagscale.runner.auto_tuner.search.time_cost_pruning import (
    REPLENISH_REASON,
    should_replenish_time_cost_pruned_strategy,
)

logger = logging.getLogger("FlagScale-AutoTuner")


def prune_by_time_cost(strategy, history=None, config=None):
    if not strategy.get("time_cost_pruned", False):
        return False
    if should_replenish_time_cost_pruned_strategy(strategy, history, config):
        logger.info(
            "The strategy %s has been released by profiled time-cost replenishment: %s.",
            strategy,
            REPLENISH_REASON,
        )
        strategy["time_cost_replenished"] = True
        strategy["time_cost_replenish_reason"] = REPLENISH_REASON
        return False
    reason = strategy.get("time_cost_prune_reason", "time_cost.family_topk")
    logger.info(
        "The strategy %s has been pruned by profiled time cost: %s.",
        strategy,
        reason,
    )
    strategy["max_mem"] = None
    strategy["performance"] = None
    strategy["pruned"] = True
    return True
