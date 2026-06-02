import logging

logger = logging.getLogger("FlagScale-AutoTuner")


def prune_by_time_cost(strategy):
    if not strategy.get("time_cost_pruned", False):
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
