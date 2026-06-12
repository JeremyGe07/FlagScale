import logging

logger = logging.getLogger("FlagScale-AutoTuner")


def _format_memory_breakdown(strategy):
    breakdown = strategy.get("memory_breakdown")
    if not isinstance(breakdown, dict):
        return ""
    reserved_mb = breakdown.get("reserved_mb")
    peak_mb = breakdown.get("peak_mb")
    peak_bias_mb = breakdown.get("peak_activation_bias_mb")
    profiled_total_mb = breakdown.get("profiled_memory_total_mb")
    return (
        f" reserved_mb={reserved_mb}, peak_mb={peak_mb}, "
        f"peak_activation_bias_mb={peak_bias_mb}, "
        f"profiled_memory_total_mb={profiled_total_mb}."
    )


def prune_by_memory_model(config, strategy, history=[]):
    if "memory_model" in strategy:
        upper_bound = config.experiment.auto_tuner.memory_model.gpu_memory
        if strategy["memory_model"] > upper_bound:
            logger.info(
                f"The strategy {strategy} has been pruned by modeling memory "
                f"{int(strategy['memory_model'])}MB (>{int(upper_bound)}MB)."
                f"{_format_memory_breakdown(strategy)}"
            )
            strategy["max_mem"] = "OOM"
            strategy["performance"] = None
            strategy["pruned"] = True
            return True
    return False


def prune_by_memory_model_util(config, strategy, history=[]):
    if "memory_model" in strategy:
        lower_bound = (
            config.experiment.auto_tuner.memory_model.gpu_memory * strategy["gpu_utilization"][0]
        )
        upper_bound = (
            config.experiment.auto_tuner.memory_model.gpu_memory * strategy["gpu_utilization"][1]
        )
        if strategy["memory_model"] > upper_bound or strategy["memory_model"] < lower_bound:
            logger.info(
                f"The strategy {strategy} has been pruned by modeling memory util and its memory "
                f"is {int(strategy['memory_model'])}MB (>{int(upper_bound)}MB or "
                f"<{int(lower_bound)}MB).{_format_memory_breakdown(strategy)}"
            )
            strategy["max_mem"] = None
            strategy["performance"] = None
            strategy["pruned"] = True
            return True
    return False
