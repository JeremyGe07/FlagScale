from collections import defaultdict

DEFAULT_FAMILY_DIMS = (
    "data_parallel_size",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "sequence_parallel",
    "use_distributed_optimizer",
)
DEFAULT_MODE = "family_topk"
DEFAULT_PER_FAMILY_TOPK = 32
DEFAULT_MAX_REPLENISH_PER_FAMILY = 1
OOM_ERROR_MARKERS = ("OutOfMemoryError", "CUDA out of memory", "OOM|")
PRUNE_REASON = "time_cost.family_topk"
REPLENISH_REASON = "time_cost.family_topk.replenished_on_oom"
SOURCE_ALL = "all"
SOURCE_MEMORY_FEASIBLE = "memory_feasible"


def is_time_cost_pruning_enabled(config):
    return bool(_settings(config).get("enabled", False))


def is_family_replenish_on_oom_enabled(config):
    return bool(_settings(config).get("family_replenish_on_oom", False))


def mark_time_cost_pruned_strategies(strategies, config):
    settings = _settings(config)
    if not settings.get("enabled", False):
        return 0
    _validate_config(config, settings, strategies)

    topk = _per_family_topk(settings)
    family_dims = tuple(settings.get("family_dims", DEFAULT_FAMILY_DIMS))
    groups = _group_by_family(strategies, family_dims)
    pruned_count = 0
    for family_key, family in groups.items():
        ranked_source, source_name = _rank_family_source(family, config)
        _annotate_family(family, family_key, ranked_source, source_name, topk)
        keep_ids = {id(strategy) for strategy in ranked_source[:topk]}
        for strategy in family:
            if id(strategy) in keep_ids:
                continue
            _mark_strategy(strategy)
            pruned_count += 1
    return pruned_count


def should_replenish_time_cost_pruned_strategy(strategy, history, config):
    if not strategy.get("time_cost_pruned", False):
        return False
    settings = _settings(config)
    if not settings.get("family_replenish_on_oom", False):
        return False
    family_key, rank, topk = _strategy_family_metadata(strategy)
    if rank <= topk:
        return False
    if _replenished_family_count(history, family_key) >= _max_replenish_per_family(settings):
        return False
    topk_history = _topk_family_history(history, family_key, topk)
    if len(topk_history) < topk:
        return False
    return all(_is_oom_no_perf_failure(entry) for entry in topk_history)


def _settings(config):
    algo = config.experiment.auto_tuner.get("algo", {})
    return algo.get("time_cost_pruning", {})


def _validate_config(config, settings, strategies):
    if not config.experiment.auto_tuner.algo.get("use_profiled_time_cost", False):
        raise ValueError("time_cost_pruning requires algo.use_profiled_time_cost=true.")
    if settings.get("mode", DEFAULT_MODE) != DEFAULT_MODE:
        raise ValueError("Only time_cost_pruning.mode=family_topk is supported.")
    if any("time_cost" not in strategy for strategy in strategies):
        raise ValueError("time_cost_pruning requires every strategy to contain time_cost.")
    _per_family_topk(settings)
    _max_replenish_per_family(settings)


def _per_family_topk(settings):
    topk = int(settings.get("per_family_topk", DEFAULT_PER_FAMILY_TOPK))
    if topk < 1:
        raise ValueError("time_cost_pruning.per_family_topk must be >= 1.")
    return topk


def _max_replenish_per_family(settings):
    value = int(settings.get("max_replenish_per_family", DEFAULT_MAX_REPLENISH_PER_FAMILY))
    if value < 0:
        raise ValueError("time_cost_pruning.max_replenish_per_family must be >= 0.")
    return value


def _group_by_family(strategies, family_dims):
    groups = defaultdict(list)
    for strategy in strategies:
        groups[_family_key(strategy, family_dims)].append(strategy)
    return groups


def _family_key(strategy, family_dims):
    return tuple((dim, strategy.get(dim)) for dim in family_dims)


def _rank_family_source(family, config):
    feasible = _memory_feasible_strategies(family, config)
    source = feasible if feasible else family
    source_name = SOURCE_MEMORY_FEASIBLE if feasible else SOURCE_ALL
    return sorted(source, key=_time_cost_sort_key), source_name


def _memory_feasible_strategies(family, config):
    gpu_memory = _gpu_memory(config)
    if gpu_memory is None:
        return []
    return [
        strategy
        for strategy in family
        if strategy.get("memory_model") is not None and strategy["memory_model"] <= gpu_memory
    ]


def _gpu_memory(config):
    memory_model = config.experiment.auto_tuner.get("memory_model", {})
    return memory_model.get("gpu_memory")


def _time_cost_sort_key(strategy):
    return (
        strategy["time_cost"],
        -strategy.get("chip_score", float("-inf")),
        -strategy.get("memory_model", float("-inf")),
    )


def _rank_by_time_cost(family):
    ranked = sorted(family, key=_time_cost_sort_key)
    return {id(strategy): rank for rank, strategy in enumerate(ranked, start=1)}


def _annotate_family(family, family_key, ranked_source, source_name, topk):
    ranks = _rank_by_time_cost(ranked_source)
    for strategy in family:
        strategy["time_cost_family_key"] = dict(family_key)
        strategy["time_cost_family_rank"] = ranks.get(id(strategy))
        strategy["time_cost_family_topk"] = topk
        strategy["time_cost_family_source"] = source_name


def _strategy_family_metadata(strategy):
    family_key = strategy.get("time_cost_family_key")
    rank = strategy.get("time_cost_family_rank")
    topk = strategy.get("time_cost_family_topk")
    if family_key is None or rank is None or topk is None:
        raise ValueError("time_cost_pruning family_replenish_on_oom requires family metadata.")
    return family_key, int(rank), int(topk)


def _topk_family_history(history, family_key, topk):
    entries = []
    for entry in history or []:
        if entry.get("time_cost_family_key") != family_key:
            continue
        rank = entry.get("time_cost_family_rank")
        if rank is not None and int(rank) <= topk:
            entries.append(entry)
    return entries


def _replenished_family_count(history, family_key):
    return sum(
        1
        for entry in history or []
        if entry.get("time_cost_family_key") == family_key
        and entry.get("time_cost_replenished")
    )


def _is_oom_no_perf_failure(entry):
    if entry.get("performance") is not None:
        return False
    if entry.get("max_mem") == "OOM":
        return True
    return _error_mentions_oom(entry.get("error"))


def _error_mentions_oom(error):
    if not error:
        return False
    return any(marker in str(error) for marker in OOM_ERROR_MARKERS)


def _mark_strategy(strategy):
    strategy["time_cost_pruned"] = True
    strategy["time_cost_prune_reason"] = PRUNE_REASON
