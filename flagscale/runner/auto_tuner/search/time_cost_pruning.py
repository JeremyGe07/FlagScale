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
PRUNE_REASON = "time_cost.family_topk"


def is_time_cost_pruning_enabled(config):
    return bool(_settings(config).get("enabled", False))


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
        keep_ids = {id(strategy) for strategy in _select_family_keep_set(family, config, topk)}
        ranks = _rank_by_time_cost(family)
        for strategy in family:
            if id(strategy) in keep_ids:
                continue
            _mark_strategy(strategy, family_key, ranks[id(strategy)], topk)
            pruned_count += 1
    return pruned_count


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


def _per_family_topk(settings):
    topk = int(settings.get("per_family_topk", DEFAULT_PER_FAMILY_TOPK))
    if topk < 1:
        raise ValueError("time_cost_pruning.per_family_topk must be >= 1.")
    return topk


def _group_by_family(strategies, family_dims):
    groups = defaultdict(list)
    for strategy in strategies:
        groups[_family_key(strategy, family_dims)].append(strategy)
    return groups


def _family_key(strategy, family_dims):
    return tuple((dim, strategy.get(dim)) for dim in family_dims)


def _select_family_keep_set(family, config, topk):
    feasible = _memory_feasible_strategies(family, config)
    source = feasible if feasible else family
    return sorted(source, key=_time_cost_sort_key)[:topk]


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


def _mark_strategy(strategy, family_key, rank, topk):
    strategy["time_cost_pruned"] = True
    strategy["time_cost_prune_reason"] = PRUNE_REASON
    strategy["time_cost_family_key"] = dict(family_key)
    strategy["time_cost_family_rank"] = rank
    strategy["time_cost_family_topk"] = topk
