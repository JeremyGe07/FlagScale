from collections import Counter
from collections.abc import Mapping

from flagscale.runner.auto_tuner.chip_profile import get_attached_chip_profile
from flagscale.runner.auto_tuner.cost.memory_cost import estimate_memory_cost
from flagscale.runner.auto_tuner.cost.time_cost import estimate_time_cost
from flagscale.runner.auto_tuner.search.algorithm import (
    sort_by_chip_score,
    sort_by_time_cost,
)
from flagscale.runner.auto_tuner.search.chip_strategy_score import build_chip_score
from flagscale.runner.auto_tuner.utils import sort_by_memory_model

DEFAULT_TOPK_PLANS_FOR_SHORT_RUN = 16
DEFAULT_MIN_SHORTLIST_PER_PP_DEGREE = 1
DEFAULT_MIN_SHORTLIST_PER_PARTITION_POLICY = 1
RUNTIME_STRATEGY_KEYS = (
    "data_parallel_size",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "expert_model_parallel_size",
    "context_parallel_size",
    "decoder_first_pipeline_num_layers",
    "decoder_last_pipeline_num_layers",
    "use_distributed_optimizer",
    "sequence_parallel",
    "acc_step",
    "micro_batch_size",
    "num_layers_per_virtual_pipeline_stage",
    "use_recompute",
    "recompute_method",
    "recompute_granularity",
    "recompute_num_layers",
)
SEGMENT_RUNTIME_KEYS = (
    "segment_partition_ranges",
    "segment_strategies",
)


def sort_estimate_candidates(strategies, config):
    ranked = _inject_estimate_fields(strategies, config)
    algo_cfg = config.experiment.auto_tuner.algo
    if algo_cfg.get("use_profiled_time_cost", False):
        return sorted(ranked, key=sort_by_time_cost)
    if algo_cfg.get("chip_aware_scoring", False):
        return sorted(ranked, key=sort_by_chip_score, reverse=True)
    if "memory_model" in config.experiment.auto_tuner:
        return sorted(ranked, key=sort_by_memory_model, reverse=True)
    return ranked


def estimate_metric_name(config):
    algo_cfg = config.experiment.auto_tuner.algo
    if algo_cfg.get("use_profiled_time_cost", False):
        return "time_cost"
    if algo_cfg.get("chip_aware_scoring", False):
        return "chip_score"
    if "memory_model" in config.experiment.auto_tuner:
        return "memory_model"
    return "search_order"


def estimate_value(strategy, metric_name):
    if metric_name == "search_order":
        return None
    return strategy.get(metric_name)


def select_assignment_candidates(strategies, max_assignments, config):
    if max_assignments <= 0:
        return []
    if len(strategies) <= max_assignments:
        return list(strategies)
    return sort_estimate_candidates(strategies, config)[:max_assignments]


def build_short_run_shortlist(strategies, config):
    ranked = sort_estimate_candidates(strategies, config)
    executable = [strategy for strategy in ranked if strategy.get("runtime_executable", False)]
    planner_cfg = _planner_cfg(config)
    topk = planner_cfg.get("topk_plans_for_short_run", DEFAULT_TOPK_PLANS_FOR_SHORT_RUN)
    shortlist = []
    seen = set()
    _append_topk_candidates(shortlist, seen, executable, topk)
    _append_bucket_floor(
        shortlist,
        seen,
        executable,
        _resolve_pp_floor(planner_cfg, executable),
        lambda strategy: strategy["pipeline_model_parallel_size"],
    )
    _append_bucket_floor(
        shortlist,
        seen,
        executable,
        _resolve_policy_floor(planner_cfg, executable),
        lambda strategy: strategy.get("partition_policy"),
    )
    return shortlist


def _planner_cfg(config):
    return config.experiment.auto_tuner.get("planner", {})


def _append_candidates(shortlist, seen, candidates):
    for candidate in candidates:
        _append_candidate(shortlist, seen, candidate)


def _append_topk_candidates(shortlist, seen, candidates, topk):
    if topk <= 0:
        return
    for candidate in candidates:
        if len(shortlist) >= topk:
            return
        _append_candidate(shortlist, seen, candidate)


def _append_candidate(shortlist, seen, candidate):
    candidate_key = _runtime_execution_key(candidate)
    if candidate_key in seen:
        return False
    shortlist.append(candidate)
    seen.add(candidate_key)
    return True


def _append_bucket_floor(shortlist, seen, strategies, floor, key_fn):
    if floor <= 0:
        return
    counts = Counter()
    for strategy in shortlist:
        bucket = key_fn(strategy)
        if bucket is not None:
            counts[bucket] += 1
    for strategy in strategies:
        bucket = key_fn(strategy)
        if bucket is None or counts[bucket] >= floor:
            continue
        if _append_candidate(shortlist, seen, strategy):
            counts[bucket] += 1


def _runtime_execution_key(strategy):
    if not _has_runtime_identity(strategy):
        return ("object", id(strategy))
    key = tuple((field, _hashable_value(strategy.get(field))) for field in RUNTIME_STRATEGY_KEYS)
    if "stage_strategies" not in strategy:
        return key
    return key + (
        ("stage_strategies", _stage_runtime_signature(strategy["stage_strategies"])),
        ("stage_layout", _stage_layout_signature(strategy)),
    )


def _has_runtime_identity(strategy):
    return all(key in strategy for key in RUNTIME_STRATEGY_KEYS)


def _hashable_value(value):
    if isinstance(value, list):
        return tuple(_hashable_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_hashable_value(item) for item in value)
    if isinstance(value, Mapping):
        return tuple(sorted((key, _hashable_value(item)) for key, item in value.items()))
    return value


def _stage_runtime_signature(stage_strategies):
    return tuple(
        tuple(
            (field, _hashable_value(stage.get(field)))
            for field in RUNTIME_STRATEGY_KEYS
        ) + tuple(
            (field, _hashable_value(stage.get(field)))
            for field in SEGMENT_RUNTIME_KEYS
            if field in stage
        )
        for stage in stage_strategies
    )


def _stage_layout_signature(strategy):
    return tuple(
        (field, _hashable_value(strategy.get(field)))
        for field in (
            "stage_partition_ranges",
            "stage_device_groups",
            "stage_device_types",
        )
        if field in strategy
    )


def _resolve_pp_floor(planner_cfg, executable):
    explicit = planner_cfg.get("min_shortlist_per_pp_degree")
    if explicit is not None:
        return explicit
    pp_degrees = {strategy["pipeline_model_parallel_size"] for strategy in executable}
    return DEFAULT_MIN_SHORTLIST_PER_PP_DEGREE if len(pp_degrees) > 1 else 0


def _resolve_policy_floor(planner_cfg, executable):
    explicit = planner_cfg.get("min_shortlist_per_partition_policy")
    if explicit is not None:
        return explicit
    policies = {
        strategy.get("partition_policy")
        for strategy in executable
        if strategy.get("partition_policy") is not None
    }
    return DEFAULT_MIN_SHORTLIST_PER_PARTITION_POLICY if len(policies) > 1 else 0


def _inject_estimate_fields(strategies, config):
    ranked = list(strategies)
    algo_cfg = config.experiment.auto_tuner.algo
    if algo_cfg.get("chip_aware_scoring", False):
        _inject_chip_scores(ranked, config)
    if "memory_model" in config.experiment.auto_tuner:
        _inject_memory_costs(ranked, config)
    if algo_cfg.get("use_profiled_time_cost", False):
        _inject_time_costs(ranked, config)
    return ranked


def _inject_chip_scores(strategies, config):
    profile = get_attached_chip_profile(config)
    if profile is None:
        return
    for strategy in strategies:
        if "chip_score" in strategy:
            continue
        chip_score = build_chip_score(strategy, profile)
        strategy["chip_score"] = chip_score["score"]
        strategy["chip_priority"] = chip_score["priority"]
        strategy["chip_score_reasons"] = chip_score["reasons"]


def _inject_memory_costs(strategies, config):
    for strategy in strategies:
        if "memory_model" in strategy:
            continue
        memory_cost = estimate_memory_cost(strategy, config)
        strategy["memory_model"] = memory_cost["memory_total_mb"]
        strategy["memory_breakdown"] = memory_cost["memory_breakdown"]


def _inject_time_costs(strategies, config):
    for strategy in strategies:
        if "time_cost" in strategy:
            continue
        time_cost = estimate_time_cost(strategy, config)
        strategy["time_cost"] = time_cost["time_total_ms"]
        strategy["time_breakdown"] = time_cost["time_breakdown"]


__all__ = [
    "DEFAULT_TOPK_PLANS_FOR_SHORT_RUN",
    "build_short_run_shortlist",
    "estimate_metric_name",
    "estimate_value",
    "select_assignment_candidates",
    "sort_estimate_candidates",
]
