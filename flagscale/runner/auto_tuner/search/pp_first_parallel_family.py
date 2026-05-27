from collections import Counter

DEFAULT_MIN_ASSIGNMENTS_PER_PARALLEL_FAMILY = 1
DEFAULT_MIN_SHORTLIST_PER_PARALLEL_FAMILY = 1
DEFAULT_MEMORY_UTILIZATION = (0.0, 1.0)
PARALLEL_FAMILY_KEYS = (
    "data_parallel_size",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
)


def parallel_family_key(strategy):
    if not all(key in strategy for key in PARALLEL_FAMILY_KEYS):
        return None
    return tuple(strategy[key] for key in PARALLEL_FAMILY_KEYS)


def resolve_shortlist_parallel_family_floor(planner_cfg, strategies):
    explicit = planner_cfg.get("min_shortlist_per_parallel_family")
    if explicit is not None:
        return explicit
    families = {
        family for family in (parallel_family_key(strategy) for strategy in strategies)
        if family is not None
    }
    if len(families) <= 1:
        return 0
    return DEFAULT_MIN_SHORTLIST_PER_PARALLEL_FAMILY


def order_for_parallel_family_recall(ranked, config):
    gpu_memory = _resolve_gpu_memory(config)
    if gpu_memory is None:
        return ranked
    indexed = list(enumerate(ranked))
    return [
        candidate for _, candidate in sorted(
            indexed,
            key=lambda item: (_memory_prune_rank(item[1], gpu_memory), item[0]),
        )
    ]


def select_with_parallel_family_floor(ranked, max_count, floor, identity_fn, recall_ranked=None):
    selected, seen = _select_top_ranked(ranked, max_count, identity_fn)
    if floor <= 0:
        return selected
    counts = _family_counts(selected)
    recall_candidates = ranked if recall_ranked is None else recall_ranked
    for candidate in recall_candidates:
        family = parallel_family_key(candidate)
        candidate_key = identity_fn(candidate)
        if family is None or counts[family] >= floor or candidate_key in seen:
            continue
        replace_index = _find_replaceable_index(selected, counts, floor)
        if replace_index is None:
            return selected
        removed = selected[replace_index]
        seen.remove(identity_fn(removed))
        removed_family = parallel_family_key(removed)
        if removed_family is not None:
            counts[removed_family] -= 1
        selected[replace_index] = candidate
        seen.add(candidate_key)
        counts[family] += 1
    return selected


def append_parallel_family_floor(shortlist, seen, ranked, floor, append_fn):
    if floor <= 0:
        return
    counts = _family_counts(shortlist)
    for candidate in ranked:
        family = parallel_family_key(candidate)
        if family is None or counts[family] >= floor:
            continue
        if append_fn(shortlist, seen, candidate):
            counts[family] += 1


def _select_top_ranked(ranked, max_count, identity_fn):
    selected = []
    seen = set()
    for candidate in ranked:
        if len(selected) >= max_count:
            break
        candidate_key = identity_fn(candidate)
        if candidate_key in seen:
            continue
        selected.append(candidate)
        seen.add(candidate_key)
    return selected, seen


def _family_counts(strategies):
    return Counter(
        family for family in (parallel_family_key(strategy) for strategy in strategies)
        if family is not None
    )


def _find_replaceable_index(selected, counts, floor):
    for index in range(len(selected) - 1, -1, -1):
        family = parallel_family_key(selected[index])
        if family is not None and counts[family] > floor:
            return index
    return None


def _resolve_gpu_memory(config):
    memory_model = config.experiment.auto_tuner.get("memory_model", {})
    if "gpu_memory" not in memory_model:
        return None
    return float(memory_model["gpu_memory"])


def _memory_prune_rank(strategy, gpu_memory):
    memory_model = strategy.get("memory_model")
    if memory_model is None:
        return 0
    lower_bound, upper_bound = _memory_bounds(strategy, gpu_memory)
    memory_model = float(memory_model)
    if lower_bound <= memory_model <= upper_bound:
        return 0
    return 1


def _memory_bounds(strategy, gpu_memory):
    utilization = strategy.get("gpu_utilization", DEFAULT_MEMORY_UTILIZATION)
    return gpu_memory * float(utilization[0]), gpu_memory * float(utilization[1])
