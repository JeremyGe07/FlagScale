from copy import deepcopy
from itertools import product

from flagscale.runner.auto_tuner.plan.lowering import lower_strategy_to_plan
from flagscale.runner.auto_tuner.plan.validator import validate_model_plan
from flagscale.runner.auto_tuner.search.pp_first_segment_errors import NoSegmentStageCandidatesError
from flagscale.runner.auto_tuner.search.pp_first_stage_candidates import build_stage_candidates
from flagscale.runner.auto_tuner.search.pp_first_transition_cost import estimate_transition_cost

FIXED_SEGMENT_FIELDS = (
    "micro_batch_size",
    "num_layers_per_virtual_pipeline_stage",
    "context_parallel_size",
    "expert_model_parallel_size",
    "use_distributed_optimizer",
    "device_type",
)
RECOMPUTE_FIELDS = (
    "use_recompute",
    "recompute_method",
    "recompute_granularity",
    "recompute_num_layers",
)
STAGE_LOCAL_METADATA = (
    "stage_index",
    "stage_range",
    "stage_device_group",
    "stage_memory_model",
    "stage_time_cost",
    "stage_partition_ranges",
    "stage_device_groups",
    "stage_strategies",
    "memory_model",
    "time_cost",
    "dp_aggregate_cost",
)


def build_segment_stage_candidates(
    *,
    searcher,
    space,
    config,
    partition,
    stage_index,
    max_segment_splits,
    max_segment_candidates,
    max_stage_candidates=None,
):
    _validate_positive_budget(max_segment_splits, "max_segment_splits")
    _validate_positive_budget(max_segment_candidates, "max_segment_candidates")
    base_candidates = _base_stage_candidates(
        searcher=searcher,
        space=space,
        config=config,
        partition=partition,
        stage_index=stage_index,
        max_stage_candidates=max_stage_candidates,
    )
    segment_candidates = _build_segment_candidates(
        base_candidates,
        config,
        partition,
        stage_index,
        max_segment_splits,
    )
    if not segment_candidates:
        raise NoSegmentStageCandidatesError("No segment-executable stage candidates matched.")
    return sorted(segment_candidates, key=_segment_candidate_sort_key)[:max_segment_candidates]


def boundary_transition_strategy(candidate, edge):
    if "segment_strategies" not in candidate:
        return dict(candidate)
    segments = candidate["segment_strategies"]
    segment = segments[0] if edge == "first" else segments[-1]
    strategy = dict(segment)
    for key in ("micro_batch_size", "pipeline_model_parallel_size"):
        strategy.setdefault(key, candidate.get(key))
    return strategy


def _base_stage_candidates(**kwargs):
    return build_stage_candidates(**kwargs)


def _validate_positive_budget(value, name):
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")


def _build_segment_candidates(
    base_candidates,
    config,
    partition,
    stage_index,
    max_segment_splits,
):
    stage_range = tuple(partition.stage_ranges[stage_index])
    device_group = tuple(partition.device_groups[stage_index])
    layer_count = stage_range[1] - stage_range[0] + 1
    candidates = []
    for segment_ranges in _segment_range_candidates(layer_count, max_segment_splits):
        candidates.extend(
            _materialize_segment_candidates(
                base_candidates,
                config,
                stage_index,
                stage_range,
                device_group,
                segment_ranges,
            )
        )
    return candidates


def _segment_range_candidates(layer_count, max_segment_splits):
    if layer_count < 2:
        return ()
    split_points = sorted(
        range(1, layer_count),
        key=lambda split: (abs(split - (layer_count - split)), split),
    )
    return tuple(
        ((0, split - 1), (split, layer_count - 1))
        for split in split_points[:max_segment_splits]
    )


def _materialize_segment_candidates(
    base_candidates,
    config,
    stage_index,
    stage_range,
    device_group,
    segment_ranges,
):
    candidates = []
    for segment_candidates in product(base_candidates, repeat=len(segment_ranges)):
        if not _is_supported_segment_tuple(segment_candidates):
            continue
        candidate = _build_segment_candidate(
            segment_candidates,
            config,
            stage_index,
            stage_range,
            device_group,
            segment_ranges,
        )
        if _is_valid_segment_candidate(candidate, config, len(device_group)):
            candidates.append(candidate)
    return candidates


def _is_supported_segment_tuple(segment_candidates):
    if not _has_tp_or_dp_transition(segment_candidates):
        return False
    reference = segment_candidates[0]
    for candidate in segment_candidates[1:]:
        if not _same_fixed_signature(reference, candidate):
            return False
    return _tensor_parallel_switch_is_sequence_parallel(segment_candidates)


def _has_tp_or_dp_transition(segment_candidates):
    signatures = {_tp_dp_signature(candidate) for candidate in segment_candidates}
    return len(signatures) > 1


def _same_fixed_signature(left, right):
    return (
        _field_signature(left, FIXED_SEGMENT_FIELDS + RECOMPUTE_FIELDS)
        == _field_signature(right, FIXED_SEGMENT_FIELDS + RECOMPUTE_FIELDS)
    )


def _tensor_parallel_switch_is_sequence_parallel(segment_candidates):
    tp_values = {candidate["tensor_model_parallel_size"] for candidate in segment_candidates}
    if len(tp_values) <= 1:
        return True
    return all(
        candidate.get("sequence_parallel") is True
        for candidate in segment_candidates
        if candidate["tensor_model_parallel_size"] > 1
    )


def _build_segment_candidate(
    segment_candidates,
    config,
    stage_index,
    stage_range,
    device_group,
    segment_ranges,
):
    candidate = _strip_metadata(segment_candidates[0])
    segment_strategies = _segment_strategies(segment_candidates)
    if _has_tensor_parallel_transition(segment_strategies):
        candidate["sequence_parallel"] = True
    candidate.update(
        {
            "stage_index": stage_index,
            "stage_range": stage_range,
            "stage_device_group": device_group,
            "num_layers": stage_range[1] - stage_range[0] + 1,
            "segment_partition_ranges": [list(item) for item in segment_ranges],
            "segment_strategies": segment_strategies,
            "segment_stage_candidate": True,
            "runtime_mode": "segment-executable",
            "runtime_executable": True,
        }
    )
    _attach_segment_costs(candidate, segment_candidates, segment_ranges, config)
    return candidate


def _segment_strategies(segment_candidates):
    strategies = [_strip_metadata(item) for item in segment_candidates]
    if _has_tensor_parallel_transition(strategies):
        for strategy in strategies:
            strategy["sequence_parallel"] = True
    return strategies


def _has_tensor_parallel_transition(strategies):
    return len({strategy["tensor_model_parallel_size"] for strategy in strategies}) > 1


def _strip_metadata(candidate):
    strategy = deepcopy(candidate)
    for key in STAGE_LOCAL_METADATA:
        strategy.pop(key, None)
    return strategy


def _attach_segment_costs(candidate, segment_candidates, segment_ranges, config):
    stage_layers = candidate["num_layers"]
    stage_costs = _scaled_stage_costs(segment_candidates, segment_ranges, stage_layers)
    transition_costs = _internal_transition_costs(segment_candidates, config)
    stage_time = sum(stage_costs) + sum(item["transition_total_ms"] for item in transition_costs)
    stage_memory = max(float(item["stage_memory_model"]) for item in segment_candidates)
    candidate["stage_time_cost"] = stage_time
    candidate["stage_memory_model"] = stage_memory
    candidate["time_cost"] = stage_time
    candidate["memory_model"] = stage_memory
    candidate["segment_stage_cost_breakdown"] = tuple(stage_costs)
    candidate["segment_transition_breakdown"] = tuple(transition_costs)


def _scaled_stage_costs(segment_candidates, segment_ranges, stage_layers):
    costs = []
    for candidate, segment_range in zip(segment_candidates, segment_ranges, strict=True):
        segment_layers = segment_range[1] - segment_range[0] + 1
        costs.append(float(candidate["stage_time_cost"]) * segment_layers / stage_layers)
    return tuple(costs)


def _internal_transition_costs(segment_candidates, config):
    transitions = []
    for previous, current in zip(segment_candidates, segment_candidates[1:], strict=False):
        transitions.append(
            estimate_transition_cost(
                boundary_transition_strategy(previous, "last"),
                boundary_transition_strategy(current, "first"),
                config,
            )
        )
    return tuple(transitions)


def _is_valid_segment_candidate(candidate, config, device_count):
    validation_strategy = deepcopy(candidate)
    validation_strategy["stage_partition_ranges"] = [[0, candidate["num_layers"] - 1]]
    validation_strategy["stage_device_groups"] = [list(range(device_count))]
    validation_strategy["stage_strategies"] = [candidate]
    try:
        plan = lower_strategy_to_plan(validation_strategy, config)
        return validate_model_plan(plan).runtime_mode == "segment-executable"
    except ValueError:
        return False


def _segment_candidate_sort_key(candidate):
    return (
        candidate["stage_time_cost"],
        _tp_dp_signature(candidate["segment_strategies"][0]),
        _tp_dp_signature(candidate["segment_strategies"][-1]),
        tuple(tuple(item) for item in candidate["segment_partition_ranges"]),
    )


def _field_signature(candidate, fields):
    return tuple(candidate.get(field) for field in fields)


def _tp_dp_signature(candidate):
    return (candidate.get("tensor_model_parallel_size"), candidate.get("data_parallel_size"))


__all__ = ["boundary_transition_strategy", "build_segment_stage_candidates"]
