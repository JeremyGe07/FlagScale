import copy
from types import SimpleNamespace

from flagscale.runner.auto_tuner.search.pp_first_dp_solver import solve_stage_level_dp
from flagscale.runner.auto_tuner.search.pp_first_segment_candidates import (
    boundary_transition_strategy,
    build_segment_stage_candidates,
)
from flagscale.runner.auto_tuner.search.pp_first_segment_errors import (
    NoSegmentAssignmentCandidatesError,
    NoSegmentStageCandidatesError,
)
from flagscale.runner.auto_tuner.search.pp_first_transition_cost import (
    estimate_transition_cost,
)

SEGMENT_DP_ASSIGNMENT_SOLVER = "segment_dp"


def generate_segment_dp_assignment_candidates(
    *,
    searcher,
    space,
    config,
    partition,
    max_assignments,
    planner_cfg,
):
    if partition is None:
        raise ValueError("partition is required when planner.assignment_solver=segment_dp")
    stage_candidates = _build_segment_stage_candidates(
        searcher,
        space,
        config,
        partition,
        planner_cfg,
    )
    chains = _solve_segment_dp_chains(
        stage_candidates,
        config,
        max_assignments,
        planner_cfg.get("max_dp_results_per_partition"),
    )
    return [_materialize_segment_assignment(partition, chain) for chain in chains]


def _build_segment_stage_candidates(searcher, space, config, partition, planner_cfg):
    local_partition = _local_partition_view(partition)
    local_config = _local_config_view(config, partition)
    max_splits = _required_planner_budget(planner_cfg, "max_segment_splits_per_stage")
    max_candidates = _required_planner_budget(planner_cfg, "max_segment_candidates_per_stage")
    stage_candidates = []
    for stage_index in range(len(partition.stage_ranges)):
        try:
            candidates = build_segment_stage_candidates(
                searcher=searcher,
                space=space,
                config=local_config,
                partition=local_partition,
                stage_index=stage_index,
                max_stage_candidates=planner_cfg.get("max_stage_candidates_per_stage"),
                max_segment_splits=max_splits,
                max_segment_candidates=max_candidates,
            )
        except NoSegmentStageCandidatesError as exc:
            raise NoSegmentAssignmentCandidatesError(
                f"pp={partition.pp_degree} stage={stage_index}: {exc}"
            ) from exc
        stage_candidates.append(candidates)
    return stage_candidates


def _solve_segment_dp_chains(stage_candidates, config, max_assignments, max_dp_results):
    max_results = _resolve_dp_result_budget(max_assignments, max_dp_results)
    chains = []
    seen_signatures = set()
    for candidate in stage_candidates[0]:
        signature = _global_signature(candidate)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        filtered = _filter_stage_candidates(stage_candidates, signature)
        if filtered is None:
            continue
        chains.extend(_solve_signature_chains(filtered, config, max_results))
    if not chains:
        raise NoSegmentAssignmentCandidatesError(
            "No segment DP-compatible assignment candidates found"
        )
    return sorted(chains, key=lambda chain: chain.dp_aggregate_cost)[:max_results]


def _solve_signature_chains(stage_candidates, config, max_results):
    return solve_stage_level_dp(
        stage_candidates,
        lambda previous, current: _estimate_segment_boundary_transition(
            previous,
            current,
            config,
        ),
        max_results=max_results,
    )


def _filter_stage_candidates(stage_candidates, signature):
    filtered = []
    for candidates in stage_candidates:
        matching = [
            candidate for candidate in candidates if _global_signature(candidate) == signature
        ]
        if not matching:
            return None
        filtered.append(matching)
    return filtered


def _global_signature(candidate):
    return (
        candidate.get("micro_batch_size"),
        candidate.get("use_distributed_optimizer"),
        candidate.get("context_parallel_size"),
        _recompute_signature(candidate),
        candidate.get("num_layers_per_virtual_pipeline_stage"),
    )


def _recompute_signature(candidate):
    return (
        candidate.get("use_recompute"),
        candidate.get("recompute_method"),
        candidate.get("recompute_granularity"),
        candidate.get("recompute_num_layers"),
    )


def _materialize_segment_assignment(partition, chain):
    strategy = dict(chain.stage_strategies[0])
    _strip_stage_local_metadata(strategy)
    strategy["stage_strategies"] = tuple(dict(item) for item in chain.stage_strategies)
    strategy["stage_partition_ranges"] = [list(rng) for rng in partition.stage_ranges]
    strategy["stage_device_groups"] = [list(group) for group in partition.device_groups]
    decoder_first, decoder_last = _stage_layer_counts(partition.stage_ranges)
    strategy["decoder_first_pipeline_num_layers"] = decoder_first
    strategy["decoder_last_pipeline_num_layers"] = decoder_last
    strategy["num_layers"] = partition.stage_ranges[-1][1] + 1
    strategy["pipeline_model_parallel_size"] = partition.pp_degree
    strategy["partition_policy"] = partition.partition_policy
    strategy["topology_signature"] = partition.topology_signature
    strategy["provenance"] = partition.provenance
    strategy["legality_flags"] = list(partition.legality_flags)
    strategy["runtime_executable"] = True
    strategy["runtime_mode"] = "segment-executable"
    strategy["segment_dp_aggregate_cost"] = chain.dp_aggregate_cost
    return strategy


def _resolve_dp_result_budget(max_assignments, max_dp_results):
    if max_dp_results is None:
        raise ValueError("planner.max_dp_results_per_partition is required")
    return min(max_assignments, int(max_dp_results))


def _required_planner_budget(planner_cfg, key):
    value = planner_cfg.get(key)
    if value is None:
        raise ValueError(f"planner.{key} is required")
    if int(value) <= 0:
        raise ValueError(f"planner.{key} must be positive")
    return int(value)


def _estimate_segment_boundary_transition(previous, current, config):
    return estimate_transition_cost(
        boundary_transition_strategy(previous, "last"),
        boundary_transition_strategy(current, "first"),
        config,
    )


def _local_partition_view(partition):
    return SimpleNamespace(
        stage_ranges=partition.stage_ranges,
        device_groups=tuple(
            tuple(range(len(group))) for group in partition.device_groups
        ),
        pp_degree=1,
    )


def _local_config_view(config, partition):
    local_config = copy.deepcopy(config)
    local_config.experiment.auto_tuner.cards = len(partition.device_groups[0])
    return local_config


def _strip_stage_local_metadata(strategy):
    for key in (
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
    ):
        strategy.pop(key, None)


def _stage_layer_counts(stage_ranges):
    first_range = stage_ranges[0]
    last_range = stage_ranges[-1]
    return first_range[1] - first_range[0] + 1, last_range[1] - last_range[0] + 1


__all__ = ["SEGMENT_DP_ASSIGNMENT_SOLVER", "generate_segment_dp_assignment_candidates"]
