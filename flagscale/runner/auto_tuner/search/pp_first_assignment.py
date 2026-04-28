import copy
from types import SimpleNamespace

from flagscale.runner.auto_tuner.search.pp_first_dp_solver import solve_stage_level_dp
from flagscale.runner.auto_tuner.search.pp_first_segment_assignment import (
    SEGMENT_DP_ASSIGNMENT_SOLVER,
    generate_segment_dp_assignment_candidates,
)
from flagscale.runner.auto_tuner.search.pp_first_stage_candidates import (
    build_stage_candidates,
)
from flagscale.runner.auto_tuner.search.pp_first_selection import (
    select_assignment_candidates,
)
from flagscale.runner.auto_tuner.search.pp_first_transition_cost import (
    estimate_transition_cost,
)

DEFAULT_ASSIGNMENT_SOLVER = "enumerate"
DP_ASSIGNMENT_SOLVER = "dp"


def generate_assignment_candidates(
    *,
    searcher,
    space,
    config,
    pp_degree,
    max_assignments,
    partition=None,
):
    if max_assignments <= 0:
        return []
    planner_cfg = _planner_cfg(config)
    solver = _resolve_assignment_solver(planner_cfg)
    if solver == DP_ASSIGNMENT_SOLVER:
        return _generate_dp_assignment_candidates(
            searcher=searcher,
            space=space,
            config=config,
            partition=partition,
            max_assignments=max_assignments,
            planner_cfg=planner_cfg,
        )
    if solver == SEGMENT_DP_ASSIGNMENT_SOLVER:
        return generate_segment_dp_assignment_candidates(
            searcher=searcher,
            space=space,
            config=config,
            partition=partition,
            max_assignments=max_assignments,
            planner_cfg=planner_cfg,
        )
    if solver != DEFAULT_ASSIGNMENT_SOLVER:
        raise ValueError(f"Unsupported planner.assignment_solver: {solver!r}")
    return _generate_enumerated_assignment_candidates(
        searcher=searcher,
        space=space,
        config=config,
        pp_degree=pp_degree,
        max_assignments=max_assignments,
    )


def _generate_enumerated_assignment_candidates(
    *,
    searcher,
    space,
    config,
    pp_degree,
    max_assignments,
):
    partition_space = copy.deepcopy(space)
    partition_space["pipeline_model_parallel_size"] = [pp_degree]
    parallelism_part = searcher._product_parallel_dims(partition_space, config)
    micro_batch_part = searcher._product_micro_batch_size_vpp_dims(
        parallelism_part, partition_space, config
    )
    assignments = searcher._product_recompute_dims(
        micro_batch_part, partition_space, config
    )
    return select_assignment_candidates(assignments, max_assignments, config)


def _generate_dp_assignment_candidates(
    *,
    searcher,
    space,
    config,
    partition,
    max_assignments,
    planner_cfg,
):
    if partition is None:
        raise ValueError("partition is required when planner.assignment_solver=dp")
    stage_candidates = _build_stage_candidates(
        searcher=searcher,
        space=space,
        config=config,
        partition=partition,
        max_stage_candidates=planner_cfg.get("max_stage_candidates_per_stage"),
    )
    chains = _solve_dp_chains(
        stage_candidates,
        config=config,
        max_assignments=max_assignments,
        max_dp_results=planner_cfg.get("max_dp_results_per_partition"),
    )
    return [_materialize_dp_assignment_candidate(partition, chain) for chain in chains]


def _build_stage_candidates(searcher, space, config, partition, max_stage_candidates):
    local_partition = _local_partition_view(partition)
    local_config = _local_config_view(config, partition)
    return [
        build_stage_candidates(
            searcher=searcher,
            space=space,
            config=local_config,
            partition=local_partition,
            stage_index=stage_index,
            max_stage_candidates=max_stage_candidates,
        )
        for stage_index in range(len(partition.stage_ranges))
    ]


def _resolve_dp_result_budget(max_assignments, max_dp_results):
    if max_dp_results is None:
        raise ValueError("planner.max_dp_results_per_partition is required")
    return min(max_assignments, int(max_dp_results))


def _solve_dp_chains(stage_candidates, config, max_assignments, max_dp_results):
    resolved_max_results = _resolve_dp_result_budget(max_assignments, max_dp_results)
    grouped_chains = []
    seen_signatures = set()
    for candidate in stage_candidates[0]:
        signature = _dp_signature(candidate)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        filtered_candidates = _filter_stage_candidates(stage_candidates, signature)
        if filtered_candidates is None:
            continue
        grouped_chains.extend(
            solve_stage_level_dp(
                filtered_candidates,
                lambda previous, current: estimate_transition_cost(
                    previous, current, config
                ),
                max_results=resolved_max_results,
            )
        )
    if not grouped_chains:
        raise ValueError("No DP-compatible assignment candidates found")
    return sorted(grouped_chains, key=_dp_chain_sort_key)[:resolved_max_results]


def _filter_stage_candidates(stage_candidates, signature):
    filtered = []
    for stage in stage_candidates:
        stage_candidates_for_signature = [
            candidate for candidate in stage if _dp_signature(candidate) == signature
        ]
        if not stage_candidates_for_signature:
            return None
        filtered.append(stage_candidates_for_signature)
    return filtered


def _dp_signature(candidate):
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


def _dp_chain_sort_key(chain):
    return (chain.dp_aggregate_cost, _chain_signature(chain))


def _chain_signature(chain):
    return tuple(_candidate_signature(candidate) for candidate in chain.stage_strategies)


def _candidate_signature(candidate):
    return tuple(sorted((key, _sort_value(value)) for key, value in dict(candidate).items()))


def _sort_value(value):
    if isinstance(value, list):
        return tuple(_sort_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _sort_value(item)) for key, item in value.items()))
    return (type(value).__name__, repr(value))


def _materialize_dp_assignment_candidate(partition, chain):
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
    strategy["runtime_mode"] = "stage-executable"
    strategy["dp_aggregate_cost"] = chain.dp_aggregate_cost
    return strategy


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


def _planner_cfg(config):
    return config.experiment.auto_tuner.get("planner", {})


def _resolve_assignment_solver(planner_cfg):
    return planner_cfg.get("assignment_solver", DEFAULT_ASSIGNMENT_SOLVER)


__all__ = ["generate_assignment_candidates"]
