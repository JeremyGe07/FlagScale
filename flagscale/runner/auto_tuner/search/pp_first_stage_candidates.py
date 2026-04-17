from copy import deepcopy

from flagscale.runner.auto_tuner.chip_profile import get_attached_chip_profile
from flagscale.runner.auto_tuner.cost.memory_cost import estimate_memory_cost
from flagscale.runner.auto_tuner.cost.time_cost import estimate_time_cost
from flagscale.runner.auto_tuner.plan.lowering import lower_strategy_to_plan
from flagscale.runner.auto_tuner.plan.validator import validate_model_plan
from flagscale.runner.auto_tuner.search.algorithm import sort_by_chip_score
from flagscale.runner.auto_tuner.search.chip_strategy_score import build_chip_score


def build_stage_candidates(
    *,
    searcher,
    space,
    config,
    partition,
    stage_index,
    max_stage_candidates=None,
):
    max_stage_candidates = _resolve_max_stage_candidates(config, max_stage_candidates)
    if max_stage_candidates <= 0:
        raise ValueError("max_stage_candidates must be positive")

    stage_range = tuple(partition.stage_ranges[stage_index])
    stage_device_group = tuple(partition.device_groups[stage_index])
    _validate_stage_context(space, config, partition, stage_device_group)

    stage_space = deepcopy(space)
    stage_space["pipeline_model_parallel_size"] = [partition.pp_degree]
    candidates = _generate_stage_candidates(searcher, stage_space, config)
    valid_candidates = []
    last_error = None
    for candidate in candidates:
        if not _candidate_matches_stage_mesh(candidate, stage_device_group):
            continue
        if not _candidate_matches_global_batch(candidate, config):
            continue
        try:
            valid_candidates.append(
                _enrich_stage_candidate(
                    candidate,
                    config,
                    stage_index,
                    stage_range,
                    stage_device_group,
                    partition.pp_degree,
                )
            )
        except ValueError as exc:
            last_error = exc
    if not valid_candidates:
        if last_error is not None:
            raise last_error
        raise ValueError("No executable stage candidates matched the stage constraints.")
    return _sort_stage_candidates(valid_candidates, config)[:max_stage_candidates]


def _resolve_max_stage_candidates(config, max_stage_candidates):
    if max_stage_candidates is not None:
        return max_stage_candidates
    planner_cfg = config.experiment.auto_tuner.get("planner", {})
    if "max_stage_candidates_per_stage" not in planner_cfg:
        raise ValueError("planner.max_stage_candidates_per_stage is required")
    return planner_cfg["max_stage_candidates_per_stage"]


def _validate_stage_context(space, config, partition, stage_device_group):
    expected_stage_mesh = _expected_stage_mesh_size(config, partition)
    if len(stage_device_group) != expected_stage_mesh:
        raise ValueError("stage device_group mesh size mismatch")
    if not _has_valid_global_batch_candidate(space, config, partition.pp_degree):
        raise ValueError("stage-local candidates violate global batch divisibility")


def _expected_stage_mesh_size(config, partition):
    cards = int(config.experiment.auto_tuner.cards)
    if cards % int(partition.pp_degree) != 0:
        raise ValueError("cards must be divisible by pipeline_model_parallel_size")
    return cards // int(partition.pp_degree)


def _has_valid_global_batch_candidate(space, config, pp_degree):
    cards = int(config.experiment.auto_tuner.cards)
    gbs = int(config.train.model.global_batch_size)
    for data_parallel_size in space["data_parallel_size"]:
        if cards % data_parallel_size != 0 or gbs % data_parallel_size != 0:
            continue
        for micro_batch_size in space["micro_batch_size"]:
            if gbs % (data_parallel_size * micro_batch_size) == 0:
                if cards % (data_parallel_size * pp_degree) == 0:
                    return True
    return False


def _generate_stage_candidates(searcher, space, config):
    parallelism_part = searcher._product_parallel_dims(space, config)
    micro_batch_part = searcher._product_micro_batch_size_vpp_dims(
        parallelism_part, space, config
    )
    return searcher._product_recompute_dims(micro_batch_part, space, config)


def _candidate_matches_stage_mesh(candidate, stage_device_group):
    required_mesh_size = (
        int(candidate["data_parallel_size"])
        * int(candidate["tensor_model_parallel_size"])
        * int(candidate["context_parallel_size"])
    )
    return required_mesh_size == len(stage_device_group)


def _candidate_matches_global_batch(candidate, config):
    gbs = int(config.train.model.global_batch_size)
    required = int(candidate["data_parallel_size"]) * int(candidate["micro_batch_size"])
    return gbs % required == 0


def _enrich_stage_candidate(
    candidate,
    config,
    stage_index,
    stage_range,
    stage_device_group,
    partition_pp_degree,
):
    local_layer_count = stage_range[1] - stage_range[0] + 1
    local_stage_range = (0, local_layer_count - 1)
    stage_strategy = deepcopy(candidate)
    _ensure_pipeline_model_parallel_size(stage_strategy, partition_pp_degree)
    stage_strategy.pop("stage_index", None)
    stage_strategy.pop("stage_range", None)
    stage_strategy.pop("stage_device_group", None)
    stage_strategy.pop("stage_memory_model", None)
    stage_strategy.pop("stage_time_cost", None)
    stage_strategy.pop("stage_partition_ranges", None)
    stage_strategy.pop("stage_device_groups", None)
    stage_strategy.pop("stage_strategies", None)
    stage_strategy["num_layers"] = local_layer_count

    stage_candidate = deepcopy(stage_strategy)
    stage_candidate["stage_index"] = stage_index
    stage_candidate["stage_range"] = stage_range
    stage_candidate["stage_device_group"] = stage_device_group
    stage_candidate["num_layers"] = local_layer_count
    stage_candidate["stage_partition_ranges"] = (local_stage_range,)
    stage_candidate["stage_device_groups"] = (stage_device_group,)
    stage_candidate["stage_strategies"] = (stage_strategy,)

    stage_config = _stage_config(config, len(stage_device_group))
    plan = lower_strategy_to_plan(stage_candidate, stage_config)
    validation = validate_model_plan(plan)
    if validation.runtime_mode != "stage-executable":
        raise ValueError("stage-local candidate is not executable")

    memory_cost = estimate_memory_cost(plan, stage_config)
    time_cost = estimate_time_cost(plan, stage_config)
    _validate_memory_limit(stage_candidate, memory_cost, config)

    stage_candidate["runtime_mode"] = validation.runtime_mode
    stage_candidate["runtime_executable"] = True
    stage_candidate["stage_memory_model"] = memory_cost["memory_total_mb"]
    stage_candidate["stage_time_cost"] = time_cost["time_total_ms"]
    stage_candidate["memory_model"] = stage_candidate["stage_memory_model"]
    stage_candidate["time_cost"] = stage_candidate["stage_time_cost"]
    _attach_chip_score(stage_candidate, config)
    return stage_candidate


def _ensure_pipeline_model_parallel_size(strategy, partition_pp_degree):
    if strategy.get("pipeline_model_parallel_size") != partition_pp_degree:
        raise ValueError("stage candidate pipeline_model_parallel_size must equal partition pp_degree")


def _stage_config(config, cards):
    stage_config = deepcopy(config)
    stage_config.experiment.auto_tuner.cards = cards
    return stage_config


def _validate_memory_limit(candidate, memory_cost, config):
    memory_limit_mb = _resolve_memory_limit_mb(config)
    if memory_limit_mb is None:
        return
    if memory_cost["memory_total_mb"] > memory_limit_mb:
        raise ValueError(
            "stage candidate memory exceeds available device memory "
            f"({memory_cost['memory_total_mb']} > {memory_limit_mb})"
        )


def _resolve_memory_limit_mb(config):
    memory_model = config.experiment.auto_tuner.get("memory_model", {})
    if "gpu_memory" in memory_model:
        return float(memory_model["gpu_memory"])
    profile = get_attached_chip_profile(config)
    if profile is None:
        return None
    return float(profile["memory"]["total_memory_mb"])


def _sort_stage_candidates(candidates, config):
    if config.experiment.auto_tuner.algo.get("use_profiled_time_cost", False):
        return sorted(candidates, key=lambda candidate: candidate["stage_time_cost"])
    if config.experiment.auto_tuner.algo.get("chip_aware_scoring", False):
        return sorted(candidates, key=sort_by_chip_score, reverse=True)
    if "memory_model" in config.experiment.auto_tuner:
        return sorted(
            candidates,
            key=lambda candidate: candidate["stage_memory_model"],
            reverse=True,
        )
    return candidates


def _attach_chip_score(strategy, config):
    if not config.experiment.auto_tuner.algo.get("chip_aware_scoring", False):
        return
    profile = get_attached_chip_profile(config)
    if profile is None:
        raise ValueError("chip_aware_scoring requires a chip profile")
    chip_score = build_chip_score(strategy, profile)
    strategy["chip_score"] = chip_score["score"]
    strategy["chip_priority"] = chip_score["priority"]
    strategy["chip_score_reasons"] = chip_score["reasons"]


__all__ = ["build_stage_candidates"]
