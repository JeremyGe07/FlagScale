from flagscale.runner.auto_tuner.search.algorithm import GridAlgo
from flagscale.runner.auto_tuner.search.pp_first_assignment import (
    generate_assignment_candidates,
)
from flagscale.runner.auto_tuner.search.pp_first_segment_errors import (
    NoSegmentAssignmentCandidatesError,
)
from flagscale.runner.auto_tuner.search.searcher import (
    Searcher,
    get_first_last_num_layers_for_pp,
)
from flagscale.runner.auto_tuner.search.pp_first_partition import (
    POLICY_LAYER_COUNT_BALANCED,
    generate_partition_candidates,
    is_power_of_two,
)
from flagscale.runner.auto_tuner.search.pp_first_selection import (
    DEFAULT_TOPK_PLANS_FOR_SHORT_RUN,
    build_short_run_shortlist,
    estimate_metric_name,
    estimate_value,
    sort_estimate_candidates,
)

DEFAULT_MAX_PP_CANDIDATES = 4
DEFAULT_MAX_PARTITIONS_PER_PP = 1
DEFAULT_MAX_ASSIGNMENTS_PER_PARTITION = 256
DEFAULT_ASSIGNMENT_SOLVER = "enumerate"
RUNTIME_INEXPRESSIBLE_PARTITION = "runtime-inexpressible-partition"


class PPFirstSearcher(Searcher):
    def __init__(self, config):
        super().__init__(config)
        self._annotate_pp_first_metadata()

    def build_space(self, config):
        space = super().build_space(config)
        planner_cfg = _planner_cfg(config)
        pp_values = [
            value for value in space["pipeline_model_parallel_size"] if is_power_of_two(value)
        ]
        max_pp = planner_cfg.get("max_pp_candidates", DEFAULT_MAX_PP_CANDIDATES)
        space["pipeline_model_parallel_size"] = pp_values[:max_pp]
        return space

    def build_strategies(self, space, config):
        strategies = []
        planner_cfg = _planner_cfg(config)
        world_size = config.experiment.auto_tuner.cards
        num_layers = config.train.model.num_layers
        model_meta = _resolve_partition_model_meta(config)
        max_partitions = planner_cfg.get(
            "max_partitions_per_pp", DEFAULT_MAX_PARTITIONS_PER_PP
        )
        max_assignments = planner_cfg.get(
            "max_assignments_per_partition", DEFAULT_MAX_ASSIGNMENTS_PER_PARTITION
        )
        for pp_index, pp_degree in enumerate(space["pipeline_model_parallel_size"]):
            partitions = generate_partition_candidates(
                num_layers=num_layers,
                pp_degree=pp_degree,
                world_size=world_size,
                partition_policy=planner_cfg.get(
                    "partition_policy", POLICY_LAYER_COUNT_BALANCED
                ),
                max_partitions=max_partitions,
                hidden_size=model_meta["hidden_size"],
                padded_vocab_size=model_meta["padded_vocab_size"],
                seq_length=model_meta["seq_length"],
            )
            for partition_index, partition in enumerate(partitions):
                runtime_layer_counts = _runtime_layer_counts_for_partition(partition)
                if runtime_layer_counts == RUNTIME_INEXPRESSIBLE_PARTITION:
                    continue
                try:
                    partition_strategies = generate_assignment_candidates(
                        searcher=self,
                        space=space,
                        config=config,
                        pp_degree=pp_degree,
                        max_assignments=max_assignments,
                        partition=partition,
                    )
                except NoSegmentAssignmentCandidatesError as exc:
                    self.logger.warning(
                        "PPFirstSearcher: skip segment_dp partition pp=%s index=%s reason=%s",
                        pp_degree,
                        partition_index,
                        exc,
                    )
                    continue
                for assignment_index, strategy in enumerate(
                    partition_strategies
                ):
                    _apply_partition_runtime_layers(strategy, runtime_layer_counts)
                    strategy["partition_policy"] = partition.partition_policy
                    strategy["topology_signature"] = partition.topology_signature
                    strategy["provenance"] = partition.provenance
                    strategy["legality_flags"] = list(partition.legality_flags)
                    strategy["pp_candidate_rank"] = pp_index
                    strategy["partition_candidate_rank"] = partition_index
                    strategy["partition_candidate_count"] = len(partitions)
                    strategy["assignment_candidate_rank"] = assignment_index
                    strategy["stage_partition_ranges"] = [
                        list(rng) for rng in partition.stage_ranges
                    ]
                    strategy["stage_device_groups"] = [
                        list(group) for group in partition.device_groups
                    ]
                    strategies.append(strategy)
        if not strategies:
            raise ValueError("PPFirstSearcher produced no candidate strategies")
        return strategies

    def _annotate_pp_first_metadata(self):
        topk = _planner_cfg(self.config).get(
            "topk_plans_for_short_run",
            DEFAULT_TOPK_PLANS_FOR_SHORT_RUN,
        )
        short_run_ids = {id(strategy) for strategy in getattr(self, "short_run_strategies", [])}
        estimate_metric = estimate_metric_name(self.config)
        ranked = sort_estimate_candidates(self.strategies, self.config)
        estimate_rank_by_id = {id(strategy): rank for rank, strategy in enumerate(ranked)}
        shortlist_count = len(getattr(self, "short_run_strategies", []))
        for strategy in self.strategies:
            strategy["planner_name"] = "pp_first"
            strategy["topk_plans_for_short_run"] = topk
            strategy["planner_budget"] = _planner_budget(self.config, topk)
            strategy["estimate_metric"] = estimate_metric
            strategy["estimate_rank"] = estimate_rank_by_id[id(strategy)]
            strategy["estimate_value"] = estimate_value(strategy, estimate_metric)
            strategy["estimated_stage_costs"] = _build_estimated_stage_costs(strategy)
            strategy["short_run_candidate"] = id(strategy) in short_run_ids
            strategy["short_run_shortlist_count"] = shortlist_count
            strategy["legality_flags"] = _build_legality_flags(strategy)

    def build_algo(self, strategies, config):
        shortlist = build_short_run_shortlist(strategies, config)
        self.short_run_strategies = shortlist
        self.logger.info(
            "PPFirstSearcher: total_candidates=%s executable_candidates=%s shortlist=%s metric=%s",
            len(strategies),
            len([strategy for strategy in strategies if strategy.get("runtime_executable", False)]),
            len(shortlist),
            estimate_metric_name(config),
        )
        return GridAlgo(shortlist, config)


def _planner_cfg(config):
    return config.experiment.auto_tuner.get("planner", {})


def _planner_budget(config, topk):
    planner_cfg = _planner_cfg(config)
    return {
        "assignment_solver": planner_cfg.get(
            "assignment_solver", DEFAULT_ASSIGNMENT_SOLVER
        ),
        "max_pp_candidates": planner_cfg.get(
            "max_pp_candidates", DEFAULT_MAX_PP_CANDIDATES
        ),
        "max_partitions_per_pp": planner_cfg.get(
            "max_partitions_per_pp", DEFAULT_MAX_PARTITIONS_PER_PP
        ),
        "max_assignments_per_partition": planner_cfg.get(
            "max_assignments_per_partition", DEFAULT_MAX_ASSIGNMENTS_PER_PARTITION
        ),
        "max_stage_candidates_per_stage": planner_cfg.get(
            "max_stage_candidates_per_stage"
        ),
        "max_dp_results_per_partition": planner_cfg.get(
            "max_dp_results_per_partition"
        ),
        "max_segment_splits_per_stage": planner_cfg.get(
            "max_segment_splits_per_stage"
        ),
        "max_segment_candidates_per_stage": planner_cfg.get(
            "max_segment_candidates_per_stage"
        ),
        "topk_plans_for_short_run": topk,
    }


def _build_estimated_stage_costs(strategy):
    time_stages = (
        strategy.get("time_breakdown", {})
        .get("plan", {})
        .get("stages", [])
    )
    memory_stages = (
        strategy.get("memory_breakdown", {})
        .get("plan", {})
        .get("stages", [])
    )
    if not time_stages and not memory_stages:
        return []
    count = max(len(time_stages), len(memory_stages))
    stage_costs = []
    for index in range(count):
        time_stage = time_stages[index] if index < len(time_stages) else {}
        memory_stage = memory_stages[index] if index < len(memory_stages) else {}
        stage_costs.append(
            {
                "stage_id": time_stage.get("stage_id", memory_stage.get("stage_id", index)),
                "time_ms": time_stage.get("time_total_ms"),
                "memory_mb": memory_stage.get("memory_total_mb"),
            }
        )
    return stage_costs


def _build_legality_flags(strategy):
    flags = set(strategy.get("legality_flags", []))
    plan_kind = strategy.get("plan_kind")
    runtime_mode = strategy.get("runtime_mode")
    if plan_kind:
        flags.add(f"plan_kind:{plan_kind}")
    if runtime_mode:
        flags.add(f"runtime_mode:{runtime_mode}")
    if strategy.get("runtime_executable", False):
        flags.add("runtime_executable")
    else:
        flags.add("analysis_only")
    return sorted(flags)


def _runtime_layer_counts_for_partition(partition):
    stage_counts = _stage_counts_from_ranges(partition.stage_ranges)
    if _matches_default_runtime_layer_counts(stage_counts):
        return None
    if not _can_represent_stage_counts(stage_counts):
        return RUNTIME_INEXPRESSIBLE_PARTITION
    return stage_counts


def _apply_partition_runtime_layers(strategy, runtime_layer_counts):
    if runtime_layer_counts is None:
        return
    strategy["decoder_first_pipeline_num_layers"] = runtime_layer_counts[0]
    strategy["decoder_last_pipeline_num_layers"] = runtime_layer_counts[-1]


def _stage_counts_from_ranges(stage_ranges):
    return tuple(end - start + 1 for start, end in stage_ranges)


def _matches_default_runtime_layer_counts(stage_counts):
    pp_degree = len(stage_counts)
    total_layers = sum(stage_counts)
    if pp_degree == 1:
        return True
    if total_layers % pp_degree == 0:
        default_count = total_layers // pp_degree
        return stage_counts == tuple([default_count] * pp_degree)
    first_count, last_count = get_first_last_num_layers_for_pp(total_layers, pp_degree)
    if pp_degree == 2:
        return stage_counts == (first_count, last_count)
    middle_total = total_layers - first_count - last_count
    middle_stages = pp_degree - 2
    if middle_total <= 0 or middle_total % middle_stages != 0:
        return False
    middle_count = middle_total // middle_stages
    return stage_counts == (first_count,) + tuple([middle_count] * middle_stages) + (last_count,)


def _can_represent_stage_counts(stage_counts):
    if len(stage_counts) <= 2:
        return True
    return len(set(stage_counts[1:-1])) == 1


def _resolve_partition_model_meta(config):
    hidden_size = int(config.train.model.hidden_size)
    seq_length = int(config.train.model.seq_length)
    padded_vocab_size = config.train.model.get("padded_vocab_size")
    tokenizer_cfg = config.train.get("data", {}).get("tokenizer", {})
    if padded_vocab_size is None:
        padded_vocab_size = tokenizer_cfg.get("padded_vocab_size")
    if padded_vocab_size is None:
        padded_vocab_size = tokenizer_cfg.get("vocab_size")
    if padded_vocab_size is None:
        raise ValueError(
            "PP-first partition policies require padded_vocab_size or tokenizer vocab_size."
        )
    return {
        "hidden_size": hidden_size,
        "seq_length": seq_length,
        "padded_vocab_size": int(padded_vocab_size),
    }


__all__ = ["PPFirstSearcher"]
