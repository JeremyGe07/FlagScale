import copy

from flagscale.runner.auto_tuner.search.algorithm import (
    GridAlgo,
    sort_by_chip_score,
    sort_by_time_cost,
)
from flagscale.runner.auto_tuner.search.searcher import Searcher
from flagscale.runner.auto_tuner.search.pp_first_partition import (
    build_layer_count_balanced_partition,
    is_power_of_two,
)
from flagscale.runner.auto_tuner.utils import sort_by_memory_model

DEFAULT_TOPK_PLANS_FOR_SHORT_RUN = 16
DEFAULT_MAX_PP_CANDIDATES = 4
DEFAULT_MAX_PARTITIONS_PER_PP = 1
DEFAULT_MAX_ASSIGNMENTS_PER_PARTITION = 256


class PPFirstSearcher(Searcher):
    def __init__(self, config):
        super().__init__(config)
        self._annotate_pp_first_metadata()

    def build_space(self, config):
        space = super().build_space(config)
        planner_cfg = _planner_cfg(config)
        pp_values = [value for value in space["pipeline_model_parallel_size"] if is_power_of_two(value)]
        max_pp = planner_cfg.get("max_pp_candidates", DEFAULT_MAX_PP_CANDIDATES)
        space["pipeline_model_parallel_size"] = pp_values[:max_pp]
        return space

    def build_strategies(self, space, config):
        strategies = []
        planner_cfg = _planner_cfg(config)
        world_size = config.experiment.auto_tuner.cards
        num_layers = config.train.model.num_layers
        max_assignments = planner_cfg.get(
            "max_assignments_per_partition", DEFAULT_MAX_ASSIGNMENTS_PER_PARTITION
        )
        for pp_index, pp_degree in enumerate(space["pipeline_model_parallel_size"]):
            partition = build_layer_count_balanced_partition(
                num_layers=num_layers,
                pp_degree=pp_degree,
                world_size=world_size,
            )
            partition_space = copy.deepcopy(space)
            partition_space["pipeline_model_parallel_size"] = [pp_degree]
            partition_strategies = super().build_strategies(partition_space, config)
            for assignment_index, strategy in enumerate(partition_strategies[:max_assignments]):
                strategy["partition_policy"] = partition.partition_policy
                strategy["topology_signature"] = partition.topology_signature
                strategy["provenance"] = partition.provenance
                strategy["legality_flags"] = list(partition.legality_flags)
                strategy["pp_candidate_rank"] = pp_index
                strategy["partition_candidate_rank"] = 0
                strategy["stage_partition_ranges"] = [list(rng) for rng in partition.stage_ranges]
                strategy["stage_device_groups"] = [list(group) for group in partition.device_groups]
                strategies.append(strategy)
        return strategies

    def _annotate_pp_first_metadata(self):
        topk = _planner_cfg(self.config).get("topk_plans_for_short_run", DEFAULT_TOPK_PLANS_FOR_SHORT_RUN)
        short_run_ids = {id(strategy) for strategy in getattr(self, "short_run_strategies", [])}
        estimate_metric = _estimate_metric_name(self.config)
        ranked = _sort_estimate_candidates(self.strategies, self.config)
        estimate_rank_by_id = {id(strategy): rank for rank, strategy in enumerate(ranked)}
        shortlist_count = len(getattr(self, "short_run_strategies", []))
        for strategy in self.strategies:
            strategy["planner_name"] = "pp_first"
            strategy["topk_plans_for_short_run"] = topk
            strategy["planner_budget"] = {
                "max_pp_candidates": _planner_cfg(self.config).get(
                    "max_pp_candidates", DEFAULT_MAX_PP_CANDIDATES
                ),
                "max_partitions_per_pp": _planner_cfg(self.config).get(
                    "max_partitions_per_pp", DEFAULT_MAX_PARTITIONS_PER_PP
                ),
                "max_assignments_per_partition": _planner_cfg(self.config).get(
                    "max_assignments_per_partition", DEFAULT_MAX_ASSIGNMENTS_PER_PARTITION
                ),
                "topk_plans_for_short_run": topk,
            }
            strategy["estimate_metric"] = estimate_metric
            strategy["estimate_rank"] = estimate_rank_by_id[id(strategy)]
            strategy["estimate_value"] = _estimate_value(strategy, estimate_metric)
            strategy["estimated_stage_costs"] = _build_estimated_stage_costs(strategy)
            strategy["short_run_candidate"] = id(strategy) in short_run_ids
            strategy["short_run_shortlist_count"] = shortlist_count

    def build_algo(self, strategies, config):
        shortlist = _build_short_run_shortlist(strategies, config)
        self.short_run_strategies = shortlist
        self.logger.info(
            "PPFirstSearcher: total_candidates=%s executable_candidates=%s shortlist=%s metric=%s",
            len(strategies),
            len([strategy for strategy in strategies if strategy.get("runtime_executable", False)]),
            len(shortlist),
            _estimate_metric_name(config),
        )
        return GridAlgo(shortlist, config)


def _planner_cfg(config):
    return config.experiment.auto_tuner.get("planner", {})


def _build_short_run_shortlist(strategies, config):
    ranked = _sort_estimate_candidates(strategies, config)
    topk = _planner_cfg(config).get("topk_plans_for_short_run", DEFAULT_TOPK_PLANS_FOR_SHORT_RUN)
    executable = [strategy for strategy in ranked if strategy.get("runtime_executable", False)]
    return executable[:topk]


def _sort_estimate_candidates(strategies, config):
    ranked = list(strategies)
    algo_cfg = config.experiment.auto_tuner.algo
    if algo_cfg.get("use_profiled_time_cost", False):
        return sorted(ranked, key=sort_by_time_cost)
    if algo_cfg.get("chip_aware_scoring", False):
        return sorted(ranked, key=sort_by_chip_score, reverse=True)
    if "memory_model" in config.experiment.auto_tuner:
        return sorted(ranked, key=sort_by_memory_model, reverse=True)
    return ranked


def _estimate_metric_name(config):
    algo_cfg = config.experiment.auto_tuner.algo
    if algo_cfg.get("use_profiled_time_cost", False):
        return "time_cost"
    if algo_cfg.get("chip_aware_scoring", False):
        return "chip_score"
    if "memory_model" in config.experiment.auto_tuner:
        return "memory_model"
    return "search_order"


def _estimate_value(strategy, metric_name):
    if metric_name == "search_order":
        return None
    return strategy.get(metric_name)


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


__all__ = ["PPFirstSearcher"]
