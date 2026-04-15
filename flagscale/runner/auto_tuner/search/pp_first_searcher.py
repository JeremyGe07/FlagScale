import copy

from flagscale.runner.auto_tuner.search.searcher import Searcher
from flagscale.runner.auto_tuner.search.pp_first_partition import (
    build_layer_count_balanced_partition,
    is_power_of_two,
)

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
        for rank, strategy in enumerate(self.strategies):
            strategy["planner_name"] = "pp_first"
            strategy["topk_plans_for_short_run"] = topk
            strategy["short_run_candidate"] = rank < topk and bool(
                strategy.get("runtime_executable", False)
            )


def _planner_cfg(config):
    return config.experiment.auto_tuner.get("planner", {})


__all__ = ["PPFirstSearcher"]
