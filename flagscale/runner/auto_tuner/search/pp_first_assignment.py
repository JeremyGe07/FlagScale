import copy

from flagscale.runner.auto_tuner.search.pp_first_selection import (
    select_assignment_candidates,
)


def generate_assignment_candidates(
    *,
    searcher,
    space,
    config,
    pp_degree,
    max_assignments,
):
    if max_assignments <= 0:
        return []
    partition_space = copy.deepcopy(space)
    partition_space["pipeline_model_parallel_size"] = [pp_degree]
    parallelism_part = searcher._product_parallel_dims(partition_space, config)
    micro_batch_part = searcher._product_micro_batch_size_vpp_dims(
        parallelism_part, partition_space, config
    )
    assignments = searcher._product_recompute_dims(micro_batch_part, partition_space, config)
    return select_assignment_candidates(assignments, max_assignments, config)


__all__ = ["generate_assignment_candidates"]
