import logging

from flagscale.runner.auto_tuner.search.pp_first_searcher import PPFirstSearcher
from tests.unit_tests.runner.test_auto_tuner_pp_first_planner import _config


def test_segment_dp_skips_infeasible_partitions_without_hiding_valid_ones(
    tmp_path,
    caplog,
):
    config = _config(tmp_path, cards=2, pps=[1, 2])
    config.experiment.auto_tuner.planner = {
        "name": "pp_first",
        "assignment_solver": "segment_dp",
        "max_pp_candidates": 2,
        "max_partitions_per_pp": 1,
        "max_assignments_per_partition": 2,
        "max_stage_candidates_per_stage": 100,
        "max_segment_splits_per_stage": 1,
        "max_segment_candidates_per_stage": 4,
        "max_dp_results_per_partition": 2,
        "topk_plans_for_short_run": 2,
    }

    with caplog.at_level(logging.WARNING, logger="FlagScale-AutoTuner"):
        searcher = PPFirstSearcher(config)

    assert searcher.strategies
    assert {strategy["pipeline_model_parallel_size"] for strategy in searcher.strategies} == {1}
    assert any("skip segment_dp partition" in record.message for record in caplog.records)
