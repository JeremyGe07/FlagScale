from flagscale.runner.auto_tuner.search.pp_first_partition import (
    POLICY_PROFILE_TIME_BALANCED,
)
from flagscale.runner.auto_tuner.search.pp_first_searcher import PPFirstSearcher
from tests.unit_tests.runner.test_auto_tuner_pp_first_planner import _config


def test_pp_first_searcher_applies_profile_partition_to_runtime_layers(tmp_path):
    config = _config(tmp_path, cards=2, pps=[2])
    config.train.model.num_layers = 28
    config.train.model.global_batch_size = 32
    config.train.model.hidden_size = 64
    config.train.model.seq_length = 2048
    config.train.model.padded_vocab_size = 32000
    config.experiment.auto_tuner.space.data_parallel_size = [1]
    config.experiment.auto_tuner.space.tensor_model_parallel_size = [1]
    config.experiment.auto_tuner.space.micro_batch_size = [2]
    config.experiment.auto_tuner.planner = {
        "name": "pp_first",
        "partition_policy": [POLICY_PROFILE_TIME_BALANCED],
        "max_pp_candidates": 1,
        "max_partitions_per_pp": 1,
        "max_assignments_per_partition": 1,
        "topk_plans_for_short_run": 1,
    }

    searcher = PPFirstSearcher(config)
    strategy = searcher.strategies[0]
    summary = strategy["plan_summary"]

    assert strategy["stage_partition_ranges"] == [[0, 15], [16, 27]]
    assert strategy["decoder_first_pipeline_num_layers"] == 16
    assert strategy["decoder_last_pipeline_num_layers"] == 12
    assert summary["stages"][0]["segments"][0]["end"] == 15
    assert summary["stages"][1]["segments"][0]["start"] == 16
