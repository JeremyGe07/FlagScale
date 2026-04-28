from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.search.pp_first_selection import (
    build_short_run_shortlist,
    select_assignment_candidates,
    _runtime_execution_key,
)


def _config():
    return OmegaConf.create(
        {
            "experiment": {
                "auto_tuner": {
                    "algo": {"name": "grid", "use_profiled_time_cost": True},
                    "planner": {"topk_plans_for_short_run": 2},
                }
            }
        }
    )


def test_select_assignment_candidates_uses_estimate_ranking():
    config = _config()
    strategies = [
        {"name": "slow", "time_cost": 9.0},
        {"name": "best", "time_cost": 1.0},
        {"name": "mid", "time_cost": 4.0},
    ]

    selected = select_assignment_candidates(strategies, max_assignments=2, config=config)

    assert [strategy["name"] for strategy in selected] == ["best", "mid"]


def test_build_short_run_shortlist_adds_pp_recall_floor():
    config = _config()
    strategies = [
        {
            "name": "pp2-best",
            "time_cost": 1.0,
            "pipeline_model_parallel_size": 2,
            "partition_policy": "layer_count_balanced",
            "runtime_executable": True,
        },
        {
            "name": "pp2-next",
            "time_cost": 2.0,
            "pipeline_model_parallel_size": 2,
            "partition_policy": "layer_count_balanced",
            "runtime_executable": True,
        },
        {
            "name": "pp1-recall",
            "time_cost": 3.0,
            "pipeline_model_parallel_size": 1,
            "partition_policy": "layer_count_balanced",
            "runtime_executable": True,
        },
    ]

    shortlist = build_short_run_shortlist(strategies, config)

    assert [strategy["name"] for strategy in shortlist] == [
        "pp2-best",
        "pp2-next",
        "pp1-recall",
    ]


def test_build_short_run_shortlist_adds_policy_recall_floor():
    config = _config()
    strategies = [
        {
            "name": "layer-best",
            "time_cost": 1.0,
            "pipeline_model_parallel_size": 2,
            "partition_policy": "layer_count_balanced",
            "runtime_executable": True,
        },
        {
            "name": "layer-next",
            "time_cost": 2.0,
            "pipeline_model_parallel_size": 1,
            "partition_policy": "layer_count_balanced",
            "runtime_executable": True,
        },
        {
            "name": "param-recall",
            "time_cost": 3.0,
            "pipeline_model_parallel_size": 1,
            "partition_policy": "param_balanced",
            "runtime_executable": True,
        },
    ]

    shortlist = build_short_run_shortlist(strategies, config)

    assert [strategy["name"] for strategy in shortlist] == [
        "layer-best",
        "layer-next",
        "param-recall",
    ]


def test_build_short_run_shortlist_dedupes_equivalent_runtime_plans():
    config = _config()
    config.experiment.auto_tuner.planner.topk_plans_for_short_run = 4
    strategies = [
        {
            "name": "layer-pp1",
            "time_cost": 1.0,
            "data_parallel_size": 2,
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "use_distributed_optimizer": True,
            "sequence_parallel": False,
            "acc_step": 16,
            "micro_batch_size": 1,
            "num_layers_per_virtual_pipeline_stage": None,
            "use_recompute": False,
            "recompute_method": None,
            "recompute_granularity": None,
            "recompute_num_layers": None,
            "decoder_first_pipeline_num_layers": None,
            "decoder_last_pipeline_num_layers": None,
            "partition_policy": "layer_count_balanced",
            "runtime_executable": True,
        },
        {
            "name": "profile-pp1-duplicate",
            "time_cost": 2.0,
            "data_parallel_size": 2,
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "use_distributed_optimizer": True,
            "sequence_parallel": False,
            "acc_step": 16,
            "micro_batch_size": 1,
            "num_layers_per_virtual_pipeline_stage": None,
            "use_recompute": False,
            "recompute_method": None,
            "recompute_granularity": None,
            "recompute_num_layers": None,
            "decoder_first_pipeline_num_layers": None,
            "decoder_last_pipeline_num_layers": None,
            "partition_policy": "profile_time_balanced",
            "runtime_executable": True,
        },
        {
            "name": "profile-pp2-distinct",
            "time_cost": 3.0,
            "data_parallel_size": 1,
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 2,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "use_distributed_optimizer": False,
            "sequence_parallel": False,
            "acc_step": 16,
            "micro_batch_size": 2,
            "num_layers_per_virtual_pipeline_stage": None,
            "use_recompute": False,
            "recompute_method": None,
            "recompute_granularity": None,
            "recompute_num_layers": None,
            "decoder_first_pipeline_num_layers": 16,
            "decoder_last_pipeline_num_layers": 12,
            "partition_policy": "profile_time_balanced",
            "runtime_executable": True,
        },
    ]

    shortlist = build_short_run_shortlist(strategies, config)

    assert [strategy["name"] for strategy in shortlist] == [
        "layer-pp1",
        "profile-pp2-distinct",
    ]


def test_build_short_run_shortlist_keeps_dp_stage_heterogeneous_plans_distinct():
    config = _config()
    config.experiment.auto_tuner.planner.topk_plans_for_short_run = 4
    base_stage = {
        "data_parallel_size": 2,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 2,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "use_distributed_optimizer": True,
        "sequence_parallel": False,
        "acc_step": 16,
        "micro_batch_size": 1,
        "num_layers_per_virtual_pipeline_stage": None,
        "use_recompute": False,
        "recompute_method": None,
        "recompute_granularity": None,
        "recompute_num_layers": None,
        "decoder_first_pipeline_num_layers": 16,
        "decoder_last_pipeline_num_layers": 12,
    }
    strategies = [
        {
            "name": "dp-chain-a",
            "time_cost": 1.0,
            **base_stage,
            "stage_strategies": (
                {**base_stage, "name": "stage-0"},
                {**base_stage, "name": "stage-1-a"},
            ),
            "runtime_executable": True,
        },
        {
            "name": "dp-chain-b",
            "time_cost": 2.0,
            **base_stage,
            "stage_strategies": (
                {**base_stage, "name": "stage-0"},
                {**base_stage, "name": "stage-1-b", "micro_batch_size": 2},
            ),
            "runtime_executable": True,
        },
    ]

    shortlist = build_short_run_shortlist(strategies, config)

    assert [strategy["name"] for strategy in shortlist] == ["dp-chain-a", "dp-chain-b"]
    assert _runtime_execution_key(shortlist[0]) != _runtime_execution_key(shortlist[1])


def test_build_short_run_shortlist_hashes_segment_stage_runtime_signatures():
    config = _config()
    config.experiment.auto_tuner.planner.topk_plans_for_short_run = 2
    base_stage = {
        "data_parallel_size": 2,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 2,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "use_distributed_optimizer": False,
        "sequence_parallel": True,
        "acc_step": 2,
        "micro_batch_size": 2,
        "num_layers_per_virtual_pipeline_stage": None,
        "use_recompute": False,
        "recompute_method": None,
        "recompute_granularity": None,
        "recompute_num_layers": None,
        "decoder_first_pipeline_num_layers": 5,
        "decoder_last_pipeline_num_layers": 5,
    }
    segment_stage = {
        **base_stage,
        "segment_partition_ranges": [[0, 1], [2, 4]],
        "segment_strategies": (
            {**base_stage, "tensor_model_parallel_size": 1, "data_parallel_size": 2},
            {**base_stage, "tensor_model_parallel_size": 2, "data_parallel_size": 1},
        ),
    }
    strategies = [
        {
            "name": "segment-chain-a",
            "time_cost": 1.0,
            **base_stage,
            "stage_strategies": (segment_stage, segment_stage),
            "runtime_executable": True,
        },
        {
            "name": "segment-chain-b",
            "time_cost": 2.0,
            **base_stage,
            "stage_strategies": (
                segment_stage,
                {**segment_stage, "segment_partition_ranges": [[0, 2], [3, 4]]},
            ),
            "runtime_executable": True,
        },
    ]

    shortlist = build_short_run_shortlist(strategies, config)

    assert [strategy["name"] for strategy in shortlist] == [
        "segment-chain-a",
        "segment-chain-b",
    ]
    assert _runtime_execution_key(shortlist[0]) != _runtime_execution_key(shortlist[1])
