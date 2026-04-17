from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.search.pp_first_selection import (
    _runtime_execution_key,
    build_short_run_shortlist,
)


def _config():
    return OmegaConf.create(
        {
            "experiment": {
                "auto_tuner": {
                    "algo": {"name": "grid", "use_profiled_time_cost": True},
                    "planner": {"topk_plans_for_short_run": 4},
                }
            }
        }
    )


def _stage_strategies():
    return (
        {
            "data_parallel_size": 1,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 2,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "use_distributed_optimizer": False,
            "sequence_parallel": True,
            "acc_step": 16,
            "micro_batch_size": 2,
            "num_layers_per_virtual_pipeline_stage": None,
            "use_recompute": False,
            "recompute_method": None,
            "recompute_granularity": None,
            "recompute_num_layers": None,
        },
        {
            "data_parallel_size": 2,
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 2,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "use_distributed_optimizer": True,
            "sequence_parallel": True,
            "acc_step": 16,
            "micro_batch_size": 2,
            "num_layers_per_virtual_pipeline_stage": None,
            "use_recompute": False,
            "recompute_method": None,
            "recompute_granularity": None,
            "recompute_num_layers": None,
        },
    )


def _base_strategy(name, *, stage_partition_ranges, stage_device_groups):
    first_stage = _stage_strategies()[0]
    return {
        "name": name,
        "time_cost": 1.0,
        "plan_kind": "stage-heterogeneous",
        "runtime_mode": "stage-executable",
        "runtime_executable": True,
        "decoder_first_pipeline_num_layers": 2,
        "decoder_last_pipeline_num_layers": 2,
        "stage_partition_ranges": stage_partition_ranges,
        "stage_device_groups": stage_device_groups,
        "stage_strategies": _stage_strategies(),
        **first_stage,
    }


def test_build_short_run_shortlist_keeps_distinct_dp_stage_chains_with_same_stage_strategies():
    config = _config()
    strategies = [
        _base_strategy("dp-chain-left", stage_partition_ranges=[[0, 1], [2, 3]], stage_device_groups=[[0, 1], [2, 3]]),
        _base_strategy("dp-chain-right", stage_partition_ranges=[[0, 2], [3, 3]], stage_device_groups=[[0, 1, 2], [3]]),
    ]

    shortlist = build_short_run_shortlist(strategies, config)

    assert [strategy["name"] for strategy in shortlist] == [
        "dp-chain-left",
        "dp-chain-right",
    ]
    assert _runtime_execution_key(shortlist[0]) != _runtime_execution_key(shortlist[1])


def test_build_short_run_shortlist_drops_true_duplicate_dp_stage_chains():
    config = _config()
    strategies = [
        _base_strategy("dp-chain-primary", stage_partition_ranges=[[0, 1], [2, 3]], stage_device_groups=[[0, 1], [2, 3]]),
        _base_strategy("dp-chain-duplicate", stage_partition_ranges=[[0, 1], [2, 3]], stage_device_groups=[[0, 1], [2, 3]]),
    ]

    shortlist = build_short_run_shortlist(strategies, config)

    assert [strategy["name"] for strategy in shortlist] == ["dp-chain-primary"]
