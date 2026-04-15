import pytest

from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.search.pp_first_partition import (
    POLICY_LAYER_COUNT_BALANCED,
    build_layer_count_balanced_partition,
    generate_partition_candidates,
    is_power_of_two,
)
from flagscale.runner.auto_tuner.search.pp_first_searcher import PPFirstSearcher
from flagscale.runner.auto_tuner.tuner import AutoTuner


def _chip_profile():
    return {
        "identity": {"name": "nvidia_l20", "vendor": "nvidia", "chip_class": "gpu"},
        "memory": {"total_memory_mb": 46000, "bandwidth_gbps": 864},
        "compute": {"bf16_tflops": 119.5, "attention_tflops": 119.5},
        "interconnect": {
            "intra_node": {
                "fabric": "pcie",
                "p2p_bandwidth_gbps": 64,
                "p2p_latency_us": 3,
                "all_reduce_bandwidth_gbps": 45,
                "all_reduce_latency_us": 8,
            },
            "host_device": {"bandwidth_gbps": 24, "latency_us": 10},
        },
        "kernel_support": {
            "transformer_engine": True,
            "flash_attention": True,
            "fused_rmsnorm": True,
        },
        "topology": {"max_nodes": 1, "devices_per_node": 4, "homogeneous_only": True},
        "strategy_hints": {
            "default_search_priority": "performance",
            "max_tensor_model_parallel_size": 4,
            "max_pipeline_model_parallel_size": 4,
            "disabled_dims": {},
        },
        "cost_model": {
            "reserved_memory_bias_mb": 0,
            "peak_activation_bias_mb": 0,
            "overlap": {
                "dp_comm_overlap_ratio": 0.0,
                "tp_comm_overlap_ratio": 0.0,
                "pp_comm_overlap_ratio": 0.0,
            },
        },
    }


def _config(tmp_path, *, cards=4, pps=None, planner=None):
    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {"nnodes": 1, "nproc_per_node": cards},
                "auto_tuner": {
                    "algo": {"name": "grid"},
                    "control": {"train_iters": 3},
                    "chip_profile": {"profile": _chip_profile()},
                    "space": {
                        "data_parallel_size": [1, 2, 4],
                        "use_distributed_optimizer": [False, True],
                        "tensor_model_parallel_size": [1, 2, 4],
                        "sequence_parallel": [False, True],
                        "pipeline_model_parallel_size": pps or [1, 2, 3, 4],
                        "num_layers_per_virtual_pipeline_stage": [0],
                        "context_parallel_size": [1],
                        "expert_model_parallel_size": [1],
                        "micro_batch_size": [1, 2],
                        "use_recompute": [False],
                    },
                },
            },
            "train": {
                "system": {"logging": {}, "checkpoint": {"save_interval": 100}},
                "model": {
                    "num_layers": 10,
                    "global_batch_size": 8,
                    "hidden_size": 64,
                    "num_attention_heads": 8,
                    "seq_length": 32,
                    "padded_vocab_size": 256,
                    "eval_iters": 0,
                    "optimizer": {"lr_scheduler": {"lr": 1e-5, "min_lr": 0}},
                },
            },
        }
    )


def test_is_power_of_two_only_accepts_positive_powers():
    assert is_power_of_two(1) is True
    assert is_power_of_two(4) is True
    assert is_power_of_two(3) is False
    assert is_power_of_two(0) is False


def test_layer_count_balanced_partition_builds_contiguous_ranges():
    partition = build_layer_count_balanced_partition(num_layers=10, pp_degree=4, world_size=8)

    assert partition.partition_policy == POLICY_LAYER_COUNT_BALANCED
    assert partition.stage_ranges == ((0, 2), (3, 4), (5, 6), (7, 9))
    assert partition.device_groups == ((0, 1), (2, 3), (4, 5), (6, 7))


def test_generate_partition_candidates_honors_max_partitions_budget():
    partitions = generate_partition_candidates(
        num_layers=10,
        pp_degree=4,
        world_size=8,
        partition_policy=POLICY_LAYER_COUNT_BALANCED,
        max_partitions=1,
    )

    assert len(partitions) == 1
    assert partitions[0].partition_policy == POLICY_LAYER_COUNT_BALANCED


def test_generate_partition_candidates_rejects_unknown_policy():
    with pytest.raises(ValueError, match="Unsupported partition policy"):
        generate_partition_candidates(
            num_layers=10,
            pp_degree=4,
            world_size=8,
            partition_policy="unknown",
            max_partitions=1,
        )


def test_pp_first_searcher_limits_pipeline_candidates_to_power_of_two(tmp_path):
    searcher = PPFirstSearcher(_config(tmp_path))

    assert searcher.space["pipeline_model_parallel_size"] == [1, 2, 4]
    assert all(is_power_of_two(value) for value in searcher.space["pipeline_model_parallel_size"])


def test_pp_first_searcher_injects_partition_metadata(tmp_path):
    config = _config(
        tmp_path,
        planner={
            "name": "pp_first",
            "max_pp_candidates": 2,
            "max_assignments_per_partition": 4,
            "topk_plans_for_short_run": 3,
        },
    )
    config.experiment.auto_tuner.planner = {
        "name": "pp_first",
        "max_pp_candidates": 2,
        "max_assignments_per_partition": 4,
        "topk_plans_for_short_run": 3,
    }

    searcher = PPFirstSearcher(config)

    assert len(searcher.strategies) <= 8
    first = searcher.strategies[0]
    assert first["planner_name"] == "pp_first"
    assert first["partition_policy"] == POLICY_LAYER_COUNT_BALANCED
    assert first["provenance"] == "heuristic"
    assert "topology_signature" in first
    assert "stage_partition_ranges" in first
    assert first["topk_plans_for_short_run"] == 3
    assert first["planner_budget"]["max_pp_candidates"] == 2
    assert first["planner_budget"]["max_partitions_per_pp"] == 1
    assert first["estimate_metric"] in {"search_order", "memory_model", "chip_score", "time_cost"}
    assert isinstance(first["estimated_stage_costs"], list)
    assert first["partition_candidate_count"] == 1
    assert first["assignment_candidate_rank"] == 0


def test_pp_first_searcher_marks_only_executable_topk_for_short_run(tmp_path):
    config = _config(tmp_path)
    config.experiment.auto_tuner.planner = {
        "name": "pp_first",
        "topk_plans_for_short_run": 2,
    }

    searcher = PPFirstSearcher(config)
    marked = [s for s in searcher.strategies if s["short_run_candidate"]]

    assert len(marked) == 2
    assert all(s["runtime_executable"] is True for s in marked)
    assert len(searcher.short_run_strategies) == 2
    assert all("estimate_rank" in s for s in searcher.strategies)
    assert all("short_run_shortlist_count" in s for s in searcher.strategies)
    assert {s["short_run_shortlist_count"] for s in searcher.strategies} == {2}


def test_pp_first_searcher_prefers_time_cost_when_profiled_time_enabled(tmp_path):
    config = _config(tmp_path)
    config.experiment.auto_tuner.algo.use_profiled_time_cost = True
    config.experiment.auto_tuner.memory_model = {"model_name": "default", "gpu_memory": 46000}
    config.experiment.auto_tuner.planner = {"name": "pp_first", "topk_plans_for_short_run": 2}

    searcher = PPFirstSearcher(config)

    assert all(strategy["estimate_metric"] == "time_cost" for strategy in searcher.strategies)
    assert all("time_cost" in strategy for strategy in searcher.short_run_strategies)


def test_auto_tuner_uses_pp_first_searcher_when_planner_enabled(tmp_path):
    config = _config(tmp_path)
    config.experiment.auto_tuner.planner = {"name": "pp_first"}

    tuner = AutoTuner(config)

    assert isinstance(tuner.searcher, PPFirstSearcher)


def test_auto_tuner_rejects_unknown_planner_name(tmp_path):
    config = _config(tmp_path)
    config.experiment.auto_tuner.planner = {"name": "unknown"}

    with pytest.raises(ValueError, match="Unsupported auto_tuner planner"):
        AutoTuner(config)
