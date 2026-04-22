from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.plan.lowering import lower_strategy_to_plan
from flagscale.runner.auto_tuner.plan.runtime import build_segment_hetero_runtime_overrides
from flagscale.runner.auto_tuner.plan.summary import (
    SEGMENT_HETEROGENEOUS_PLAN,
    plan_kind,
    summarize_plan,
)
from flagscale.runner.auto_tuner.plan.validator import validate_model_plan


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


def _config(tmp_path):
    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {"nnodes": 1, "nproc_per_node": 4},
                "auto_tuner": {
                    "cards": 4,
                    "nnodes": 1,
                    "nproc_per_node": 4,
                    "chip_profile": {"profile": _chip_profile()},
                },
            },
            "train": {
                "model": {
                    "num_layers": 4,
                    "global_batch_size": 8,
                    "hidden_size": 8,
                    "num_attention_heads": 4,
                    "seq_length": 16,
                },
                "system": {},
            },
        }
    )


def _segment_strategy(*, tp, dp):
    return {
        "context_parallel_size": 1,
        "data_parallel_size": dp,
        "device_type": "nvidia_l20",
        "expert_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "pp_local": 1,
        "sequence_parallel": True,
        "tensor_model_parallel_size": tp,
        "use_distributed_optimizer": False,
    }


def _segment_runtime_strategy():
    stage0_segments = (
        (0, 0, _segment_strategy(tp=2, dp=1)),
        (1, 1, _segment_strategy(tp=1, dp=2)),
    )
    stage1_segments = (
        (2, 2, _segment_strategy(tp=1, dp=2)),
        (3, 3, _segment_strategy(tp=2, dp=1)),
    )
    return {
        "idx": 0,
        "data_parallel_size": 1,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 2,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "micro_batch_size": 2,
        "acc_step": 4,
        "use_distributed_optimizer": False,
        "sequence_parallel": True,
        "num_layers_per_virtual_pipeline_stage": None,
        "recompute_method": None,
        "recompute_granularity": None,
        "recompute_num_layers": None,
        "decoder_first_pipeline_num_layers": None,
        "decoder_last_pipeline_num_layers": None,
        "stage_partition_ranges": [[0, 1], [2, 3]],
        "stage_device_groups": [[0, 1], [2, 3]],
        "stage_strategies": [
            {
                "segment_partition_ranges": [[0, 0], [1, 1]],
                "segment_strategies": [dict(segment) for _, _, segment in stage0_segments],
            },
            {
                "segment_partition_ranges": [[0, 0], [1, 1]],
                "segment_strategies": [dict(segment) for _, _, segment in stage1_segments],
            },
        ],
    }


def test_lower_strategy_to_plan_builds_segment_heterogeneous_plan_from_explicit_stage_metadata(
    tmp_path,
):
    config = _config(tmp_path)
    plan = lower_strategy_to_plan(_segment_runtime_strategy(), config)
    validation = validate_model_plan(plan)
    summary = summarize_plan(plan)

    assert plan_kind(plan) == SEGMENT_HETEROGENEOUS_PLAN
    assert validation.runtime_mode == "segment-executable"
    assert [(len(stage.segments)) for stage in plan.stages] == [2, 2]
    assert [(segment.start, segment.end) for segment in plan.stages[0].segments] == [
        (0, 0),
        (1, 1),
    ]
    assert summary["transitions"][0]["metadata"]["source_mesh"]["tensor_model_parallel_size"] == 2
    assert summary["transitions"][1]["metadata"]["target_mesh"]["tensor_model_parallel_size"] == 2


def test_build_segment_runtime_overrides_materializes_exact_contract(tmp_path):
    config = _config(tmp_path)
    strategy = _segment_runtime_strategy()
    plan = lower_strategy_to_plan(strategy, config)

    overrides = build_segment_hetero_runtime_overrides(strategy, config)
    runtime = overrides["hetero"]["segment_runtime"]

    assert runtime["hetero_stage_segment_splits"] == [[1, 1], [1, 1]]
    assert runtime["hetero_stage_segment_meshes"][0][0] == [2, 1, 1, 1, 1]
    assert runtime["hetero_stage_segment_transitions"][0]["kind"] == "segment-redistribution"
    assert set(runtime) == {
        "hetero_stage_segment_splits",
        "hetero_stage_segment_meshes",
        "hetero_stage_segment_transitions",
    }
    assert len(runtime["hetero_stage_segment_transitions"]) == len(plan.transitions)
