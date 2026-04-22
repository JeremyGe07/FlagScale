from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.plan.lowering import lower_strategy_to_plan
from flagscale.runner.auto_tuner.plan.runtime import build_segment_hetero_runtime_overrides
from flagscale.runner.auto_tuner.plan.schema import (
    ModelPlan,
    SegmentPlan,
    StagePlan,
    TransitionPlan,
)
from flagscale.runner.auto_tuner.plan.segment_runtime_contract import (
    build_segment_runtime_contract,
)


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
        "sequence_parallel": True,
        "tensor_model_parallel_size": tp,
        "use_distributed_optimizer": False,
    }


def _segment_runtime_strategy():
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
                "segment_strategies": [
                    _segment_strategy(tp=2, dp=1),
                    _segment_strategy(tp=1, dp=2),
                ],
            },
            {
                "segment_partition_ranges": [[0, 0], [1, 1]],
                "segment_strategies": [
                    _segment_strategy(tp=1, dp=2),
                    _segment_strategy(tp=2, dp=1),
                ],
            },
        ],
    }


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


def test_build_segment_runtime_contract_keeps_only_segment_redistribution_transitions():
    plan = ModelPlan(
        stages=(
            StagePlan(
                stage_id=0,
                segments=(
                    SegmentPlan(
                        start=0,
                        end=0,
                        strategy={
                            "data_parallel_size": 1,
                            "tensor_model_parallel_size": 2,
                            "context_parallel_size": 1,
                            "expert_model_parallel_size": 1,
                            "pp_local": 1,
                        },
                    ),
                    SegmentPlan(
                        start=1,
                        end=1,
                        strategy={
                            "data_parallel_size": 2,
                            "tensor_model_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_model_parallel_size": 1,
                            "pp_local": 1,
                        },
                    ),
                ),
            ),
        ),
        transitions=(
            TransitionPlan(
                source_stage_id=0,
                target_stage_id=0,
                kind="segment-redistribution",
                source_segment_index=0,
                target_segment_index=1,
                metadata={
                    "source_mesh": {
                        "tensor_model_parallel_size": 2,
                        "context_parallel_size": 1,
                        "expert_model_parallel_size": 1,
                        "data_parallel_size": 1,
                        "pp_local": 1,
                    },
                    "target_mesh": {
                        "tensor_model_parallel_size": 1,
                        "context_parallel_size": 1,
                        "expert_model_parallel_size": 1,
                        "data_parallel_size": 2,
                        "pp_local": 1,
                    },
                },
            ),
            TransitionPlan(
                source_stage_id=0,
                target_stage_id=1,
                kind="pipeline",
                metadata={"buffer_layers": 1},
            ),
        ),
    )

    contract = build_segment_runtime_contract(plan)

    assert len(contract["hetero_stage_segment_transitions"]) == 1
    assert contract["hetero_stage_segment_transitions"][0]["kind"] == "segment-redistribution"
