import pytest

from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.cost.memory_cost import estimate_memory_cost
from flagscale.runner.auto_tuner.cost.profile_store import build_cost_profile
from flagscale.runner.auto_tuner.cost.time_cost import estimate_time_cost
from flagscale.runner.auto_tuner.plan.lowering import lower_strategy_to_plan
from flagscale.runner.auto_tuner.plan.validator import validate_model_plan


def _build_chip_profile():
    return {
        "identity": {
            "name": "nvidia_l20",
            "vendor": "nvidia",
            "chip_class": "gpu",
        },
        "memory": {
            "total_memory_mb": 46000,
            "bandwidth_gbps": 864,
        },
        "compute": {
            "bf16_tflops": 119.5,
            "attention_tflops": 119.5,
        },
        "interconnect": {
            "intra_node": {
                "fabric": "pcie",
                "p2p_bandwidth_gbps": 64,
                "p2p_latency_us": 3,
                "all_reduce_bandwidth_gbps": 45,
                "all_reduce_latency_us": 8,
            },
            "host_device": {
                "bandwidth_gbps": 24,
                "latency_us": 10,
            },
        },
        "kernel_support": {
            "transformer_engine": True,
            "flash_attention": True,
            "fused_rmsnorm": True,
        },
        "topology": {
            "max_nodes": 1,
            "devices_per_node": 4,
            "homogeneous_only": True,
        },
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


def _make_config(
    tmp_path,
    *,
    num_layers=8,
    global_batch_size=8,
    cards=4,
):
    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {"nnodes": 1, "nproc_per_node": cards},
                "auto_tuner": {
                    "cards": cards,
                    "nnodes": 1,
                    "nproc_per_node": cards,
                    "chip_profile": {"profile": _build_chip_profile()},
                },
            },
            "train": {
                "model": {
                    "num_layers": num_layers,
                    "global_batch_size": global_batch_size,
                    "hidden_size": 64,
                    "num_attention_heads": 8,
                    "seq_length": 32,
                    "padded_vocab_size": 256,
                },
                "system": {},
            },
        }
    )


def _strategy(**overrides):
    strategy = {
        "data_parallel_size": 1,
        "use_distributed_optimizer": False,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 2,
        "num_layers_per_virtual_pipeline_stage": None,
        "use_recompute": False,
        "recompute_method": None,
        "recompute_granularity": None,
        "recompute_num_layers": None,
        "micro_batch_size": 2,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "acc_step": 4,
        "decoder_first_pipeline_num_layers": None,
        "decoder_last_pipeline_num_layers": None,
        "sequence_parallel": False,
    }
    strategy.update(overrides)
    return strategy


def test_build_cost_profile_accepts_homogeneous_model_plan(tmp_path):
    config = _make_config(tmp_path)
    strategy = _strategy()
    plan = lower_strategy_to_plan(strategy, config)

    profile = build_cost_profile(config, plan)

    assert profile["runtime"]["plan"]["stage_count"] == 2
    assert profile["runtime"]["plan"]["vpp_stage_segment_counts"] == [1, 1]
    assert profile["runtime"]["strategy"] == strategy


def test_estimate_memory_cost_accepts_model_plan_and_keeps_scalar_total(tmp_path):
    config = _make_config(tmp_path)
    strategy = _strategy()
    plan = lower_strategy_to_plan(strategy, config)

    strategy_cost = estimate_memory_cost(strategy, config)
    plan_cost = estimate_memory_cost(plan, config)

    assert plan_cost["memory_total_mb"] == pytest.approx(strategy_cost["memory_total_mb"])
    assert plan_cost["memory_breakdown"]["plan"]["stages"][0]["stage_id"] == 0
    assert plan_cost["memory_breakdown"]["plan"]["stages"][0]["segments"][0]["layer_count"] == 4
    assert plan_cost["memory_breakdown"]["plan"]["stages"][0]["segments"][0]["memory_total_mb"] > 0


def test_estimate_time_cost_accepts_model_plan_and_exposes_transitions(tmp_path):
    config = _make_config(tmp_path)
    strategy = _strategy()
    plan = lower_strategy_to_plan(strategy, config)

    strategy_cost = estimate_time_cost(strategy, config)
    plan_cost = estimate_time_cost(plan, config)

    assert plan_cost["time_total_ms"] == pytest.approx(strategy_cost["time_total_ms"])
    assert plan_cost["time_breakdown"]["plan"]["stages"][1]["segments"][0]["layer_count"] == 4
    assert plan_cost["time_breakdown"]["plan"]["transitions"] == [
        {"source_stage_id": 0, "target_stage_id": 1, "time_ms": pytest.approx(
            plan_cost["time_breakdown"]["pp_comm_ms"]
        )}
    ]


def test_plan_cost_models_consume_analysis_only_multisegment_plan(tmp_path):
    config = _make_config(tmp_path, num_layers=18, global_batch_size=12, cards=3)
    plan = lower_strategy_to_plan(
        _strategy(
            data_parallel_size=1,
            pipeline_model_parallel_size=3,
            num_layers_per_virtual_pipeline_stage=2,
            micro_batch_size=2,
            acc_step=6,
        ),
        config,
    )

    validation = validate_model_plan(plan)
    profile = build_cost_profile(config, plan)
    memory_cost = estimate_memory_cost(plan, config)
    time_cost = estimate_time_cost(plan, config)

    assert validation.runtime_mode == "analysis-only"
    assert profile["runtime"]["plan"]["vpp_stage_segment_counts"] == [3, 3, 3]
    assert memory_cost["memory_total_mb"] > 0
    assert len(memory_cost["memory_breakdown"]["plan"]["stages"][0]["segments"]) == 3
    assert time_cost["time_total_ms"] > 0
    assert len(time_cost["time_breakdown"]["plan"]["transitions"]) == 2
