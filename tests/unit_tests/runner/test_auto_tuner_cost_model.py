import importlib
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.search.algorithm import GridAlgo

SINGLE_STRATEGY_SPACE = {
    "data_parallel_size": [1],
    "use_distributed_optimizer": [False],
    "tensor_model_parallel_size": [2],
    "sequence_parallel": [False],
    "pipeline_model_parallel_size": [1],
    "num_layers_per_virtual_pipeline_stage": [0],
    "use_recompute": [False],
    "recompute_method": ["uniform"],
    "recompute_granularity": ["full"],
    "recompute_num_layers": [1],
    "micro_batch_size": [4],
    "context_parallel_size": [1],
    "expert_model_parallel_size": [1],
}

FAKE_MEMORY_TOTAL_MB = 1234.0
FAKE_TIME_TOTAL_MS = 78.0


def build_autotuner_config(
    tmp_path,
    chip_profile=None,
    algo=None,
    include_memory_model=True,
    runner_nnodes=1,
    runner_nproc_per_node=2,
    model_overrides=None,
):
    auto_tuner = {
        "algo": algo or {"name": "grid"},
    }
    if include_memory_model:
        auto_tuner["memory_model"] = {
            "model_name": "default",
            "gpu_memory": 46000,
            "gpu_utilization": [0.1, 1.0],
        }
    if chip_profile is not None:
        auto_tuner["chip_profile"] = chip_profile

    model = {
        "num_layers": 28,
        "hidden_size": 1536,
        "num_attention_heads": 12,
        "global_batch_size": 32,
        "seq_length": 2048,
        "padded_vocab_size": 32000,
    }
    if model_overrides:
        model.update(model_overrides)

    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {
                    "nnodes": runner_nnodes,
                    "nproc_per_node": runner_nproc_per_node,
                },
                "auto_tuner": auto_tuner,
            },
            "train": {
                "system": {"logging": {}},
                "model": model,
            },
        }
    )


def _estimate_memory_cost(strategy, config):
    from flagscale.runner.auto_tuner.cost.memory_cost import estimate_memory_cost

    return estimate_memory_cost(strategy, config)


def _estimate_time_cost(strategy, config):
    from flagscale.runner.auto_tuner.cost.time_cost import estimate_time_cost

    return estimate_time_cost(strategy, config)


def _build_chip_profile(
    *,
    fabric="pcie",
    p2p_bandwidth_gbps=64,
    p2p_latency_us=3,
    all_reduce_bandwidth_gbps=45,
    all_reduce_latency_us=8,
    reserved_memory_bias_mb=0,
    peak_activation_bias_mb=0,
):
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
                "fabric": fabric,
                "p2p_bandwidth_gbps": p2p_bandwidth_gbps,
                "p2p_latency_us": p2p_latency_us,
                "all_reduce_bandwidth_gbps": all_reduce_bandwidth_gbps,
                "all_reduce_latency_us": all_reduce_latency_us,
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
            "devices_per_node": 2,
            "homogeneous_only": True,
        },
        "strategy_hints": {
            "default_search_priority": "performance",
            "max_tensor_model_parallel_size": 2,
            "max_pipeline_model_parallel_size": 2,
            "disabled_dims": {},
        },
        "cost_model": {
            "reserved_memory_bias_mb": reserved_memory_bias_mb,
            "peak_activation_bias_mb": peak_activation_bias_mb,
            "overlap": {
                "dp_comm_overlap_ratio": 0.0,
                "tp_comm_overlap_ratio": 0.0,
                "pp_comm_overlap_ratio": 0.0,
            },
        },
    }


def _build_group_size_all_reduce_profile():
    profile = _build_chip_profile(
        all_reduce_bandwidth_gbps=45,
        all_reduce_latency_us=8,
    )
    profile["interconnect"]["intra_node"]["collective_profiles"] = {
        "all_reduce": {
            "group_size_2": {"bandwidth_gbps": 200, "latency_us": 2},
            "group_size_4": {"bandwidth_gbps": 80, "latency_us": 20},
        }
    }
    return profile


def _build_strategy(**overrides):
    strategy = {
        "data_parallel_size": 1,
        "use_distributed_optimizer": False,
        "tensor_model_parallel_size": 1,
        "sequence_parallel": False,
        "pipeline_model_parallel_size": 1,
        "num_layers_per_virtual_pipeline_stage": None,
        "use_recompute": False,
        "recompute_method": None,
        "recompute_granularity": None,
        "recompute_num_layers": None,
        "micro_batch_size": 4,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "acc_step": 8,
        "decoder_first_pipeline_num_layers": None,
        "decoder_last_pipeline_num_layers": None,
    }
    strategy.update(overrides)
    return strategy


def test_load_chip_profile_fills_default_cost_model_fields(tmp_path):
    from flagscale.runner.auto_tuner.chip_profile import load_chip_profile

    profile_path = tmp_path / "nvidia_l20.yaml"
    profile_path.write_text(
        """
schema_version: v1alpha1
identity:
  name: nvidia_l20
  vendor: nvidia
  chip_class: gpu
memory:
  total_memory_mb: 46000
  bandwidth_gbps: 864
compute:
  bf16_tflops: 119.5
  attention_tflops: 119.5
interconnect:
  intra_node:
    fabric: pcie
    p2p_bandwidth_gbps: 64
    p2p_latency_us: 3
    all_reduce_bandwidth_gbps: 45
    all_reduce_latency_us: 8
  host_device:
    bandwidth_gbps: 24
    latency_us: 10
kernel_support:
  transformer_engine: true
  flash_attention: true
  fused_rmsnorm: true
topology:
  max_nodes: 1
  devices_per_node: 2
  homogeneous_only: true
strategy_hints:
  default_search_priority: performance
  max_tensor_model_parallel_size: 2
  max_pipeline_model_parallel_size: 2
  disabled_dims: {}
""".strip(),
        encoding="utf-8",
    )

    profile = load_chip_profile(str(profile_path))

    assert profile["cost_model"] == {
        "reserved_memory_bias_mb": 0,
        "peak_activation_bias_mb": 0,
        "overlap": {
            "dp_comm_overlap_ratio": 0.0,
            "tp_comm_overlap_ratio": 0.0,
            "pp_comm_overlap_ratio": 0.0,
        },
    }


def test_build_cost_profile_returns_normalized_sections(tmp_path):
    from flagscale.runner.auto_tuner.cost.profile_store import build_cost_profile

    config = build_autotuner_config(
        tmp_path,
        chip_profile={"profile": _build_chip_profile()},
    )
    strategy = _build_strategy(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        micro_batch_size=4,
    )

    result = build_cost_profile(config, strategy)

    assert result["hardware"]["memory"]["total_memory_mb"] == 46000
    assert result["hardware"]["cost_model"]["peak_activation_bias_mb"] == 0
    assert result["runtime"]["nnodes"] == 1
    assert result["runtime"]["nproc_per_node"] == 2
    assert result["runtime"]["world_size"] == 2
    assert result["runtime"]["strategy"] == {
        "data_parallel_size": 1,
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "num_layers_per_virtual_pipeline_stage": None,
        "recompute_method": None,
        "recompute_granularity": None,
        "recompute_num_layers": None,
        "micro_batch_size": 4,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "acc_step": 8,
        "decoder_first_pipeline_num_layers": None,
        "decoder_last_pipeline_num_layers": None,
        "sequence_parallel": False,
        "use_distributed_optimizer": False,
        "use_recompute": False,
    }
    assert result["runtime"]["plan"]["stage_count"] == 1
    assert result["model"] == {
        "num_layers": 28,
        "hidden_size": 1536,
        "num_attention_heads": 12,
        "global_batch_size": 32,
        "seq_length": 2048,
    }


def test_build_cost_profile_normalizes_inline_attached_profile_defaults(tmp_path):
    from flagscale.runner.auto_tuner.cost.profile_store import build_cost_profile

    inline_profile = _build_chip_profile()
    inline_profile.pop("cost_model")
    config = build_autotuner_config(
        tmp_path,
        chip_profile={"profile": inline_profile},
    )

    result = build_cost_profile(config, _build_strategy())

    assert result["hardware"]["cost_model"] == {
        "reserved_memory_bias_mb": 0,
        "peak_activation_bias_mb": 0,
        "overlap": {
            "dp_comm_overlap_ratio": 0.0,
            "tp_comm_overlap_ratio": 0.0,
            "pp_comm_overlap_ratio": 0.0,
        },
    }


def test_build_cost_profile_loads_chip_profile_from_path(tmp_path):
    from flagscale.runner.auto_tuner.cost.profile_store import build_cost_profile

    profile_path = tmp_path / "nvidia_l20.yaml"
    OmegaConf.save(
        config=OmegaConf.create(_build_chip_profile()),
        f=str(profile_path),
    )
    config = build_autotuner_config(
        tmp_path,
        chip_profile={"path": str(profile_path)},
    )

    result = build_cost_profile(config, _build_strategy())

    assert result["hardware"]["identity"]["name"] == "nvidia_l20"
    assert result["hardware"]["cost_model"]["reserved_memory_bias_mb"] == 0


def test_build_cost_profile_supports_plain_mapping_config(tmp_path):
    from flagscale.runner.auto_tuner.cost.profile_store import build_cost_profile

    config = {
        "experiment": {
            "exp_dir": str(tmp_path),
            "runner": {"nnodes": 1, "nproc_per_node": 2},
            "auto_tuner": {"chip_profile": {"profile": _build_chip_profile()}},
        },
        "train": {
            "system": {"logging": {}},
            "model": {
                "num_layers": 28,
                "hidden_size": 1536,
                "num_attention_heads": 12,
                "global_batch_size": 32,
                "seq_length": 2048,
            },
        },
    }

    result = build_cost_profile(config, _build_strategy())

    assert result["runtime"]["world_size"] == 2
    assert result["runtime"]["strategy"]["acc_step"] == 8


def test_build_cost_profile_requires_chip_profile_input(tmp_path):
    from flagscale.runner.auto_tuner.cost.profile_store import build_cost_profile

    config = build_autotuner_config(tmp_path)

    with pytest.raises(ValueError, match="config.experiment.auto_tuner.chip_profile"):
        build_cost_profile(config, _build_strategy())


def test_build_cost_profile_requires_train_model_fields(tmp_path):
    from flagscale.runner.auto_tuner.cost.profile_store import build_cost_profile

    config = build_autotuner_config(
        tmp_path,
        chip_profile={"profile": _build_chip_profile()},
    )
    del config.train.model.seq_length

    with pytest.raises(ValueError, match="config.train.model.seq_length"):
        build_cost_profile(config, _build_strategy())


def test_build_cost_profile_requires_strategy_fields(tmp_path):
    from flagscale.runner.auto_tuner.cost.profile_store import build_cost_profile

    config = build_autotuner_config(
        tmp_path,
        chip_profile={"profile": _build_chip_profile()},
    )
    strategy = _build_strategy()
    del strategy["acc_step"]

    with pytest.raises(ValueError, match="strategy.acc_step"):
        build_cost_profile(config, strategy)


def test_load_chip_profile_rejects_invalid_optional_p2p_class(tmp_path):
    from flagscale.runner.auto_tuner.chip_profile import normalize_chip_profile

    profile = _build_chip_profile()
    profile["interconnect"]["intra_node"]["p2p_classes"] = {
        "sys": {"bandwidth_gbps": 0, "latency_us": 11.4, "gpu_pair": [0, 2]}
    }

    with pytest.raises(ValueError, match="p2p_classes.sys.bandwidth_gbps"):
        normalize_chip_profile(profile)


def test_load_chip_profile_rejects_invalid_all_reduce_group_profile(tmp_path):
    from flagscale.runner.auto_tuner.chip_profile import normalize_chip_profile

    profile = _build_chip_profile()
    profile["interconnect"]["intra_node"]["collective_profiles"] = {
        "all_reduce": {"group_size_zero": {"bandwidth_gbps": 80, "latency_us": 20}}
    }

    with pytest.raises(ValueError, match="group_size_zero"):
        normalize_chip_profile(profile)


def test_memory_cost_returns_breakdown_and_total(tmp_path):
    config = build_autotuner_config(
        tmp_path,
        chip_profile={"profile": _build_chip_profile()},
    )
    strategy = _build_strategy()

    result = _estimate_memory_cost(strategy, config)

    assert result["memory_total_mb"] > 0
    assert result["memory_breakdown"]["reserved_mb"] >= 0
    assert result["memory_breakdown"]["peak_mb"] >= 0


def test_memory_cost_increases_when_reserved_bias_is_present(tmp_path):
    strategy = _build_strategy()
    config_without_reserved_bias = build_autotuner_config(
        tmp_path / "without_bias",
        chip_profile={"profile": _build_chip_profile(reserved_memory_bias_mb=0)},
    )
    config_with_reserved_bias = build_autotuner_config(
        tmp_path / "with_bias",
        chip_profile={"profile": _build_chip_profile(reserved_memory_bias_mb=512)},
    )

    without_bias = _estimate_memory_cost(strategy, config_without_reserved_bias)
    with_bias = _estimate_memory_cost(strategy, config_with_reserved_bias)

    assert with_bias["memory_total_mb"] > without_bias["memory_total_mb"]
    assert with_bias["memory_breakdown"]["reserved_mb"] > without_bias["memory_breakdown"][
        "reserved_mb"
    ]


def test_peak_activation_bias_is_reported_without_pruning_total(tmp_path):
    strategy = _build_strategy()
    config_without_peak_bias = build_autotuner_config(
        tmp_path / "without_peak_bias",
        chip_profile={"profile": _build_chip_profile(peak_activation_bias_mb=0)},
    )
    config_with_peak_bias = build_autotuner_config(
        tmp_path / "with_peak_bias",
        chip_profile={"profile": _build_chip_profile(peak_activation_bias_mb=512)},
    )

    without_bias = _estimate_memory_cost(strategy, config_without_peak_bias)
    with_bias = _estimate_memory_cost(strategy, config_with_peak_bias)

    assert with_bias["memory_total_mb"] == pytest.approx(without_bias["memory_total_mb"])
    assert with_bias["memory_breakdown"]["peak_activation_bias_mb"] == pytest.approx(512.0)
    assert with_bias["memory_breakdown"]["profiled_peak_mb"] == pytest.approx(
        without_bias["memory_breakdown"]["peak_mb"] + 512.0
    )


def test_memory_cost_reports_recompute_saved_memory(tmp_path):
    config = build_autotuner_config(
        tmp_path,
        chip_profile={"profile": _build_chip_profile()},
    )
    strategy_without_recompute = _build_strategy()
    strategy_with_recompute = _build_strategy(
        use_recompute=True,
        recompute_method="uniform",
        recompute_granularity="full",
        recompute_num_layers=28,
    )

    no_recompute = _estimate_memory_cost(strategy_without_recompute, config)
    with_recompute = _estimate_memory_cost(strategy_with_recompute, config)

    assert with_recompute["memory_breakdown"]["recompute_saved_mb"] >= 0
    assert with_recompute["memory_total_mb"] <= no_recompute["memory_total_mb"]


def test_memory_cost_reuses_megatron_moe_layer_freq_parsing(tmp_path):
    from flagscale.runner.auto_tuner.cost.memory_cost import _build_memory_args

    config = build_autotuner_config(
        tmp_path,
        chip_profile={"profile": _build_chip_profile()},
    )
    config.train.model.num_experts = 8
    config.train.model.moe_router_topk = 2
    config.train.model.moe_layer_freq = "[0]+[1]*3"

    args = _build_memory_args(config, _build_strategy())

    assert args.moe_layer_freq == [0, 1, 1, 1] * 7


def test_memory_cost_slices_moe_layer_freq_for_pipeline_segments(tmp_path):
    config = build_autotuner_config(
        tmp_path,
        chip_profile={"profile": _build_chip_profile()},
        model_overrides={
            "num_layers": 4,
            "ffn_hidden_size": 4096,
            "moe_ffn_hidden_size": 1024,
            "num_experts": 8,
            "moe_router_topk": 2,
            "moe_layer_freq": "[0]+[1]*3",
        },
    )
    strategy = _build_strategy(pipeline_model_parallel_size=2)

    result = _estimate_memory_cost(strategy, config)

    assert result["memory_breakdown"]["plan"]["stages"][0]["segments"][0]["layer_count"] == 2


def test_memory_cost_estimator_does_not_print_to_stdout(tmp_path, capsys):
    config = build_autotuner_config(
        tmp_path,
        chip_profile={"profile": _build_chip_profile()},
    )

    _estimate_memory_cost(_build_strategy(), config)

    captured = capsys.readouterr()
    assert captured.out == ""


def test_time_cost_returns_breakdown_and_total(tmp_path):
    config = build_autotuner_config(
        tmp_path,
        chip_profile={"profile": _build_chip_profile()},
    )
    strategy = _build_strategy()

    result = _estimate_time_cost(strategy, config)

    assert result["time_total_ms"] > 0
    assert result["time_breakdown"]["compute_ms"] > 0


def test_time_cost_reports_higher_pp_comm_on_weaker_fabric(tmp_path):
    strategy = _build_strategy(
        pipeline_model_parallel_size=2,
        decoder_first_pipeline_num_layers=14,
    )
    pcie_config = build_autotuner_config(
        tmp_path / "pcie",
        chip_profile={"profile": _build_chip_profile(fabric="pcie")},
    )
    fast_fabric_config = build_autotuner_config(
        tmp_path / "fast",
        chip_profile={
            "profile": _build_chip_profile(
                fabric="nvlink",
                p2p_bandwidth_gbps=900,
                p2p_latency_us=1,
                all_reduce_bandwidth_gbps=450,
                all_reduce_latency_us=2,
            )
        },
    )

    pcie = _estimate_time_cost(strategy, pcie_config)
    fast_fabric = _estimate_time_cost(strategy, fast_fabric_config)

    assert pcie["time_breakdown"]["pp_comm_ms"] > 0
    assert fast_fabric["time_breakdown"]["pp_comm_ms"] > 0
    assert (
        pcie["time_breakdown"]["pp_comm_ms"]
        > fast_fabric["time_breakdown"]["pp_comm_ms"]
    )
    assert pcie["time_breakdown"]["compute_ms"] == pytest.approx(
        fast_fabric["time_breakdown"]["compute_ms"]
    )


def test_time_cost_cross_node_dp_comm_is_more_expensive_than_single_node(tmp_path):
    strategy = _build_strategy(data_parallel_size=2)
    single_node_config = build_autotuner_config(
        tmp_path / "single-node",
        chip_profile={"profile": _build_group_size_all_reduce_profile()},
        runner_nnodes=1,
        runner_nproc_per_node=2,
    )
    cross_node_config = build_autotuner_config(
        tmp_path / "cross-node",
        chip_profile={"profile": _build_group_size_all_reduce_profile()},
        runner_nnodes=2,
        runner_nproc_per_node=1,
    )

    single_node = _estimate_time_cost(strategy, single_node_config)
    cross_node = _estimate_time_cost(strategy, cross_node_config)

    assert cross_node["time_breakdown"]["dp_comm_ms"] > single_node["time_breakdown"]["dp_comm_ms"]


def test_resolve_all_reduce_metrics_prefers_group_size_specific_profile():
    from flagscale.runner.auto_tuner.cost.collective_profiles import (
        resolve_all_reduce_metrics,
    )

    interconnect = _build_chip_profile()["interconnect"]
    interconnect["intra_node"]["collective_profiles"] = {
        "all_reduce": {
            "group_size_2": {"bandwidth_gbps": 200, "latency_us": 2},
            "group_size_4": {"bandwidth_gbps": 80, "latency_us": 20},
        }
    }

    assert resolve_all_reduce_metrics(interconnect, 2) == (200.0, 2.0)


def test_resolve_all_reduce_metrics_falls_back_to_legacy_all_reduce_scalar():
    from flagscale.runner.auto_tuner.cost.collective_profiles import (
        resolve_all_reduce_metrics,
    )

    interconnect = _build_chip_profile(
        all_reduce_bandwidth_gbps=45,
        all_reduce_latency_us=8,
    )["interconnect"]

    assert resolve_all_reduce_metrics(interconnect, 2) == (45.0, 8.0)


def test_time_cost_prefers_group_size_specific_all_reduce_profile(tmp_path):
    config = build_autotuner_config(
        tmp_path,
        chip_profile={"profile": _build_group_size_all_reduce_profile()},
        runner_nnodes=1,
        runner_nproc_per_node=4,
    )

    dp2 = _estimate_time_cost(_build_strategy(data_parallel_size=2), config)
    dp4 = _estimate_time_cost(_build_strategy(data_parallel_size=4), config)

    assert dp2["time_breakdown"]["dp_comm_ms"] < dp4["time_breakdown"]["dp_comm_ms"]


def test_time_cost_uses_tp_group_size_specific_all_reduce_profile(tmp_path):
    config = build_autotuner_config(
        tmp_path,
        chip_profile={"profile": _build_group_size_all_reduce_profile()},
        runner_nnodes=1,
        runner_nproc_per_node=4,
    )

    tp2 = _estimate_time_cost(_build_strategy(tensor_model_parallel_size=2), config)
    tp4 = _estimate_time_cost(_build_strategy(tensor_model_parallel_size=4), config)

    assert tp2["time_breakdown"]["tp_comm_ms"] < tp4["time_breakdown"]["tp_comm_ms"]


def test_time_cost_uses_ep_group_size_specific_all_reduce_profile(tmp_path):
    config = build_autotuner_config(
        tmp_path,
        chip_profile={"profile": _build_group_size_all_reduce_profile()},
        runner_nnodes=1,
        runner_nproc_per_node=4,
        model_overrides={
            "num_experts": 8,
            "moe_router_topk": 2,
            "moe_layer_freq": 1,
            "moe_token_dispatcher_type": "alltoall",
        },
    )

    ep2 = _estimate_time_cost(_build_strategy(expert_model_parallel_size=2), config)
    ep4 = _estimate_time_cost(_build_strategy(expert_model_parallel_size=4), config)

    assert ep2["time_breakdown"]["expert_comm_ms"] < ep4["time_breakdown"]["expert_comm_ms"]


def test_time_cost_falls_back_to_legacy_all_reduce_scalar_when_group_profile_missing(
    tmp_path,
):
    config = build_autotuner_config(
        tmp_path,
        chip_profile={
            "profile": _build_chip_profile(
                all_reduce_bandwidth_gbps=45,
                all_reduce_latency_us=8,
            )
        },
        runner_nnodes=1,
        runner_nproc_per_node=4,
    )

    result = _estimate_time_cost(_build_strategy(data_parallel_size=2), config)

    assert result["time_breakdown"]["dp_comm_ms"] > 0


def test_time_cost_dense_ep_does_not_reduce_compute_or_dp_cost(tmp_path):
    config = build_autotuner_config(
        tmp_path,
        chip_profile={"profile": _build_chip_profile()},
        runner_nnodes=1,
        runner_nproc_per_node=4,
    )
    without_ep = _estimate_time_cost(
        _build_strategy(data_parallel_size=2, expert_model_parallel_size=1),
        config,
    )
    with_ep = _estimate_time_cost(
        _build_strategy(data_parallel_size=2, expert_model_parallel_size=4),
        config,
    )

    assert with_ep["time_breakdown"]["compute_ms"] == pytest.approx(
        without_ep["time_breakdown"]["compute_ms"]
    )
    assert with_ep["time_breakdown"]["dp_comm_ms"] == pytest.approx(
        without_ep["time_breakdown"]["dp_comm_ms"]
    )


def test_time_cost_moe_ep_adds_explicit_expert_comm_penalty(tmp_path):
    config = build_autotuner_config(
        tmp_path,
        chip_profile={"profile": _build_chip_profile()},
        runner_nnodes=1,
        runner_nproc_per_node=4,
        model_overrides={
            "num_experts": 8,
            "moe_router_topk": 2,
            "moe_layer_freq": 1,
            "moe_token_dispatcher_type": "alltoall",
        },
    )
    no_ep = _estimate_time_cost(
        _build_strategy(expert_model_parallel_size=1),
        config,
    )
    with_ep = _estimate_time_cost(
        _build_strategy(expert_model_parallel_size=4),
        config,
    )

    assert with_ep["time_breakdown"]["expert_comm_ms"] > 0
    assert with_ep["time_total_ms"] > (no_ep["time_total_ms"] / 4.0)


def test_time_cost_moe_layer_freq_int_matches_materialized_pattern(tmp_path):
    model_overrides = {
        "num_experts": 8,
        "moe_router_topk": 2,
        "moe_token_dispatcher_type": "alltoall",
    }
    int_freq_config = build_autotuner_config(
        tmp_path / "int-freq",
        chip_profile={"profile": _build_chip_profile()},
        runner_nnodes=1,
        runner_nproc_per_node=4,
        model_overrides={**model_overrides, "moe_layer_freq": 2},
    )
    pattern = [1 if (i % 2 == 0) else 0 for i in range(28)]
    pattern_config = build_autotuner_config(
        tmp_path / "pattern-freq",
        chip_profile={"profile": _build_chip_profile()},
        runner_nnodes=1,
        runner_nproc_per_node=4,
        model_overrides={**model_overrides, "moe_layer_freq": pattern},
    )

    int_freq = _estimate_time_cost(_build_strategy(), int_freq_config)
    pattern_freq = _estimate_time_cost(_build_strategy(), pattern_config)

    assert int_freq["time_breakdown"]["compute_ms"] == pytest.approx(
        pattern_freq["time_breakdown"]["compute_ms"]
    )
    assert int_freq["time_breakdown"]["expert_comm_ms"] == pytest.approx(
        pattern_freq["time_breakdown"]["expert_comm_ms"]
    )


def test_time_cost_tp_does_not_treat_cp_as_tp_cross_node_trigger(tmp_path):
    strategy = _build_strategy(tensor_model_parallel_size=2, context_parallel_size=2)
    single_node_config = build_autotuner_config(
        tmp_path / "single-node",
        chip_profile={"profile": _build_chip_profile()},
        runner_nnodes=1,
        runner_nproc_per_node=4,
    )
    multi_node_config = build_autotuner_config(
        tmp_path / "multi-node",
        chip_profile={"profile": _build_chip_profile()},
        runner_nnodes=2,
        runner_nproc_per_node=2,
    )

    single_node = _estimate_time_cost(strategy, single_node_config)
    multi_node = _estimate_time_cost(strategy, multi_node_config)

    assert multi_node["time_breakdown"]["tp_comm_ms"] == pytest.approx(
        single_node["time_breakdown"]["tp_comm_ms"]
    )


def test_time_cost_ep_keeps_intra_node_when_ep_group_fits_on_one_node(tmp_path):
    strategy = _build_strategy(expert_model_parallel_size=2)
    model_overrides = {
        "num_experts": 8,
        "moe_router_topk": 2,
        "moe_layer_freq": 1,
        "moe_token_dispatcher_type": "alltoall",
    }
    single_node_config = build_autotuner_config(
        tmp_path / "single-node",
        chip_profile={"profile": _build_chip_profile()},
        runner_nnodes=1,
        runner_nproc_per_node=4,
        model_overrides=model_overrides,
    )
    multi_node_config = build_autotuner_config(
        tmp_path / "multi-node",
        chip_profile={"profile": _build_chip_profile()},
        runner_nnodes=2,
        runner_nproc_per_node=4,
        model_overrides=model_overrides,
    )

    single_node = _estimate_time_cost(strategy, single_node_config)
    multi_node = _estimate_time_cost(strategy, multi_node_config)

    assert multi_node["time_breakdown"]["expert_comm_ms"] == pytest.approx(
        single_node["time_breakdown"]["expert_comm_ms"]
    )


def test_calculate_hetero_memory_uses_megatron_args_converter(monkeypatch, tmp_path):
    import flagscale.runner.auto_tuner.hetero.hetero_theoretical_memory as hetero_memory_module
    import flagscale.runner.auto_tuner.memory_model as memory_model_module
    import flagscale.runner.auto_tuner.utils as utils_module

    observed = {}
    strategy = {"data_parallel_size": 1}
    config = build_autotuner_config(tmp_path)

    def fake_convert_config_to_megatron_args(actual_config, actual_strategy):
        observed["config"] = actual_config
        observed["strategy"] = actual_strategy
        return SimpleNamespace()

    def fake_hetero_report_theoretical_memory(strategy, config, base_args):
        observed["base_args"] = base_args
        return [2048.0]

    monkeypatch.setattr(
        utils_module,
        "convert_config_to_megatron_args",
        fake_convert_config_to_megatron_args,
    )
    monkeypatch.setattr(
        hetero_memory_module,
        "hetero_report_theoretical_memory",
        fake_hetero_report_theoretical_memory,
    )

    memory_model_module = importlib.reload(memory_model_module)

    assert memory_model_module.calculate_hetero_memory(strategy, config) == [2048.0]
    assert observed["config"] is config
    assert observed["strategy"] is strategy
    assert observed["base_args"].global_batch_size == config.train.model.global_batch_size


def test_searcher_only_injects_memory_cost_fields_into_strategy(monkeypatch, tmp_path):
    memory_result = {
        "memory_total_mb": FAKE_MEMORY_TOTAL_MB,
        "memory_breakdown": {"peak_mb": 234.0, "reserved_mb": 56.0},
    }

    import flagscale.runner.auto_tuner.search.searcher as searcher_module

    monkeypatch.setattr(
        searcher_module,
        "estimate_memory_cost",
        lambda strategy, config: memory_result,
        raising=False,
    )

    config = build_autotuner_config(
        tmp_path,
        chip_profile={"profile": _build_chip_profile()},
        include_memory_model=False,
    )
    config.experiment.auto_tuner.space = OmegaConf.create(SINGLE_STRATEGY_SPACE)
    searcher = searcher_module.Searcher(config)

    strategy = searcher.strategies[0]

    assert strategy["memory_model"] == FAKE_MEMORY_TOTAL_MB
    assert strategy["memory_breakdown"] == memory_result["memory_breakdown"]
    assert "time_cost" not in strategy
    assert "time_breakdown" not in strategy


def test_grid_algo_can_sort_by_time_cost(tmp_path):
    config_with_time_cost_sorting = build_autotuner_config(
        tmp_path,
        algo={"name": "grid", "use_profiled_time_cost": True},
        include_memory_model=False,
    )
    algo = GridAlgo(
        [{"name": "slow", "time_cost": 20}, {"name": "fast", "time_cost": 10}],
        config_with_time_cost_sorting,
    )

    assert algo.search()["name"] == "fast"
