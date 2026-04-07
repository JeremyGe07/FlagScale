import importlib
import sys
import types

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


def build_autotuner_config(tmp_path, chip_profile=None, algo=None, include_memory_model=True):
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

    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {"nnodes": 1, "nproc_per_node": 2},
                "auto_tuner": auto_tuner,
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


def _install_fake_cost_modules(monkeypatch, memory_result, time_result):
    cost_pkg = types.ModuleType("flagscale.runner.auto_tuner.cost")
    memory_mod = types.ModuleType("flagscale.runner.auto_tuner.cost.memory_cost")
    time_mod = types.ModuleType("flagscale.runner.auto_tuner.cost.time_cost")

    memory_mod.estimate_memory_cost = lambda strategy, config: memory_result
    time_mod.estimate_time_cost = lambda strategy, config: time_result
    cost_pkg.estimate_memory_cost = memory_mod.estimate_memory_cost
    cost_pkg.estimate_time_cost = time_mod.estimate_time_cost
    cost_pkg.memory_cost = memory_mod
    cost_pkg.time_cost = time_mod

    monkeypatch.setitem(sys.modules, "flagscale.runner.auto_tuner.cost", cost_pkg)
    monkeypatch.setitem(
        sys.modules, "flagscale.runner.auto_tuner.cost.memory_cost", memory_mod
    )
    monkeypatch.setitem(sys.modules, "flagscale.runner.auto_tuner.cost.time_cost", time_mod)


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


def test_searcher_injects_cost_fields_into_strategy(monkeypatch, tmp_path):
    memory_result = {
        "memory_total_mb": FAKE_MEMORY_TOTAL_MB,
        "memory_breakdown": {"peak_mb": 234.0, "reserved_mb": 56.0},
    }
    time_result = {
        "time_total_ms": FAKE_TIME_TOTAL_MS,
        "time_breakdown": {"compute_ms": 12.0, "pp_comm_ms": 3.0},
    }

    _install_fake_cost_modules(monkeypatch, memory_result, time_result)

    import flagscale.runner.auto_tuner.search.searcher as searcher_module

    searcher_module = importlib.reload(searcher_module)
    monkeypatch.setattr(
        searcher_module,
        "estimate_memory_cost",
        lambda strategy, config: memory_result,
        raising=False,
    )
    monkeypatch.setattr(
        searcher_module,
        "estimate_time_cost",
        lambda strategy, config: time_result,
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
    assert strategy["time_cost"] == time_result["time_total_ms"]
    assert strategy["time_breakdown"] == time_result["time_breakdown"]


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
