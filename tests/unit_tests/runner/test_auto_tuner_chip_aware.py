import sys
from pathlib import Path

import pytest

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flagscale.runner.auto_tuner.search.algorithm import GridAlgo
from flagscale.runner.auto_tuner.search.searcher import Searcher


DEFAULT_MODEL = {
    "num_layers": 2,
    "global_batch_size": 2,
    "hidden_size": 2,
    "num_attention_heads": 2,
    "seq_length": 2,
}

DEFAULT_SPACE = {
    "data_parallel_size": [1],
    "use_distributed_optimizer": [False],
    "tensor_model_parallel_size": "auto",
    "sequence_parallel": [False],
    "pipeline_model_parallel_size": "auto",
    "num_layers_per_virtual_pipeline_stage": [0],
    "use_recompute": [False],
    "recompute_method": ["uniform"],
    "recompute_granularity": ["full"],
    "recompute_num_layers": [1],
    "micro_batch_size": [1],
    "context_parallel_size": "auto",
    "expert_model_parallel_size": [1],
}

CHIP_PROFILE = {
    "schema_version": "v1alpha1",
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
        "devices_per_node": 2,
        "homogeneous_only": True,
    },
    "strategy_hints": {
        "default_search_priority": "performance",
        "max_tensor_model_parallel_size": 2,
        "max_pipeline_model_parallel_size": 2,
        "disabled_dims": {},
    },
}

STRATEGIES = [
    {"label": "alpha", "chip_score": 0.2, "memory_model": 30},
    {"label": "beta", "chip_score": 0.9, "memory_model": 10},
    {"label": "gamma", "chip_score": 0.5, "memory_model": 20},
]


def _make_config(
    tmp_path,
    *,
    nnodes=1,
    nproc_per_node=2,
    space_overrides=None,
    algo_overrides=None,
    chip_profile=None,
    include_memory_model=False,
):
    space = dict(DEFAULT_SPACE)
    if space_overrides:
        space.update(space_overrides)

    auto_tuner = {
        "cards": nnodes * nproc_per_node,
        "nnodes": nnodes,
        "nproc_per_node": nproc_per_node,
        "algo": {"name": "grid", **(algo_overrides or {})},
        "platform": {},
        "space": space,
    }
    if chip_profile is not None:
        auto_tuner["chip_profile"] = {"profile": chip_profile}
    if include_memory_model:
        auto_tuner["memory_model"] = {}

    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {
                    "nnodes": nnodes,
                    "nproc_per_node": nproc_per_node,
                },
                "auto_tuner": auto_tuner,
            },
            "train": {
                "system": {},
                "model": dict(DEFAULT_MODEL),
            },
        }
    )


def _make_chip_profile(
    *,
    max_tensor_model_parallel_size=2,
    max_pipeline_model_parallel_size=2,
    disabled_dims=None,
    max_nodes=1,
    devices_per_node=2,
):
    profile = OmegaConf.create(CHIP_PROFILE)
    profile.strategy_hints.max_tensor_model_parallel_size = max_tensor_model_parallel_size
    profile.strategy_hints.max_pipeline_model_parallel_size = max_pipeline_model_parallel_size
    profile.strategy_hints.disabled_dims = disabled_dims or {}
    profile.topology.max_nodes = max_nodes
    profile.topology.devices_per_node = devices_per_node
    return OmegaConf.to_container(profile, resolve=True)


def _drain_labels(algo):
    labels = []
    while True:
        strategy = algo.search()
        if strategy is None:
            return labels
        labels.append(strategy["label"])


def test_searcher_applies_chip_profile_tensor_parallel_limit(tmp_path):
    config = _make_config(
        tmp_path,
        chip_profile=_make_chip_profile(max_tensor_model_parallel_size=1),
    )

    searcher = Searcher(config)

    assert searcher.space["tensor_model_parallel_size"] == [1]


def test_searcher_applies_chip_profile_pipeline_parallel_limit(tmp_path):
    config = _make_config(
        tmp_path,
        chip_profile=_make_chip_profile(max_pipeline_model_parallel_size=1),
    )

    searcher = Searcher(config)

    assert searcher.space["pipeline_model_parallel_size"] == [1]


def test_searcher_filters_disabled_dims_from_chip_profile(tmp_path):
    config = _make_config(
        tmp_path,
        chip_profile=_make_chip_profile(disabled_dims={"context_parallel_size": [2]}),
    )

    searcher = Searcher(config)

    assert searcher.space["context_parallel_size"] == [1]


def test_searcher_rejects_runtime_topology_outside_chip_profile(tmp_path):
    config = _make_config(
        tmp_path,
        nnodes=2,
        nproc_per_node=1,
        chip_profile=_make_chip_profile(max_nodes=1, devices_per_node=1),
    )

    with pytest.raises(ValueError, match="topology"):
        Searcher(config)


def test_grid_algo_uses_chip_score_order_when_chip_aware_scoring_enabled(tmp_path):
    config = _make_config(
        tmp_path,
        algo_overrides={"chip_aware_scoring": True},
        include_memory_model=True,
    )

    algo = GridAlgo(STRATEGIES, config)

    assert _drain_labels(algo) == ["beta", "gamma", "alpha"]


def test_grid_algo_keeps_input_order_without_chip_aware_scoring(tmp_path):
    config = _make_config(
        tmp_path,
        include_memory_model=True,
    )

    algo = GridAlgo(STRATEGIES, config)

    assert _drain_labels(algo) == ["alpha", "beta", "gamma"]
