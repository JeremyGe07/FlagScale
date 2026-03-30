import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
FLAGSCALE_ROOT = ROOT / "flagscale"


def _import_flagscale_module(module_name):
    package_name = "flagscale"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name,
            FLAGSCALE_ROOT / "__init__.py",
            submodule_search_locations=[str(FLAGSCALE_ROOT)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)

    return importlib.import_module(module_name)

GridAlgo = _import_flagscale_module("flagscale.runner.auto_tuner.search.algorithm").GridAlgo
Searcher = _import_flagscale_module("flagscale.runner.auto_tuner.search.searcher").Searcher


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
    )

    algo = GridAlgo(STRATEGIES, config)

    assert _drain_labels(algo) == ["beta", "gamma", "alpha"]


def test_grid_algo_keeps_input_order_without_chip_aware_scoring(tmp_path):
    config = _make_config(
        tmp_path,
    )

    algo = GridAlgo(STRATEGIES, config)

    assert _drain_labels(algo) == ["alpha", "beta", "gamma"]
