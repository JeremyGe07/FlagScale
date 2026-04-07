import pytest

from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.prune.chip import prune_by_chip_profile
from flagscale.runner.auto_tuner.prune.pruner import Pruner
from flagscale.runner.auto_tuner.search.algorithm import GridAlgo
from flagscale.runner.auto_tuner.search.searcher import Searcher
from flagscale.runner.auto_tuner.tuner import AutoTuner


DEFAULT_MODEL = {
    "num_layers": 2,
    "global_batch_size": 2,
    "hidden_size": 2,
    "num_attention_heads": 2,
    "seq_length": 2,
    "padded_vocab_size": 16,
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
    "identity": {
        "name": "test_chip",
        "vendor": "test_vendor",
        "chip_class": "gpu",
    },
    "memory": {
        "total_memory_mb": 46000,
        "bandwidth_gbps": 864,
    },
    "compute": {
        "bf16_tflops": 100,
        "attention_tflops": 100,
    },
    "kernel_support": {
        "transformer_engine": True,
        "flash_attention": True,
        "fused_rmsnorm": True,
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
    {"label": "alpha", "chip_score": 0.2},
    {"label": "beta", "chip_score": 0.9},
    {"label": "gamma", "chip_score": 0.5},
]

TIE_BREAKER_STRATEGIES = [
    {"label": "alpha", "chip_score": 0.7, "memory_model": 10},
    {"label": "beta", "chip_score": 0.7, "memory_model": 20},
    {"label": "gamma", "chip_score": 0.4, "memory_model": 30},
]


def _make_config(
    tmp_path,
    *,
    nnodes=1,
    nproc_per_node=2,
    include_auto_tuner_runtime=True,
    include_auto_tuner_platform=True,
    space_overrides=None,
    algo_overrides=None,
    chip_profile=None,
):
    space = dict(DEFAULT_SPACE)
    if space_overrides:
        space.update(space_overrides)

    auto_tuner = {
        "algo": {"name": "grid", **(algo_overrides or {})},
        "space": space,
    }
    if include_auto_tuner_platform:
        auto_tuner["platform"] = {}
    if include_auto_tuner_runtime:
        auto_tuner.update(
            {
                "cards": nnodes * nproc_per_node,
                "nnodes": nnodes,
                "nproc_per_node": nproc_per_node,
            }
        )
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


def _write_chip_profile(tmp_path, profile=None):
    profile_path = tmp_path / "chip_profile.yaml"
    OmegaConf.save(config=OmegaConf.create(profile or _make_chip_profile()), f=profile_path)
    return str(profile_path)


def _drain_labels(algo):
    labels = []
    while True:
        strategy = algo.search()
        if strategy is None:
            return labels
        labels.append(strategy["label"])


def _make_prune_strategy(**overrides):
    strategy = {
        "data_parallel_size": 1,
        "use_distributed_optimizer": False,
        "tensor_model_parallel_size": 1,
        "sequence_parallel": False,
        "pipeline_model_parallel_size": 1,
        "num_layers_per_virtual_pipeline_stage": 0,
        "use_recompute": False,
        "recompute_method": "uniform",
        "recompute_granularity": "full",
        "recompute_num_layers": 1,
        "micro_batch_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
    }
    strategy.update(overrides)
    return strategy


def test_pruner_marks_strategy_pruned_by_chip_profile(tmp_path):
    config = _make_config(
        tmp_path,
        chip_profile=_make_chip_profile(disabled_dims={"context_parallel_size": [1]}),
    )
    strategy = _make_prune_strategy()

    pruned = Pruner(config).prune(strategy, [])

    assert pruned is True
    assert strategy["pruned"] is True
    assert strategy["performance"] is None
    assert strategy["pruned_reason"] == "chip_profile.disabled_dims"


def test_pruner_counts_chip_profile_prunes_separately(tmp_path):
    config = _make_config(
        tmp_path,
        chip_profile=_make_chip_profile(disabled_dims={"context_parallel_size": [1]}),
    )
    strategy = _make_prune_strategy()
    pruner = Pruner(config)

    pruner.prune(strategy, [])

    assert pruner.pruned_by_chip_profile == 1


def test_pruner_chip_pruned_history_item_has_safe_max_mem_shape(tmp_path):
    config = _make_config(
        tmp_path,
        chip_profile=_make_chip_profile(disabled_dims={"use_distributed_optimizer": [True]}),
    )
    pruner = Pruner(config)
    history = []
    chip_pruned = _make_prune_strategy(use_distributed_optimizer=True)
    followup = _make_prune_strategy(use_distributed_optimizer=False)

    assert pruner.prune(chip_pruned, history) is True
    assert chip_pruned["max_mem"] is None
    assert pruner.prune(followup, history) is False


def test_pruner_sets_reason_for_memory_model_prunes(tmp_path):
    config = _make_config(tmp_path, chip_profile=_make_chip_profile())
    config.experiment.auto_tuner.memory_model = {"gpu_memory": 80}
    strategy = _make_prune_strategy(memory_model=81)

    pruned = Pruner(config).prune(strategy, [])

    assert pruned is True
    assert strategy["pruned_reason"] == "memory_model.upper_bound"


def test_pruner_logs_memory_breakdown_details_for_memory_model_prunes(tmp_path, caplog):
    config = _make_config(tmp_path, chip_profile=_make_chip_profile())
    config.experiment.auto_tuner.memory_model = {"gpu_memory": 80}
    strategy = _make_prune_strategy(
        memory_model=81,
        memory_breakdown={"reserved_mb": 12.5, "peak_mb": 68.5},
    )

    with caplog.at_level("INFO", logger="FlagScale-AutoTuner"):
        pruned = Pruner(config).prune(strategy, [])

    assert pruned is True
    assert "reserved_mb=12.5" in caplog.text
    assert "peak_mb=68.5" in caplog.text


def test_prune_by_chip_profile_reports_topology_reason(tmp_path):
    config = _make_config(
        tmp_path,
        nnodes=2,
        nproc_per_node=1,
        chip_profile=_make_chip_profile(max_nodes=1, devices_per_node=1),
    )

    pruned, reason = prune_by_chip_profile(config, _make_prune_strategy())

    assert pruned is True
    assert reason == "chip_profile.topology"


def test_pruner_sets_specific_reason_for_history_prunes(tmp_path):
    config = _make_config(tmp_path, chip_profile=_make_chip_profile())
    pruner = Pruner(config)
    history = [_make_prune_strategy(micro_batch_size=2, performance=123.0, max_mem=None)]
    strategy = _make_prune_strategy(micro_batch_size=1)

    pruned = pruner.prune(strategy, history)

    assert pruned is True
    assert strategy["pruned_reason"] == "history.prune_by_micro_batch_size"


def test_searcher_derives_runtime_defaults_from_runner_for_direct_construction(tmp_path):
    config = _make_config(
        tmp_path,
        include_auto_tuner_runtime=False,
        include_auto_tuner_platform=False,
        algo_overrides={"chip_aware_scoring": True},
    )
    config.experiment.auto_tuner.chip_profile = {"path": _write_chip_profile(tmp_path)}

    searcher = Searcher(config)

    assert config.experiment.auto_tuner.nnodes == 1
    assert config.experiment.auto_tuner.nproc_per_node == 2
    assert config.experiment.auto_tuner.cards == 2
    assert OmegaConf.to_container(config.experiment.auto_tuner.platform, resolve=True) == {}
    assert "profile" in config.experiment.auto_tuner.chip_profile
    assert searcher.strategies
    assert any("chip_score" in strategy for strategy in searcher.strategies)


def test_autotuner_summary_log_reports_chip_prune_count(monkeypatch, tmp_path):
    class DummySearcher:
        def __init__(self, config):
            self.strategies = [{"label": "first", "memory_model": 10}]
            self.algo = GridAlgo(self.strategies, config)

        def search(self):
            return self.algo.search()

        def has_done(self):
            return self.algo.has_done()

    class DummyPruner:
        def __init__(self, config):
            self.pruned_count = 0
            self.pruned_by_memory_model = 2
            self.pruned_by_chip_profile = 1

        def prune(self, strategy, history):
            history.append(strategy)
            return False

    class DummyGenerator:
        def __init__(self, config):
            self.config = config

        def gen(self, strategy):
            return {"strategy": strategy}

    class DummyRecorder:
        def __init__(self, config):
            self.config = config

        def read(self):
            return []

    monkeypatch.setattr("flagscale.runner.auto_tuner.tuner.Searcher", DummySearcher)
    monkeypatch.setattr("flagscale.runner.auto_tuner.tuner.Pruner", DummyPruner)
    monkeypatch.setattr("flagscale.runner.auto_tuner.tuner.Generator", DummyGenerator)
    monkeypatch.setattr("flagscale.runner.auto_tuner.tuner.Recorder", DummyRecorder)
    monkeypatch.setattr(
        "flagscale.runner.auto_tuner.tuner.AutoTuner.find_pruned_num_value",
        lambda self, path: "3",
    )
    monkeypatch.setattr(
        "flagscale.runner.auto_tuner.tuner.AutoTuner.find_search_num_value",
        lambda self, path: "0",
    )

    config = _make_config(tmp_path, chip_profile=_make_chip_profile())
    config.experiment.auto_tuner.memory_model = {"gpu_memory": 80}

    tuner = AutoTuner(config)
    tuner.gen()

    log_text = (tmp_path / "auto_tuner" / "tuner.log").read_text(encoding="utf-8")
    assert "Pruned 3 strategy, 2 by memory model, 1 by chip profile." in log_text


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


def test_searcher_fails_fast_when_chip_limits_empty_a_space_dim(tmp_path):
    config = _make_config(
        tmp_path,
        chip_profile=_make_chip_profile(disabled_dims={"context_parallel_size": [1, 2]}),
    )

    with pytest.raises(ValueError, match="context_parallel_size"):
        Searcher(config)


def test_searcher_fails_fast_when_final_chip_filter_removes_all_strategies(tmp_path):
    config = _make_config(
        tmp_path,
        space_overrides={"use_distributed_optimizer": [True, False]},
        chip_profile=_make_chip_profile(disabled_dims={"use_distributed_optimizer": [False]}),
    )

    with pytest.raises(ValueError, match="zero strategies"):
        Searcher(config)


def test_searcher_rejects_invalid_disabled_dim_name(tmp_path):
    config = _make_config(
        tmp_path,
        chip_profile=_make_chip_profile(disabled_dims={"invalid_dim": [1]}),
    )

    with pytest.raises(ValueError, match="invalid_dim"):
        Searcher(config)


def test_searcher_without_chip_profile_allows_natural_zero_strategies(tmp_path):
    config = _make_config(
        tmp_path,
        space_overrides={"tensor_model_parallel_size": [3]},
    )

    searcher = Searcher(config)

    assert searcher.strategies == []


def test_searcher_with_chip_profile_allows_natural_zero_strategies(tmp_path):
    config = _make_config(
        tmp_path,
        space_overrides={"micro_batch_size": [3]},
        chip_profile=_make_chip_profile(),
    )

    searcher = Searcher(config)

    assert searcher.strategies == []


def test_searcher_injects_chip_score_fields_when_chip_aware_scoring_enabled(tmp_path):
    config = _make_config(
        tmp_path,
        chip_profile=_make_chip_profile(),
        algo_overrides={"chip_aware_scoring": True},
    )

    searcher = Searcher(config)

    assert searcher.strategies
    strategy = searcher.strategies[0]
    assert "chip_score" in strategy
    assert "chip_priority" in strategy
    assert "chip_score_reasons" in strategy
    assert strategy["chip_priority"] == "performance"
    assert strategy["chip_score_reasons"]


def test_searcher_rejects_chip_aware_scoring_without_chip_profile(tmp_path):
    config = _make_config(
        tmp_path,
        algo_overrides={"chip_aware_scoring": True},
    )

    with pytest.raises(ValueError, match="chip_aware_scoring.*chip_profile"):
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


def test_grid_algo_keeps_plain_memory_model_order_without_chip_scores(tmp_path):
    config = _make_config(
        tmp_path,
        algo_overrides={"chip_aware_scoring": True},
    )
    config.experiment.auto_tuner.memory_model = {"model_name": "default"}
    strategies = [
        {"label": "alpha", "memory_model": 10},
        {"label": "beta", "memory_model": 20},
        {"label": "gamma", "memory_model": 30},
    ]

    algo = GridAlgo(strategies, config)

    assert _drain_labels(algo) == ["alpha", "beta", "gamma"]


def test_grid_algo_uses_memory_model_as_tie_breaker_for_chip_score(tmp_path):
    config = _make_config(
        tmp_path,
        algo_overrides={"chip_aware_scoring": True},
    )
    config.experiment.auto_tuner.memory_model = {"model_name": "default"}

    algo = GridAlgo(TIE_BREAKER_STRATEGIES, config)

    assert _drain_labels(algo) == ["beta", "alpha", "gamma"]
