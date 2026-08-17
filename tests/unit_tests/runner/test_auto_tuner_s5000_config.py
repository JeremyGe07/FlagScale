from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.search.chip_strategy_score import build_chip_score
from flagscale.runner.auto_tuner.search.pp_first_searcher import PPFirstSearcher
from flagscale.runner.auto_tuner.search.searcher import Searcher


RUNTIME_FIELDS = (
    "data_parallel_size",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "micro_batch_size",
    "acc_step",
    "sequence_parallel",
    "use_distributed_optimizer",
    "use_recompute",
)


def _config_dir():
    return str(Path(__file__).resolve().parents[3] / "examples" / "qwen3" / "conf")


def _runtime_signature(strategy):
    return tuple(strategy.get(field) for field in RUNTIME_FIELDS)


def test_s5000_pp_first_uses_same_candidate_universe_as_baseline():
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        baseline_config = compose(config_name="train_auto_tuner_s5000_tiny_baseline")
        method_config = compose(config_name="train_auto_tuner_s5000_tiny_pp_first")

    for config in (baseline_config, method_config):
        OmegaConf.set_struct(config, False)

    baseline = Searcher(baseline_config)
    method = PPFirstSearcher(method_config)
    baseline_universe = {_runtime_signature(item) for item in baseline.strategies}
    method_universe = {_runtime_signature(item) for item in method.strategies}

    assert baseline_config.train.model.global_batch_size == 2
    assert method_config.train.model.global_batch_size == 2
    assert baseline_universe == method_universe
    assert len(baseline_universe) == 3
    assert len(method.short_run_strategies) == 1
    assert _runtime_signature(method.short_run_strategies[0]) == (
        2,
        1,
        1,
        1,
        1,
        False,
        False,
        False,
    )


def test_s5000_configs_keep_data_and_tokenizer_paths_portable():
    config_path = Path(_config_dir()) / "train" / "s5000_tiny.yaml"
    config_text = config_path.read_text(encoding="utf-8")

    assert "${oc.env:S5000_DATA_PREFIX}" in config_text
    assert "${oc.env:S5000_TOKENIZER_PATH}" in config_text
    assert "/home/" not in config_text


def test_s5000_chip_time_pruning_uses_same_8gpu_candidate_universe_as_baseline(
    monkeypatch,
):
    monkeypatch.setenv("S5000_DATA_PREFIX", "/tmp/s5000-data")
    monkeypatch.setenv("S5000_TOKENIZER_PATH", "/tmp/s5000-tokenizer")

    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        baseline_config = compose(
            config_name="train_auto_tuner_s5000_8gpu_chip_time_baseline"
        )
        method_config = compose(
            config_name="train_auto_tuner_s5000_8gpu_chip_time_prune"
        )

    for config in (baseline_config, method_config):
        OmegaConf.set_struct(config, False)

    baseline = Searcher(baseline_config)
    method = Searcher(method_config)
    baseline_universe = {_runtime_signature(item) for item in baseline.strategies}
    method_universe = {_runtime_signature(item) for item in method.strategies}
    time_cost_pruned = [
        strategy for strategy in method.strategies if strategy.get("time_cost_pruned")
    ]

    assert baseline_universe == method_universe
    assert len(baseline_universe) == 12
    assert len(time_cost_pruned) == 4
    assert baseline_config.experiment.auto_tuner.memory_model.gpu_utilization == [
        0.0,
        1.0,
    ]
    assert all("chip_score" in strategy for strategy in method.strategies)
    assert all("time_cost" in strategy for strategy in method.strategies)
    assert all(
        strategy["memory_model_peak_activation_bias"] == 0
        for strategy in method.strategies
    )
    assert (
        method_config.experiment.auto_tuner.chip_profile.profile.cost_model
        .reserved_memory_bias_mb
        == 0
    )


def test_mtlink_is_treated_as_a_high_bandwidth_chip_aware_fabric():
    strategy = {
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 2,
    }
    profile = {
        "strategy_hints": {"default_search_priority": "performance"},
        "topology": {"devices_per_node": 8},
        "interconnect": {"intra_node": {"fabric": "mtlink"}},
        "compute": {"bf16_tflops": 340, "attention_tflops": 340},
    }
    pcie_profile = OmegaConf.to_container(OmegaConf.create(profile), resolve=True)
    pcie_profile["interconnect"]["intra_node"]["fabric"] = "pcie"

    assert build_chip_score(strategy, profile)["score"] > build_chip_score(
        strategy, pcie_profile
    )["score"]


def test_s5000_qwen3_14b_baseline_uses_wide_8gpu_space(monkeypatch):
    monkeypatch.setenv("PILE_WIKIPEDIA_DEMO_PATH", "/tmp/pile-wikipedia-demo")
    monkeypatch.setenv("QWEN3_14B_PATH", "/tmp/qwen3-14b")

    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        config = compose(config_name="train_auto_tuner_14b_8xs5000_baseline")

    OmegaConf.set_struct(config, False)
    searcher = Searcher(config)
    parallel_families = {
        (
            strategy["data_parallel_size"],
            strategy["tensor_model_parallel_size"],
            strategy["pipeline_model_parallel_size"],
            strategy["context_parallel_size"],
        )
        for strategy in searcher.strategies
    }

    assert config.experiment.runner.nproc_per_node == 8
    assert config.train.model.num_layers == 40
    assert config.train.model.seq_length == 4096
    assert config.train.model.global_batch_size == 2048
    assert config.train.model.transformer_impl == "local"
    assert config.experiment.auto_tuner.space.data_parallel_size == "auto"
    assert config.experiment.auto_tuner.space.tensor_model_parallel_size == "auto"
    assert config.experiment.auto_tuner.space.pipeline_model_parallel_size == "auto"
    assert config.experiment.auto_tuner.space.context_parallel_size == [1]
    assert config.experiment.auto_tuner.space.micro_batch_size == "auto"
    assert config.experiment.auto_tuner.space.use_recompute == "auto"
    assert config.experiment.auto_tuner.space.sequence_parallel == [False]
    assert config.experiment.auto_tuner.space.num_layers_per_virtual_pipeline_stage == [0]
    assert config.experiment.envs.PYTORCH_MUSA_ALLOC_CONF == "expandable_segments:True"
    assert any(tp == 8 for _dp, tp, _pp, _cp in parallel_families)
    assert (1, 8, 1, 1) in parallel_families


def test_s5000_qwen3_14b_profiled_time_prune_keeps_baseline_universe_and_best_family(
    monkeypatch,
):
    monkeypatch.setenv("PILE_WIKIPEDIA_DEMO_PATH", "/tmp/pile-wikipedia-demo")
    monkeypatch.setenv("QWEN3_14B_PATH", "/tmp/qwen3-14b")

    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        baseline_config = compose(
            config_name="train_auto_tuner_14b_8xs5000_baseline"
        )
        method_config = compose(
            config_name="train_auto_tuner_14b_8xs5000_profiled_time_prune"
        )

    for config in (baseline_config, method_config):
        OmegaConf.set_struct(config, False)

    baseline = Searcher(baseline_config)
    method = Searcher(method_config)
    baseline_universe = {_runtime_signature(item) for item in baseline.strategies}
    method_universe = {_runtime_signature(item) for item in method.strategies}

    assert baseline_universe == method_universe
    assert len(method.strategies) == 4876
    assert sum(
        bool(strategy.get("time_cost_pruned")) for strategy in method.strategies
    ) == 4844
    assert (
        method_config.experiment.auto_tuner.algo.time_cost_pruning.per_family_topk
        == 2
    )
    assert (
        method_config.experiment.auto_tuner.chip_profile.profile.cost_model
        .reserved_memory_bias_mb
        == 0
    )
    assert (
        method_config.experiment.auto_tuner.chip_profile.profile.cost_model
        .peak_activation_bias_mb
        == 0
    )

    # The clean S5000 baseline best (task_89) is DP2/TP4/PP1, distributed
    # optimizer, MBS2, block/full recompute of one layer. It must remain in the
    # conservative TopK=2 shortlist before launching the expensive comparison.
    best_matches = [
        strategy
        for strategy in method.strategies
        if strategy["data_parallel_size"] == 2
        and strategy["tensor_model_parallel_size"] == 4
        and strategy["pipeline_model_parallel_size"] == 1
        and strategy["use_distributed_optimizer"] is True
        and strategy["micro_batch_size"] == 2
        and strategy["use_recompute"] is True
        and strategy["recompute_method"] == "block"
        and strategy["recompute_granularity"] == "full"
        and strategy["recompute_num_layers"] == 1
    ]
    assert len(best_matches) == 1
    assert best_matches[0]["time_cost_family_rank"] == 2
    assert not best_matches[0].get("time_cost_pruned", False)
