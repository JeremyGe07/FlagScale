from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

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
