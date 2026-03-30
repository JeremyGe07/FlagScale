import pytest

from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.generate import Generator
from flagscale.runner.auto_tuner.record.recorder import Recorder
from flagscale.runner.auto_tuner.search.algorithm import GridAlgo
from flagscale.runner.auto_tuner.tuner import AutoTuner


def test_generator_clamps_lr_warmup_iters_for_short_auto_tune_runs(tmp_path):
    config = OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {},
                "auto_tuner": {
                    "args_mapping": {},
                    "control": {"train_iters": 3},
                },
            },
            "train": {
                "system": {
                    "logging": {},
                    "checkpoint": {"save_interval": 100},
                },
                "model": {
                    "train_iters": 1000,
                    "optimizer": {
                        "lr_scheduler": {
                            "lr": 1e-5,
                            "min_lr": 0,
                            "lr_warmup_iters": 100,
                        }
                    },
                },
            },
        }
    )

    task = Generator(config).gen({"idx": 1})

    assert task.train.model.train_iters == 3
    assert task.train.model.optimizer.lr_scheduler.lr_warmup_iters == 2


def test_generator_disables_validation_for_auto_tune_tasks_without_zeroing_eval_interval(
    tmp_path,
):
    config = OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {},
                "auto_tuner": {
                    "args_mapping": {},
                    "control": {"train_iters": 3},
                },
            },
            "train": {
                "system": {
                    "logging": {},
                    "checkpoint": {"save_interval": 100},
                },
                "model": {
                    "eval_iters": 10,
                    "eval_interval": 100,
                    "optimizer": {
                        "lr_scheduler": {
                            "lr": 1e-5,
                            "min_lr": 0,
                        }
                    },
                },
            },
        }
    )

    task = Generator(config).gen({"idx": 1})

    assert task.train.model.eval_iters == 0
    assert task.train.model.eval_interval == 100


def test_recorder_save_serializes_omegaconf_list_values(tmp_path):
    (tmp_path / "auto_tuner").mkdir()
    config = OmegaConf.create({"experiment": {"exp_dir": str(tmp_path)}})
    recorder = Recorder(config)

    recorder.save(
        [
            {
                "idx": 1,
                "performance": 123.4,
                "gpu_utilization": OmegaConf.create([0.1, 1.0]),
            }
        ]
    )

    history = (tmp_path / "auto_tuner" / "history.csv").read_text(encoding="utf-8")
    assert "[0.1, 1.0]" in history


class _DummySearcher:
    def __init__(self, config):
        self.strategies = [{"label": "first"}, {"label": "second"}]
        self.algo = GridAlgo(self.strategies, config)

    def search(self):
        return self.algo.search()

    def has_done(self):
        return self.algo.has_done()


class _DummyPruner:
    def __init__(self, config):
        self.pruned_count = 0
        self.pruned_by_memory_model = 0

    def prune(self, strategy, history):
        history.append(strategy)
        return False


class _DummyGenerator:
    def __init__(self, config):
        self.config = config

    def gen(self, strategy):
        return {"strategy": strategy}


class _DummyRecorder:
    def __init__(self, config):
        self.config = config

    def read(self):
        return []


def test_autotuner_fresh_run_starts_from_first_strategy(monkeypatch, tmp_path):
    monkeypatch.setattr("flagscale.runner.auto_tuner.tuner.Searcher", _DummySearcher)
    monkeypatch.setattr("flagscale.runner.auto_tuner.tuner.Pruner", _DummyPruner)
    monkeypatch.setattr("flagscale.runner.auto_tuner.tuner.Generator", _DummyGenerator)
    monkeypatch.setattr("flagscale.runner.auto_tuner.tuner.Recorder", _DummyRecorder)

    config = OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {"nnodes": 1, "nproc_per_node": 2},
            },
            "train": {
                "system": {},
            },
        }
    )

    tuner = AutoTuner(config)
    tuner.gen()

    assert tuner.cur_strategy["label"] == "first"


def test_load_chip_profile_normalizes_a_minimal_profile(tmp_path):
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
  disabled_dims:
    context_parallel_size: [2, 4]
    expert_model_parallel_size: [2, 4]
""".strip(),
        encoding="utf-8",
    )

    profile = load_chip_profile(str(profile_path))

    assert profile["identity"]["name"] == "nvidia_l20"
    assert profile["memory"]["total_memory_mb"] == 46000
    assert profile["strategy_hints"]["max_tensor_model_parallel_size"] == 2


def test_load_chip_profile_rejects_missing_required_sections(tmp_path):
    from flagscale.runner.auto_tuner.chip_profile import load_chip_profile

    profile_path = tmp_path / "broken.yaml"
    profile_path.write_text(
        """
schema_version: v1alpha1
identity:
  name: broken
  vendor: nvidia
  chip_class: gpu
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required chip profile sections"):
        load_chip_profile(str(profile_path))


def test_autotuner_loads_chip_profile_into_runtime_config(monkeypatch, tmp_path):
    from flagscale.runner.auto_tuner.chip_profile import load_chip_profile

    monkeypatch.setattr("flagscale.runner.auto_tuner.tuner.Searcher", _DummySearcher)
    monkeypatch.setattr("flagscale.runner.auto_tuner.tuner.Pruner", _DummyPruner)
    monkeypatch.setattr("flagscale.runner.auto_tuner.tuner.Generator", _DummyGenerator)
    monkeypatch.setattr("flagscale.runner.auto_tuner.tuner.Recorder", _DummyRecorder)

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
  disabled_dims:
    context_parallel_size: [2, 4]
    expert_model_parallel_size: [2, 4]
""".strip(),
        encoding="utf-8",
    )

    config = OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {"nnodes": 1, "nproc_per_node": 2},
                "auto_tuner": {"chip_profile": {"path": str(profile_path)}},
            },
            "train": {"system": {}},
        }
    )

    tuner = AutoTuner(config)
    expected = load_chip_profile(str(profile_path))

    assert tuner.config.experiment.auto_tuner.chip_profile.profile == expected


def test_autotuner_rejects_hetero_mode_with_chip_aware_config(monkeypatch, tmp_path):
    class DummyHeteroSearcher:
        def __init__(self, config, resources):
            self.config = config
            self.resources = resources
            self.algo = GridAlgo([], config)

    class DummyHeteroPruner:
        def __init__(self, config):
            self.pruned_count = 0

    class DummyHeteroGenerator:
        def __init__(self, config):
            self.config = config

    class DummyHeteroRecorder:
        def __init__(self, config):
            self.config = config

        def read(self):
            return []

    monkeypatch.setattr("flagscale.runner.auto_tuner.tuner.HeteroSearcher", DummyHeteroSearcher)
    monkeypatch.setattr("flagscale.runner.auto_tuner.tuner.HeteroPruner", DummyHeteroPruner)
    monkeypatch.setattr("flagscale.runner.auto_tuner.tuner.HeteroGenerator", DummyHeteroGenerator)
    monkeypatch.setattr("flagscale.runner.auto_tuner.tuner.HeteroRecorder", DummyHeteroRecorder)
    monkeypatch.setattr(
        "flagscale.runner.auto_tuner.tuner.parse_hostfile",
        lambda path: {"localhost": {"slots": 2}},
    )

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

    config = OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {
                    "nnodes": 1,
                    "nproc_per_node": 2,
                    "hostfile": str(tmp_path / "hosts"),
                },
                "auto_tuner": {
                    "chip_profile": {"path": str(profile_path)},
                    "algo": {"chip_aware_scoring": True},
                },
            },
            "train": {
                "system": {
                    "hetero": {"enable_hetero": True},
                },
            },
        }
    )

    with pytest.raises(ValueError, match="heterogeneous.*chip-aware|chip_profile"):
        AutoTuner(config)
