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
                "runner": {"nnodes": 1, "nproc_per_node": 1},
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
                    "num_layers": 1,
                    "global_batch_size": 1,
                    "hidden_size": 8,
                    "num_attention_heads": 1,
                    "seq_length": 8,
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
                "runner": {"nnodes": 1, "nproc_per_node": 1},
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
                    "num_layers": 1,
                    "global_batch_size": 1,
                    "hidden_size": 8,
                    "num_attention_heads": 1,
                    "seq_length": 8,
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


def test_generator_injects_plan_runtime_metadata_into_task_config(tmp_path):
    config = OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {"nnodes": 1, "nproc_per_node": 1},
                "auto_tuner": {
                    "control": {"train_iters": 3},
                },
            },
            "train": {
                "system": {
                    "logging": {},
                    "checkpoint": {"save_interval": 100},
                },
                "model": {
                    "num_layers": 8,
                    "global_batch_size": 8,
                    "hidden_size": 8,
                    "num_attention_heads": 4,
                    "seq_length": 16,
                    "eval_iters": 10,
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

    task = Generator(config).gen(
        {
            "idx": 1,
            "data_parallel_size": 1,
            "use_distributed_optimizer": False,
            "tensor_model_parallel_size": 1,
            "sequence_parallel": False,
            "pipeline_model_parallel_size": 1,
            "num_layers_per_virtual_pipeline_stage": None,
            "recompute_method": None,
            "recompute_granularity": None,
            "recompute_num_layers": None,
            "micro_batch_size": 2,
            "acc_step": 4,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "decoder_first_pipeline_num_layers": None,
            "decoder_last_pipeline_num_layers": None,
            "acc_step": 4,
        }
    )

    assert task.experiment.auto_tuner.plan.plan_kind == "homogeneous"
    assert task.experiment.auto_tuner.plan.stage_count == 1
    assert task.experiment.auto_tuner.plan.segment_count == 1
    assert task.experiment.auto_tuner.plan.runtime_executable is True
    assert task.experiment.auto_tuner.plan.execution_contract.global_batch_size == 8


def test_generator_merges_plan_metadata_without_clobbering_existing_plan_fields(tmp_path):
    config = OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {"nnodes": 1, "nproc_per_node": 1},
                "auto_tuner": {
                    "control": {"train_iters": 3},
                    "plan": {"template_name": "dense-8"},
                },
            },
            "train": {
                "system": {
                    "logging": {},
                    "checkpoint": {"save_interval": 100},
                },
                "model": {
                    "num_layers": 8,
                    "global_batch_size": 8,
                    "hidden_size": 8,
                    "num_attention_heads": 4,
                    "seq_length": 16,
                    "eval_iters": 10,
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

    task = Generator(config).gen(
        {
            "idx": 1,
            "data_parallel_size": 1,
            "use_distributed_optimizer": False,
            "tensor_model_parallel_size": 1,
            "sequence_parallel": False,
            "pipeline_model_parallel_size": 1,
            "num_layers_per_virtual_pipeline_stage": None,
            "recompute_method": None,
            "recompute_granularity": None,
            "recompute_num_layers": None,
            "micro_batch_size": 2,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "decoder_first_pipeline_num_layers": None,
            "decoder_last_pipeline_num_layers": None,
            "acc_step": 4,
        }
    )

    assert task.experiment.auto_tuner.plan.template_name == "dense-8"
    assert task.experiment.auto_tuner.plan.plan_kind == "homogeneous"


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


def test_recorder_save_serializes_plan_runtime_fields(tmp_path):
    (tmp_path / "auto_tuner").mkdir()
    config = OmegaConf.create({"experiment": {"exp_dir": str(tmp_path)}})
    recorder = Recorder(config)

    recorder.save(
        [
            {
                "idx": 1,
                "performance": 123.4,
                "plan_kind": "homogeneous",
                "stage_count": 1,
                "segment_count": 1,
                "runtime_executable": True,
                "execution_contract": {
                    "world_size": 2,
                    "micro_batch_size": 2,
                    "gradient_accumulation_steps": 4,
                    "global_batch_size": 8,
                },
            }
        ]
    )

    history = recorder.read()

    assert history[0]["plan_kind"] == "homogeneous"
    assert history[0]["stage_count"] == 1
    assert history[0]["segment_count"] == 1
    assert history[0]["runtime_executable"] is True
    assert history[0]["execution_contract"]["global_batch_size"] == 8


def test_recorder_save_serializes_pp_first_planner_metadata(tmp_path):
    (tmp_path / "auto_tuner").mkdir()
    config = OmegaConf.create({"experiment": {"exp_dir": str(tmp_path)}})
    recorder = Recorder(config)

    recorder.save(
        [
            {
                "idx": 1,
                "performance": 123.4,
                "planner_name": "pp_first",
                "partition_policy": "layer_count_balanced",
                "topology_signature": "contiguous-equal:2x2",
                "provenance": "heuristic",
                "planner_budget": {
                    "max_pp_candidates": 4,
                    "max_partitions_per_pp": 1,
                    "max_assignments_per_partition": 256,
                    "topk_plans_for_short_run": 16,
                },
                "estimated_stage_costs": [
                    {"stage_id": 0, "time_ms": 12.3, "memory_mb": 1024.0},
                    {"stage_id": 1, "time_ms": 14.5, "memory_mb": 1152.0},
                ],
                "estimate_metric": "time_cost",
                "estimate_rank": 3,
                "estimate_value": 26.8,
                "short_run_candidate": True,
                "short_run_shortlist_count": 16,
            }
        ]
    )

    history = recorder.read()

    assert history[0]["planner_name"] == "pp_first"
    assert history[0]["partition_policy"] == "layer_count_balanced"
    assert history[0]["planner_budget"]["max_pp_candidates"] == 4
    assert history[0]["estimated_stage_costs"][1]["memory_mb"] == 1152.0
    assert history[0]["short_run_candidate"] is True


def test_recorder_record_tolerates_missing_autotuner_platform(monkeypatch, tmp_path):
    config = OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "auto_tuner": {},
            }
        }
    )
    recorder = Recorder(config)
    strategy = {"idx": 1}
    task = OmegaConf.create({"experiment": {"exp_dir": str(tmp_path)}})

    monkeypatch.setattr(
        recorder,
        "get_all_performance_and_host_paths",
        lambda current_task: (["perf.log"], "host_logs"),
    )
    monkeypatch.setattr(recorder, "grep_error", lambda path: set())
    monkeypatch.setattr(recorder, "grep_max_memory", lambda path: 12.3)
    monkeypatch.setattr(recorder, "grep_performance", lambda paths, pattern: 45.6)
    monkeypatch.setattr(
        recorder,
        "pass_back_to_platform",
        lambda current_strategy: pytest.fail("platform callback should not run"),
    )

    recorder.record(task, strategy)

    assert strategy["performance"] == 45.6
    assert strategy["max_mem"] == 12.3
    assert strategy["error"] is None


def test_recorder_discards_partial_performance_when_task_has_error(monkeypatch, tmp_path):
    config = OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "auto_tuner": {},
            }
        }
    )
    recorder = Recorder(config)
    strategy = {"idx": 7, "stopped_by_tuner": True}
    task = OmegaConf.create({"experiment": {"exp_dir": str(tmp_path)}})

    monkeypatch.setattr(
        recorder,
        "get_all_performance_and_host_paths",
        lambda current_task: (["perf.log"], "host_logs"),
    )
    monkeypatch.setattr(recorder, "grep_error", lambda path: {"RuntimeError: NaN"})
    monkeypatch.setattr(recorder, "grep_max_memory", lambda path: 67.0)
    monkeypatch.setattr(
        recorder,
        "grep_performance",
        lambda paths, pattern: pytest.fail("failed tasks must not retain partial timing"),
    )

    recorder.record(task, strategy)

    assert strategy["performance"] is None
    assert strategy["max_mem"] == 67.0
    assert strategy["error"] == "RuntimeError: NaN"


@pytest.mark.parametrize(
    "metric",
    [
        "grad norm: nan",
        "params norm: +inf",
        "lm loss: -inf",
        "number of nan iterations: 1",
    ],
)
def test_recorder_detects_non_finite_training_metrics(tmp_path, metric):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "host_0_localhost.output").write_text(
        f"iteration 1/1 | {metric} |\n",
        encoding="utf-8",
    )
    config = OmegaConf.create({"experiment": {"exp_dir": str(tmp_path)}})
    recorder = Recorder(config)
    recorder.cur_strategy = {"idx": 1}

    errors = recorder.grep_error(logs)

    assert "NUMERICAL_ERROR" in errors


def test_recorder_sort_and_get_best_ignore_errorful_legacy_performance(tmp_path):
    config = OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "auto_tuner": {},
            }
        }
    )
    recorder = Recorder(config)
    history = [
        {"idx": 1, "performance": 1.0, "error": "ChildFailedError"},
        {"idx": 2, "performance": "2.0", "error": None},
        {"idx": 3, "performance": None, "error": None},
        {"idx": 4, "performance": float("nan"), "error": None},
    ]

    sorted_history = recorder.sort(history)
    tuner = AutoTuner.__new__(AutoTuner)
    tuner.recorder = recorder
    tuner.history = history

    assert [strategy["idx"] for strategy in sorted_history] == [2, 1, 3, 4]
    assert tuner.get_best()["idx"] == 2

    tuner.history = [history[0]]
    assert tuner.get_best() is None


def test_recorder_read_invalidates_errorful_legacy_performance(tmp_path):
    auto_tuner_dir = tmp_path / "auto_tuner"
    auto_tuner_dir.mkdir()
    (auto_tuner_dir / "history.csv").write_text(
        "idx,performance,error\n"
        "1,1.0,ChildFailedError\n"
        "2,2.0,\n"
        "3,nan,\n",
        encoding="utf-8",
    )
    config = OmegaConf.create({"experiment": {"exp_dir": str(tmp_path)}})

    history = Recorder(config).read()

    assert history[0]["performance"] is None
    assert history[1]["performance"] == 2.0
    assert history[2]["performance"] is None


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


def test_autotuner_gen_propagates_plan_metadata_into_generated_task(tmp_path):
    config = OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {"nnodes": 1, "nproc_per_node": 2},
                "auto_tuner": {
                    "algo": {"name": "grid"},
                    "space": {
                        "data_parallel_size": [2],
                        "use_distributed_optimizer": [False],
                        "tensor_model_parallel_size": [1],
                        "sequence_parallel": [False],
                        "pipeline_model_parallel_size": [1],
                        "num_layers_per_virtual_pipeline_stage": [0],
                        "use_recompute": [False],
                        "recompute_method": ["uniform"],
                        "recompute_granularity": ["full"],
                        "recompute_num_layers": [1],
                        "micro_batch_size": [2],
                        "context_parallel_size": [1],
                        "expert_model_parallel_size": [1],
                    },
                },
            },
            "train": {
                "system": {
                    "logging": {},
                    "checkpoint": {"save_interval": 100},
                },
                "model": {
                    "num_layers": 8,
                    "global_batch_size": 16,
                    "hidden_size": 8,
                    "num_attention_heads": 4,
                    "seq_length": 16,
                    "eval_iters": 10,
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

    tuner = AutoTuner(config)
    tuner.gen()

    assert tuner.cur_strategy["plan_kind"] == "homogeneous"
    assert tuner.cur_strategy["runtime_executable"] is True
    assert tuner.cur_task.experiment.auto_tuner.plan.plan_kind == "homogeneous"
    assert tuner.cur_task.experiment.auto_tuner.plan.stage_count == 1
    assert tuner.cur_task.experiment.auto_tuner.plan.segment_count == 1
    assert tuner.cur_task.experiment.auto_tuner.plan.execution_contract.global_batch_size == 16


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
