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
