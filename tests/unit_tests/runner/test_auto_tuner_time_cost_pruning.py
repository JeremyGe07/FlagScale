from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.prune.pruner import Pruner
from flagscale.runner.auto_tuner.search.algorithm import GridAlgo
from flagscale.runner.auto_tuner.search.time_cost_pruning import (
    mark_time_cost_pruned_strategies,
)
from flagscale.runner.auto_tuner.tuner import AutoTuner


def _config(
    *,
    enabled=True,
    per_family_topk=2,
    gpu_memory=100.0,
    family_replenish_on_oom=False,
):
    pruning = {
        "enabled": enabled,
        "mode": "family_topk",
    }
    if per_family_topk is not None:
        pruning["per_family_topk"] = per_family_topk
    if family_replenish_on_oom:
        pruning["family_replenish_on_oom"] = True
    return OmegaConf.create(
        {
            "experiment": {
                "auto_tuner": {
                    "algo": {
                        "name": "grid",
                        "use_profiled_time_cost": True,
                        "time_cost_pruning": pruning,
                    },
                    "memory_model": {"gpu_memory": gpu_memory},
                }
            }
        }
    )


def _strategy(name, *, time_cost, dp=1, tp=2, pp=2, memory_model=80.0):
    return {
        "name": name,
        "time_cost": time_cost,
        "memory_model": memory_model,
        "gpu_utilization": [0.0, 1.0],
        "data_parallel_size": dp,
        "tensor_model_parallel_size": tp,
        "pipeline_model_parallel_size": pp,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "sequence_parallel": tp > 1,
        "use_distributed_optimizer": dp > 1,
    }


def test_time_cost_pruning_marks_only_extra_candidates_inside_each_family():
    strategies = [
        _strategy("family-a-fast", time_cost=1.0, dp=1, tp=2, pp=2),
        _strategy("family-a-second", time_cost=2.0, dp=1, tp=2, pp=2),
        _strategy("family-a-extra", time_cost=3.0, dp=1, tp=2, pp=2),
        _strategy("family-b-recall", time_cost=99.0, dp=2, tp=1, pp=2),
    ]

    mark_time_cost_pruned_strategies(strategies, _config(per_family_topk=2))

    pruned = [strategy["name"] for strategy in strategies if strategy.get("time_cost_pruned")]
    kept = [strategy["name"] for strategy in strategies if not strategy.get("time_cost_pruned")]

    assert pruned == ["family-a-extra"]
    assert kept == ["family-a-fast", "family-a-second", "family-b-recall"]
    assert strategies[2]["time_cost_prune_reason"] == "time_cost.family_topk"


def test_time_cost_pruning_prefers_memory_feasible_family_representatives():
    strategies = [
        _strategy("fast-but-oom", time_cost=1.0, memory_model=1000.0),
        _strategy("slower-fit", time_cost=2.0, memory_model=80.0),
        _strategy("slow-fit-extra", time_cost=3.0, memory_model=70.0),
    ]

    mark_time_cost_pruned_strategies(strategies, _config(per_family_topk=1))

    pruned = [strategy["name"] for strategy in strategies if strategy.get("time_cost_pruned")]
    kept = [strategy["name"] for strategy in strategies if not strategy.get("time_cost_pruned")]

    assert kept == ["slower-fit"]
    assert pruned == ["fast-but-oom", "slow-fit-extra"]


def test_default_family_topk_uses_current_balanced_a800_and_l20_default():
    default_topk_boundary = 32
    first_pruned_rank = default_topk_boundary + 1
    strategies = [
        _strategy(f"faster-{idx}", time_cost=float(idx))
        for idx in range(1, default_topk_boundary)
    ]
    strategies.append(
        _strategy("rank-32-kept", time_cost=float(default_topk_boundary))
    )
    strategies.append(_strategy("rank-33-extra", time_cost=float(first_pruned_rank)))

    mark_time_cost_pruned_strategies(strategies, _config(per_family_topk=None))

    assert strategies[default_topk_boundary - 1]["name"] == "rank-32-kept"
    assert not strategies[default_topk_boundary - 1].get("time_cost_pruned")
    assert strategies[first_pruned_rank - 1]["time_cost_pruned"] is True


def test_pruner_skips_strategy_marked_by_time_cost_pruning():
    strategy = _strategy("extra", time_cost=3.0)
    strategy["time_cost_pruned"] = True
    strategy["time_cost_prune_reason"] = "time_cost.family_topk"
    pruner = Pruner(_config())

    pruned = pruner.prune(strategy, [])

    assert pruned is True
    assert pruner.pruned_by_time_cost == 1
    assert strategy["pruned"] is True
    assert strategy["performance"] is None
    assert strategy["pruned_reason"] == "time_cost.family_topk"


def test_pruner_replenishes_next_family_candidate_after_topk_failures(monkeypatch):
    monkeypatch.setattr("flagscale.runner.auto_tuner.prune.pruner._HISTORY_BASED_PRUNE_FUNC", [])
    strategies = [
        _strategy("rank-1-oom", time_cost=1.0),
        _strategy("rank-2-error", time_cost=2.0),
        _strategy("rank-3-replenish", time_cost=3.0),
    ]
    config = _config(per_family_topk=2, family_replenish_on_oom=True)
    mark_time_cost_pruned_strategies(strategies, config)
    strategies[0]["performance"] = None
    strategies[0]["max_mem"] = "OOM"
    strategies[1]["performance"] = None
    strategies[1]["max_mem"] = None
    strategies[1]["error"] = "IndexError"
    pruner = Pruner(config)

    pruned = pruner.prune(strategies[2], strategies[:2])

    assert pruned is False
    assert strategies[2]["time_cost_replenished"] is True
    assert pruner.pruned_by_time_cost == 0


def test_pruner_does_not_replenish_when_a_topk_candidate_has_perf(monkeypatch):
    monkeypatch.setattr("flagscale.runner.auto_tuner.prune.pruner._HISTORY_BASED_PRUNE_FUNC", [])
    strategies = [
        _strategy("rank-1-oom", time_cost=1.0),
        _strategy("rank-2-success", time_cost=2.0),
        _strategy("rank-3-pruned", time_cost=3.0),
    ]
    config = _config(per_family_topk=2, family_replenish_on_oom=True)
    mark_time_cost_pruned_strategies(strategies, config)
    strategies[0]["performance"] = None
    strategies[0]["max_mem"] = "OOM"
    strategies[1]["performance"] = 123.4
    strategies[1]["max_mem"] = 456.0
    pruner = Pruner(config)

    pruned = pruner.prune(strategies[2], strategies[:2])

    assert pruned is True
    assert "time_cost_replenished" not in strategies[2]
    assert pruner.pruned_by_time_cost == 1


def test_autotuner_summary_log_reports_time_cost_prune_count(monkeypatch, tmp_path):
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
            self.pruned_count = 3
            self.pruned_by_memory_model = 1
            self.pruned_by_chip_profile = 0
            self.pruned_by_time_cost = 2

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
        lambda self, path: "0",
    )
    monkeypatch.setattr(
        "flagscale.runner.auto_tuner.tuner.AutoTuner.find_search_num_value",
        lambda self, path: "0",
    )
    config = OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {"nnodes": 1, "nproc_per_node": 1},
                "auto_tuner": {
                    "algo": {"name": "grid"},
                    "memory_model": {"gpu_memory": 80},
                },
            },
            "train": {"system": {}, "model": {}},
        }
    )

    tuner = AutoTuner(config)
    tuner.gen()

    log_text = (tmp_path / "auto_tuner" / "tuner.log").read_text(encoding="utf-8")
    assert "1 by memory model, 0 by chip profile, 2 by time cost." in log_text
