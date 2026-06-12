import pytest

from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.search.pp_first_selection import (
    build_short_run_shortlist,
    select_assignment_candidates,
)
from tests.unit_tests.runner.test_auto_tuner_cost_model import _build_chip_profile


def _config(gpu_memory=None):
    auto_tuner = {
        "algo": {"name": "grid", "use_profiled_time_cost": True},
        "planner": {"topk_plans_for_short_run": 2},
    }
    if gpu_memory is not None:
        auto_tuner["memory_model"] = {"gpu_memory": gpu_memory}
    return OmegaConf.create({"experiment": {"auto_tuner": auto_tuner}})


def _memory_config(tmp_path):
    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {"nnodes": 1, "nproc_per_node": 2},
                "auto_tuner": {
                    "algo": {"name": "grid"},
                    "planner": {"topk_plans_for_short_run": 1},
                    "memory_model": {"model_name": "default", "gpu_memory": 46000},
                    "chip_profile": {
                        "profile": _build_chip_profile(peak_activation_bias_mb=256)
                    },
                    "cards": 2,
                    "nnodes": 1,
                    "nproc_per_node": 2,
                },
            },
            "train": {
                "system": {},
                "model": {
                    "num_layers": 2,
                    "hidden_size": 64,
                    "num_attention_heads": 8,
                    "global_batch_size": 16,
                    "seq_length": 32,
                    "padded_vocab_size": 256,
                },
            },
        }
    )


def _strategy(name, *, time_cost, dp, tp, pp, recompute_num_layers, memory_model=1.0):
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
        "use_distributed_optimizer": dp > 1,
        "sequence_parallel": tp > 1,
        "acc_step": 16,
        "micro_batch_size": 1,
        "num_layers_per_virtual_pipeline_stage": None,
        "use_recompute": True,
        "recompute_method": "block",
        "recompute_granularity": "full",
        "recompute_num_layers": recompute_num_layers,
        "decoder_first_pipeline_num_layers": None,
        "decoder_last_pipeline_num_layers": None,
        "partition_policy": "layer_count_balanced",
        "runtime_executable": True,
    }


def test_select_assignment_candidates_keeps_parallel_family_recall_within_budget():
    config = _config()
    strategies = [
        _strategy("dp2-tp1-fast", time_cost=1.0, dp=2, tp=1, pp=2, recompute_num_layers=1),
        _strategy("dp2-tp1-next", time_cost=2.0, dp=2, tp=1, pp=2, recompute_num_layers=2),
        _strategy("dp1-tp2-recall", time_cost=99.0, dp=1, tp=2, pp=2, recompute_num_layers=3),
    ]

    selected = select_assignment_candidates(strategies, max_assignments=2, config=config)

    assert [strategy["name"] for strategy in selected] == [
        "dp2-tp1-fast",
        "dp1-tp2-recall",
    ]


def test_build_short_run_shortlist_adds_parallel_family_recall():
    config = _config()
    strategies = [
        _strategy("dp2-tp1-fast", time_cost=1.0, dp=2, tp=1, pp=2, recompute_num_layers=1),
        _strategy("dp2-tp1-next", time_cost=2.0, dp=2, tp=1, pp=2, recompute_num_layers=2),
        _strategy("dp1-tp2-recall", time_cost=99.0, dp=1, tp=2, pp=2, recompute_num_layers=3),
    ]

    shortlist = build_short_run_shortlist(strategies, config)

    assert [strategy["name"] for strategy in shortlist] == [
        "dp2-tp1-fast",
        "dp2-tp1-next",
        "dp1-tp2-recall",
    ]


def test_select_assignment_candidates_prefers_memory_viable_family_representative():
    config = _config(gpu_memory=100.0)
    strategies = [
        _strategy("dp2-tp1-fast", time_cost=1.0, dp=2, tp=1, pp=2, recompute_num_layers=1),
        _strategy("dp2-tp1-next", time_cost=2.0, dp=2, tp=1, pp=2, recompute_num_layers=2),
        _strategy(
            "dp1-tp2-oom",
            time_cost=3.0,
            dp=1,
            tp=2,
            pp=2,
            recompute_num_layers=1,
            memory_model=1000.0,
        ),
        _strategy(
            "dp1-tp2-fit",
            time_cost=99.0,
            dp=1,
            tp=2,
            pp=2,
            recompute_num_layers=3,
            memory_model=80.0,
        ),
    ]

    selected = select_assignment_candidates(strategies, max_assignments=2, config=config)

    assert [strategy["name"] for strategy in selected] == [
        "dp2-tp1-fast",
        "dp1-tp2-fit",
    ]


def test_shortlist_prefers_memory_viable_family_representative():
    config = _config(gpu_memory=100.0)
    strategies = [
        _strategy("dp2-tp1-fast", time_cost=1.0, dp=2, tp=1, pp=2, recompute_num_layers=1),
        _strategy("dp2-tp1-next", time_cost=2.0, dp=2, tp=1, pp=2, recompute_num_layers=2),
        _strategy(
            "dp1-tp2-oom",
            time_cost=3.0,
            dp=1,
            tp=2,
            pp=2,
            recompute_num_layers=1,
            memory_model=1000.0,
        ),
        _strategy(
            "dp1-tp2-fit",
            time_cost=99.0,
            dp=1,
            tp=2,
            pp=2,
            recompute_num_layers=3,
            memory_model=80.0,
        ),
    ]

    shortlist = build_short_run_shortlist(strategies, config)

    assert [strategy["name"] for strategy in shortlist] == [
        "dp2-tp1-fast",
        "dp2-tp1-next",
        "dp1-tp2-fit",
    ]


def test_shortlist_prefers_profiled_safe_family_representative():
    config = _config(gpu_memory=81920.0)
    strategies = [
        _strategy("dp2-tp1-fast", time_cost=1.0, dp=2, tp=1, pp=2, recompute_num_layers=1),
        _strategy("dp2-tp1-next", time_cost=2.0, dp=2, tp=1, pp=2, recompute_num_layers=2),
        _strategy(
            "dp1-tp2-block1-profiled-oom",
            time_cost=3.0,
            dp=1,
            tp=2,
            pp=2,
            recompute_num_layers=1,
            memory_model=81317.0,
        ),
        _strategy(
            "dp1-tp2-block3-profiled-safe",
            time_cost=99.0,
            dp=1,
            tp=2,
            pp=2,
            recompute_num_layers=3,
            memory_model=80069.0,
        ),
    ]
    strategies[2]["memory_model_profiled_total"] = 82703.0
    strategies[3]["memory_model_profiled_total"] = 81455.0

    shortlist = build_short_run_shortlist(strategies, config)

    assert [strategy["name"] for strategy in shortlist] == [
        "dp2-tp1-fast",
        "dp2-tp1-next",
        "dp1-tp2-block3-profiled-safe",
    ]


def test_shortlist_global_topk_prefers_profiled_safe_candidate():
    config = _config(gpu_memory=81920.0)
    config.experiment.auto_tuner.planner.topk_plans_for_short_run = 1
    strategies = [
        _strategy(
            "pp4-block8-profiled-oom",
            time_cost=1.0,
            dp=1,
            tp=1,
            pp=4,
            recompute_num_layers=8,
            memory_model=81490.0,
        ),
        _strategy(
            "tp2-pp2-block3-profiled-safe",
            time_cost=99.0,
            dp=1,
            tp=2,
            pp=2,
            recompute_num_layers=3,
            memory_model=80069.0,
        ),
    ]
    strategies[0]["memory_model_profiled_total"] = 82876.0
    strategies[1]["memory_model_profiled_total"] = 81455.0

    shortlist = build_short_run_shortlist(strategies, config)

    assert [strategy["name"] for strategy in shortlist] == [
        "tp2-pp2-block3-profiled-safe",
        "pp4-block8-profiled-oom",
    ]


def test_pp_first_memory_injection_records_peak_bias_audit_fields(tmp_path):
    config = _memory_config(tmp_path)
    strategy = _strategy(
        "needs-memory-injection",
        time_cost=1.0,
        dp=1,
        tp=1,
        pp=1,
        recompute_num_layers=1,
    )
    strategy.pop("memory_model")
    strategy["use_recompute"] = False
    strategy["recompute_method"] = None
    strategy["recompute_granularity"] = None
    strategy["recompute_num_layers"] = None

    shortlist = build_short_run_shortlist([strategy], config)
    injected = shortlist[0]

    assert injected["memory_model_peak_activation_bias"] == pytest.approx(256.0)
    assert injected["memory_model_profiled_peak"] == pytest.approx(
        injected["memory_model"] + 256.0
    )
