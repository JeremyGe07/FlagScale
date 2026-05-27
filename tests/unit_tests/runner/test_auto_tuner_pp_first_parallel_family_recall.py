from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.search.pp_first_selection import (
    build_short_run_shortlist,
    select_assignment_candidates,
)


def _config(gpu_memory=None):
    auto_tuner = {
        "algo": {"name": "grid", "use_profiled_time_cost": True},
        "planner": {"topk_plans_for_short_run": 2},
    }
    if gpu_memory is not None:
        auto_tuner["memory_model"] = {"gpu_memory": gpu_memory}
    return OmegaConf.create({"experiment": {"auto_tuner": auto_tuner}})


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
