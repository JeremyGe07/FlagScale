import pytest

from flagscale.runner.auto_tuner.search.pp_first_partition import (
    build_layer_count_balanced_partition,
)
from flagscale.runner.auto_tuner.search.pp_first_dp_solver import (
    solve_stage_level_dp,
)
import flagscale.runner.auto_tuner.search.pp_first_stage_candidates as stage_candidates_mod
from flagscale.runner.auto_tuner.search.pp_first_searcher import PPFirstSearcher
from flagscale.runner.auto_tuner.search.pp_first_stage_candidates import (
    build_stage_candidates,
)
from tests.unit_tests.runner.test_auto_tuner_pp_first_dp_solver_helpers import (
    config as _config,
    stage_candidate,
)


def _stage_candidate_with_cost(
    *,
    stage_time_cost,
    dp=1,
    tp=1,
    pp=2,
    sp=False,
    mb=2,
    acc_step=4,
    **metadata,
):
    candidate = stage_candidate(dp=dp, tp=tp, pp=pp, sp=sp, mb=mb, acc_step=acc_step)
    candidate["stage_time_cost"] = stage_time_cost
    candidate.update(metadata)
    return candidate


def test_stage_level_dp_solver_beats_greedy_local_choice_with_real_stage_candidates():
    stage_candidates = [
        [
            _stage_candidate_with_cost(stage_time_cost=10, dp=1, tp=1),
            _stage_candidate_with_cost(stage_time_cost=11, dp=1, tp=2),
        ],
        [
            _stage_candidate_with_cost(stage_time_cost=10, dp=1, tp=3),
            _stage_candidate_with_cost(stage_time_cost=13, dp=1, tp=4),
        ],
    ]

    def transition_cost(previous, current):
        transition_total_ms = {
            (10, 10): 10,
            (10, 13): 1,
            (11, 10): 1,
            (11, 13): 10,
        }[(previous["stage_time_cost"], current["stage_time_cost"])]
        return {
            "tp_transition_ms": 0,
            "dp_transition_ms": 0,
            "activation_transfer_ms": transition_total_ms,
            "transition_total_ms": transition_total_ms,
        }

    chains = solve_stage_level_dp(stage_candidates, transition_cost, max_results=2)

    assert [stage["stage_time_cost"] for stage in chains[0].stage_strategies] == [11, 10]
    assert chains[0].dp_aggregate_cost == 22


def test_stage_level_dp_solver_returns_top_n_chains_in_cost_order():
    stage_candidates = [
        [
            _stage_candidate_with_cost(stage_time_cost=1, dp=1, tp=1),
            _stage_candidate_with_cost(stage_time_cost=12, dp=1, tp=2),
        ],
        [
            _stage_candidate_with_cost(stage_time_cost=1, dp=1, tp=3),
            _stage_candidate_with_cost(stage_time_cost=3, dp=1, tp=4),
        ],
        [
            _stage_candidate_with_cost(stage_time_cost=1, dp=1, tp=5),
            _stage_candidate_with_cost(stage_time_cost=4, dp=1, tp=6),
        ],
    ]

    chains = solve_stage_level_dp(stage_candidates, lambda previous, current: 0.0, max_results=3)

    assert [
        [stage["stage_time_cost"] for stage in chain.stage_strategies]
        for chain in chains
    ] == [
        [1, 1, 1],
        [1, 3, 1],
        [1, 1, 4],
    ]
    assert [chain.dp_aggregate_cost for chain in chains] == [3, 5, 6]


def test_stage_level_dp_solver_preserves_metadata_and_structured_transition_breakdown():
    stage_candidates = [
        [
            _stage_candidate_with_cost(
                stage_time_cost=5,
                dp=2,
                tp=1,
                strategy_tag="first",
            ),
        ],
        [
            _stage_candidate_with_cost(
                stage_time_cost=7,
                dp=2,
                tp=3,
                strategy_tag="second",
            ),
        ],
    ]

    def transition_cost(previous, current):
        return {
            "tp_transition_ms": abs(
                previous["tensor_model_parallel_size"] - current["tensor_model_parallel_size"]
            ),
            "dp_transition_ms": 2.0,
            "activation_transfer_ms": 6.0,
            "transition_total_ms": 9.0,
        }

    chains = solve_stage_level_dp(stage_candidates, transition_cost, max_results=1)

    chain = chains[0]
    assert chain.stage_strategies[0]["strategy_tag"] == "first"
    assert chain.stage_strategies[1]["strategy_tag"] == "second"
    assert chain.stage_cost_breakdown == (
        {"stage_index": 0, "candidate_index": 0, "stage_time_cost": 5},
        {"stage_index": 1, "candidate_index": 0, "stage_time_cost": 7},
    )
    assert chain.transition_breakdown == (
        {
            "source_stage_index": 0,
            "target_stage_index": 1,
            "source_candidate_index": 0,
            "target_candidate_index": 0,
            "transition_cost_breakdown": {
                "tp_transition_ms": 2.0,
                "dp_transition_ms": 2.0,
                "activation_transfer_ms": 6.0,
                "transition_total_ms": 9.0,
            },
        },
    )
    assert chain.dp_aggregate_cost == 21


def test_stage_level_dp_solver_keeps_same_named_candidates_separate():
    stage_candidates = [
        [
            _stage_candidate_with_cost(stage_time_cost=5, dp=1, tp=1, name="dup"),
            _stage_candidate_with_cost(stage_time_cost=7, dp=1, tp=2, name="dup"),
        ],
        [
            _stage_candidate_with_cost(stage_time_cost=1, dp=1, tp=3),
        ],
    ]

    chains = solve_stage_level_dp(stage_candidates, lambda previous, current: 0.0, max_results=2)

    assert len(chains) == 2
    assert [chain.stage_cost_breakdown[0]["stage_time_cost"] for chain in chains] == [5, 7]


def test_stage_level_dp_solver_sorts_mixed_none_and_string_metadata():
    stage_candidates = [
        [
            _stage_candidate_with_cost(
                stage_time_cost=1,
                dp=1,
                tp=1,
                recompute_method=None,
            ),
            _stage_candidate_with_cost(
                stage_time_cost=1,
                dp=1,
                tp=1,
                recompute_method="block",
            ),
        ],
        [
            _stage_candidate_with_cost(stage_time_cost=1, dp=1, tp=1),
        ],
    ]

    chains = solve_stage_level_dp(stage_candidates, lambda previous, current: 0.0, max_results=2)

    assert len(chains) == 2


def test_stage_level_dp_solver_uses_boundary_aware_transition_mapping():
    stage_candidates = [
        [
            _stage_candidate_with_cost(stage_time_cost=1, dp=1, tp=1),
            _stage_candidate_with_cost(stage_time_cost=12, dp=1, tp=2),
        ],
        [
            _stage_candidate_with_cost(stage_time_cost=5, dp=1, tp=3),
            _stage_candidate_with_cost(stage_time_cost=1, dp=1, tp=4),
        ],
        [
            _stage_candidate_with_cost(stage_time_cost=1, dp=1, tp=5),
            _stage_candidate_with_cost(stage_time_cost=5, dp=1, tp=6),
        ],
    ]
    transitions = {
        (0, 0, 0): {"transition_total_ms": 10.0},
        (0, 0, 1): {"transition_total_ms": 10.0},
        (0, 1, 0): {"transition_total_ms": 0.0},
        (0, 1, 1): {"transition_total_ms": 0.0},
        (1, 0, 0): {"transition_total_ms": 10.0},
        (1, 0, 1): {"transition_total_ms": 10.0},
        (1, 1, 0): {"transition_total_ms": 0.0},
        (1, 1, 1): {"transition_total_ms": 10.0},
    }

    chains = solve_stage_level_dp(stage_candidates, transitions, max_results=1)

    assert [stage["stage_time_cost"] for stage in chains[0].stage_strategies] == [1, 1, 1]
    assert chains[0].dp_aggregate_cost == 13.0


def test_stage_level_dp_solver_rejects_boundary_agnostic_mapping_for_multi_stage():
    stage_candidates = [
        [
            _stage_candidate_with_cost(stage_time_cost=1, dp=1, tp=1),
        ],
        [
            _stage_candidate_with_cost(stage_time_cost=1, dp=1, tp=2),
        ],
        [
            _stage_candidate_with_cost(stage_time_cost=1, dp=1, tp=3),
        ],
    ]
    transitions = {
        (0, 0): {"transition_total_ms": 1.0},
    }

    with pytest.raises(ValueError, match="boundary-aware keys"):
        solve_stage_level_dp(stage_candidates, transitions, max_results=1)


def test_build_stage_candidates_prunes_invalid_and_keeps_planner_budget(tmp_path):
    config = _config(tmp_path, cards=4, global_batch_size=8)
    searcher = PPFirstSearcher(config)
    partition = build_layer_count_balanced_partition(num_layers=10, pp_degree=2, world_size=4)

    candidates = build_stage_candidates(
        searcher=searcher,
        space=searcher.space,
        config=config,
        partition=partition,
        stage_index=0,
    )

    assert len(candidates) == 2
    assert all(candidate["stage_index"] == 0 for candidate in candidates)
    assert all(candidate["stage_range"] == (0, 4) for candidate in candidates)
    assert all(candidate["stage_device_group"] == (0, 1) for candidate in candidates)
    assert all(
        candidate["pipeline_model_parallel_size"] == partition.pp_degree
        for candidate in candidates
    )
    assert candidates[0]["stage_time_cost"] <= candidates[1]["stage_time_cost"]


def test_build_stage_candidates_rejects_pipeline_model_parallel_size_mismatch(
    tmp_path, monkeypatch
):
    config = _config(tmp_path, cards=4, global_batch_size=8)
    searcher = PPFirstSearcher(config)
    partition = build_layer_count_balanced_partition(num_layers=10, pp_degree=2, world_size=4)
    bad_candidate = stage_candidate(dp=1, tp=2, pp=1)

    monkeypatch.setattr(
        stage_candidates_mod,
        "_generate_stage_candidates",
        lambda searcher, space, config: [bad_candidate],
    )

    with pytest.raises(ValueError, match="pipeline_model_parallel_size"):
        build_stage_candidates(
            searcher=searcher,
            space=searcher.space,
            config=config,
            partition=partition,
            stage_index=0,
            max_stage_candidates=2,
        )


def test_build_stage_candidates_rejects_stage_device_group_mesh_mismatch(tmp_path, monkeypatch):
    config = _config(tmp_path, cards=4, global_batch_size=8)
    searcher = PPFirstSearcher(config)
    partition = build_layer_count_balanced_partition(num_layers=10, pp_degree=2, world_size=4)
    bad_candidate = stage_candidate(dp=1, tp=1)
    good_candidate = stage_candidate(dp=1, tp=2)

    monkeypatch.setattr(
        stage_candidates_mod,
        "_generate_stage_candidates",
        lambda searcher, space, config: [bad_candidate, good_candidate],
    )

    candidates = build_stage_candidates(
        searcher=searcher,
        space=searcher.space,
        config=config,
        partition=partition,
        stage_index=0,
        max_stage_candidates=2,
    )

    assert len(candidates) == 1
    assert candidates[0]["tensor_model_parallel_size"] == 2


def test_build_stage_candidates_uses_chip_aware_estimate_sorting(tmp_path, monkeypatch):
    config = _config(tmp_path, cards=4, global_batch_size=8)
    config.experiment.auto_tuner.algo.use_profiled_time_cost = False
    config.experiment.auto_tuner.algo.chip_aware_scoring = True
    config.experiment.auto_tuner.planner.max_stage_candidates_per_stage = 1
    searcher = PPFirstSearcher(config)
    partition = build_layer_count_balanced_partition(num_layers=10, pp_degree=2, world_size=4)
    low_chip = stage_candidate(dp=2, tp=1)
    high_chip = stage_candidate(dp=1, tp=2)

    monkeypatch.setattr(
        stage_candidates_mod,
        "_generate_stage_candidates",
        lambda searcher, space, config: [low_chip, high_chip],
    )
    monkeypatch.setattr(
        stage_candidates_mod,
        "estimate_time_cost",
        lambda strategy, config: {
            "time_total_ms": 1.0
            if strategy.stages[0].segments[0].strategy["tensor_model_parallel_size"] == 1
            else 10.0,
            "time_breakdown": {},
        },
    )
    monkeypatch.setattr(
        stage_candidates_mod,
        "estimate_memory_cost",
        lambda strategy, config: {
            "memory_total_mb": 100.0,
            "memory_breakdown": {},
        },
    )

    candidates = build_stage_candidates(
        searcher=searcher,
        space=searcher.space,
        config=config,
        partition=partition,
        stage_index=0,
    )

    assert len(candidates) == 1
    assert candidates[0]["tensor_model_parallel_size"] == 2
    assert candidates[0]["chip_score"] > low_chip.get("chip_score", float("-inf"))


def test_build_stage_candidates_rejects_global_batch_divisibility(tmp_path):
    config = _config(tmp_path, cards=4, global_batch_size=3, micro_batch_size=(2,))
    searcher = PPFirstSearcher(config)
    partition = build_layer_count_balanced_partition(num_layers=10, pp_degree=2, world_size=4)

    with pytest.raises(ValueError, match="global batch"):
        build_stage_candidates(
            searcher=searcher,
            space=searcher.space,
            config=config,
            partition=partition,
            stage_index=0,
            max_stage_candidates=2,
        )


def test_build_stage_candidates_rejects_obvious_stage_memory_limits(tmp_path):
    config = _config(tmp_path, cards=4, global_batch_size=8, total_memory_mb=1)
    searcher = PPFirstSearcher(config)
    partition = build_layer_count_balanced_partition(num_layers=10, pp_degree=2, world_size=4)

    with pytest.raises(ValueError, match="memory"):
        build_stage_candidates(
            searcher=searcher,
            space=searcher.space,
            config=config,
            partition=partition,
            stage_index=0,
            max_stage_candidates=2,
        )
