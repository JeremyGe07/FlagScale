import pytest

from flagscale.runner.auto_tuner.search.pp_first_partition import (
    build_layer_count_balanced_partition,
)
import flagscale.runner.auto_tuner.search.pp_first_stage_candidates as stage_candidates_mod
from flagscale.runner.auto_tuner.search.pp_first_transition_cost import (
    estimate_transition_cost,
)
from flagscale.runner.auto_tuner.search.pp_first_searcher import PPFirstSearcher
from flagscale.runner.auto_tuner.search.pp_first_stage_candidates import (
    build_stage_candidates,
)
from tests.unit_tests.runner.test_auto_tuner_pp_first_dp_solver_helpers import (
    config as _config,
    stage_candidate,
)


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


def test_estimate_transition_cost_has_activation_transfer_when_tp_and_dp_stay_same(
    tmp_path,
):
    config = _config(tmp_path)
    previous_stage = stage_candidate(dp=2, tp=2, pp=2)
    current_stage = stage_candidate(dp=2, tp=2, pp=2)

    cost = estimate_transition_cost(previous_stage, current_stage, config)

    assert cost["transition_total_ms"] > 0
    assert cost["tp_transition_ms"] == 0
    assert cost["dp_transition_ms"] == 0
    assert cost["activation_transfer_ms"] > 0


def test_estimate_transition_cost_total_equals_component_sum(tmp_path):
    config = _config(tmp_path)
    previous_stage = stage_candidate(dp=1, tp=4, pp=2, mb=2)
    current_stage = stage_candidate(dp=2, tp=2, pp=2, mb=2)

    cost = estimate_transition_cost(previous_stage, current_stage, config)

    assert cost["transition_total_ms"] == (
        cost["tp_transition_ms"]
        + cost["dp_transition_ms"]
        + cost["activation_transfer_ms"]
    )


def test_estimate_transition_cost_penalizes_tp_changes(tmp_path):
    config = _config(tmp_path)
    previous_stage = stage_candidate(dp=1, tp=2, pp=2)
    current_stage = stage_candidate(dp=1, tp=1, pp=2)

    cost = estimate_transition_cost(previous_stage, current_stage, config)

    assert cost["transition_total_ms"] > 0
    assert cost["tp_transition_ms"] > 0
    assert cost["dp_transition_ms"] == 0


def test_estimate_transition_cost_tp_penalty_grows_with_larger_change(tmp_path):
    config = _config(tmp_path)
    previous_stage = stage_candidate(dp=1, tp=4, pp=2)
    smaller_change = stage_candidate(dp=1, tp=3, pp=2)
    larger_change = stage_candidate(dp=1, tp=1, pp=2)

    smaller_cost = estimate_transition_cost(previous_stage, smaller_change, config)
    larger_cost = estimate_transition_cost(previous_stage, larger_change, config)

    assert larger_cost["tp_transition_ms"] > smaller_cost["tp_transition_ms"]


def test_estimate_transition_cost_penalizes_dp_changes(tmp_path):
    config = _config(tmp_path)
    previous_stage = stage_candidate(dp=1, tp=2, pp=2)
    current_stage = stage_candidate(dp=2, tp=2, pp=2)

    cost = estimate_transition_cost(previous_stage, current_stage, config)

    assert cost["transition_total_ms"] > 0
    assert cost["tp_transition_ms"] == 0
    assert cost["dp_transition_ms"] > 0


def test_estimate_transition_cost_dp_penalty_grows_with_larger_change(tmp_path):
    config = _config(tmp_path)
    previous_stage = stage_candidate(dp=4, tp=2, pp=2)
    smaller_change = stage_candidate(dp=3, tp=2, pp=2)
    larger_change = stage_candidate(dp=1, tp=2, pp=2)

    smaller_cost = estimate_transition_cost(previous_stage, smaller_change, config)
    larger_cost = estimate_transition_cost(previous_stage, larger_change, config)

    assert larger_cost["dp_transition_ms"] > smaller_cost["dp_transition_ms"]


def test_estimate_transition_cost_penalizes_pipeline_boundary_activation_transfer(tmp_path):
    config = _config(tmp_path)
    previous_stage = stage_candidate(dp=2, tp=2, pp=2)
    current_stage = stage_candidate(dp=2, tp=2, pp=2)

    cost = estimate_transition_cost(previous_stage, current_stage, config)

    assert cost["transition_total_ms"] > 0
    assert cost["tp_transition_ms"] == 0
    assert cost["dp_transition_ms"] == 0
    assert cost["activation_transfer_ms"] > 0


def test_estimate_transition_cost_activation_transfer_scales_with_boundary_size(tmp_path):
    config = _config(tmp_path)
    small_previous = stage_candidate(dp=2, tp=2, pp=2, mb=1)
    small_current = stage_candidate(dp=2, tp=2, pp=2, mb=1)
    large_previous = stage_candidate(dp=2, tp=2, pp=2, mb=2)
    large_current = stage_candidate(dp=2, tp=2, pp=2, mb=2)

    small_cost = estimate_transition_cost(small_previous, small_current, config)
    large_cost = estimate_transition_cost(large_previous, large_current, config)

    assert large_cost["activation_transfer_ms"] == small_cost["activation_transfer_ms"] * 2


def test_estimate_transition_cost_rejects_inconsistent_stage_micro_batch_size(tmp_path):
    config = _config(tmp_path)
    previous_stage = stage_candidate(dp=2, tp=2, pp=2, mb=1)
    current_stage = stage_candidate(dp=2, tp=2, pp=2, mb=2)

    with pytest.raises(ValueError, match="micro_batch_size must match across a stage boundary"):
        estimate_transition_cost(previous_stage, current_stage, config)


def test_estimate_transition_cost_rejects_inconsistent_pipeline_parallel_size(tmp_path):
    config = _config(tmp_path)
    previous_stage = stage_candidate(dp=2, tp=2, pp=2)
    current_stage = stage_candidate(dp=2, tp=2, pp=4)

    with pytest.raises(
        ValueError,
        match="pipeline_model_parallel_size must match across a stage boundary",
    ):
        estimate_transition_cost(previous_stage, current_stage, config)


def test_estimate_transition_cost_rejects_non_mapping_model_config(tmp_path):
    config = _config(tmp_path)
    config.train.model = 1
    previous_stage = stage_candidate(dp=2, tp=2, pp=2)
    current_stage = stage_candidate(dp=2, tp=2, pp=2)

    with pytest.raises(ValueError, match="train.model must be a mapping"):
        estimate_transition_cost(previous_stage, current_stage, config)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    (
        ("data_parallel_size", 1.5),
        ("tensor_model_parallel_size", True),
        ("pipeline_model_parallel_size", "abc"),
        ("micro_batch_size", object()),
    ),
)
def test_estimate_transition_cost_rejects_invalid_stage_parallel_dimensions(
    tmp_path,
    field_name,
    bad_value,
):
    config = _config(tmp_path)
    previous_stage = stage_candidate(dp=2, tp=2, pp=2)
    current_stage = stage_candidate(dp=2, tp=2, pp=2)
    previous_stage[field_name] = bad_value

    with pytest.raises(
        ValueError,
        match=f"previous_stage.{field_name} must be a positive integer",
    ):
        estimate_transition_cost(previous_stage, current_stage, config)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    (
        ("hidden_size", "abc"),
        ("seq_length", []),
        ("hidden_size", float("nan")),
        ("hidden_size", float("inf")),
        ("seq_length", float("nan")),
        ("seq_length", float("inf")),
    ),
)
def test_estimate_transition_cost_rejects_non_finite_model_dimensions(
    tmp_path,
    field_name,
    bad_value,
):
    config = _config(tmp_path)
    setattr(config.train.model, field_name, bad_value)
    previous_stage = stage_candidate(dp=2, tp=2, pp=2)
    current_stage = stage_candidate(dp=2, tp=2, pp=2)

    if isinstance(bad_value, float):
        pattern = f"train.model.{field_name} must be finite"
    else:
        pattern = f"train.model.{field_name} must be a positive number"

    with pytest.raises(ValueError, match=pattern):
        estimate_transition_cost(previous_stage, current_stage, config)
