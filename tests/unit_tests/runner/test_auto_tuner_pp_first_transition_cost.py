import pytest

from flagscale.runner.auto_tuner.search.pp_first_transition_cost import (
    estimate_transition_cost,
)
from tests.unit_tests.runner.test_auto_tuner_pp_first_dp_solver_helpers import (
    config as _config,
    stage_candidate,
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
