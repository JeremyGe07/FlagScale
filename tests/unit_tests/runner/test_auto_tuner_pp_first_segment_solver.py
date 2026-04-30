from flagscale.runner.auto_tuner.plan.lowering import lower_strategy_to_plan
from flagscale.runner.auto_tuner.plan.summary import plan_kind
from flagscale.runner.auto_tuner.plan.validator import validate_model_plan
from flagscale.runner.auto_tuner.search.pp_first_assignment import (
    generate_assignment_candidates,
)
from flagscale.runner.auto_tuner.search.pp_first_partition import (
    build_layer_count_balanced_partition,
)
import flagscale.runner.auto_tuner.search.pp_first_segment_assignment as segment_assignment_mod
import flagscale.runner.auto_tuner.search.pp_first_segment_candidates as segment_mod
from flagscale.runner.auto_tuner.search.pp_first_segment_candidates import (
    build_segment_stage_candidates,
)
from tests.unit_tests.runner.test_auto_tuner_pp_first_dp_solver_helpers import (
    config as _config,
    stage_candidate,
)


def _stage_candidate(stage_index, stage_range, device_group, *, tp, dp, cost):
    candidate = stage_candidate(dp=dp, tp=tp, pp=1, sp=True)
    candidate.update(
        {
            "stage_index": stage_index,
            "stage_range": stage_range,
            "stage_device_group": device_group,
            "num_layers": stage_range[1] - stage_range[0] + 1,
            "stage_time_cost": cost,
            "stage_memory_model": 100.0 + cost,
            "time_cost": cost,
            "memory_model": 100.0 + cost,
            "runtime_mode": "stage-executable",
            "runtime_executable": True,
        }
    )
    return candidate


def _segment_stage_candidate(stage_index, stage_range, device_group, *, meshes, cost):
    candidate = _stage_candidate(
        stage_index,
        stage_range,
        device_group,
        tp=meshes[0][0],
        dp=meshes[0][1],
        cost=cost,
    )
    segment_strategies = [
        stage_candidate(dp=dp, tp=tp, pp=1, sp=True)
        for tp, dp in meshes
    ]
    candidate.update(
        {
            "segment_partition_ranges": [[0, 2], [3, 4]],
            "segment_strategies": segment_strategies,
            "segment_stage_candidate": True,
        }
    )
    return candidate


def test_build_segment_stage_candidates_creates_stage_local_segment_combinations(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path, cards=4, global_batch_size=8)
    partition = build_layer_count_balanced_partition(
        num_layers=10,
        pp_degree=2,
        world_size=4,
    )
    stage_range = (5, 9)
    device_group = (0, 1)
    base_candidates = [
        _stage_candidate(1, stage_range, device_group, tp=2, dp=1, cost=10.0),
        _stage_candidate(1, stage_range, device_group, tp=1, dp=2, cost=8.0),
    ]
    monkeypatch.setattr(
        segment_mod,
        "_base_stage_candidates",
        lambda **kwargs: base_candidates,
    )

    candidates = build_segment_stage_candidates(
        searcher=object(),
        space={},
        config=config,
        partition=partition,
        stage_index=1,
        max_segment_splits=1,
        max_segment_candidates=4,
    )

    assert candidates
    assert all(candidate["stage_index"] == 1 for candidate in candidates)
    assert all(candidate["segment_stage_candidate"] is True for candidate in candidates)
    assert candidates[0]["segment_partition_ranges"] == [[0, 1], [2, 4]]
    assert len(candidates[0]["segment_strategies"]) == 2
    assert (
        candidates[0]["segment_strategies"][0]["tensor_model_parallel_size"],
        candidates[0]["segment_strategies"][1]["tensor_model_parallel_size"],
    ) in {(2, 1), (1, 2)}


def test_segment_stage_candidates_normalize_sequence_parallel_for_tp_switch(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path, cards=4, global_batch_size=8)
    partition = build_layer_count_balanced_partition(
        num_layers=10,
        pp_degree=2,
        world_size=4,
    )
    stage_range = (0, 4)
    device_group = (0, 1)
    base_candidates = [
        _stage_candidate(0, stage_range, device_group, tp=1, dp=2, cost=8.0),
        _stage_candidate(0, stage_range, device_group, tp=2, dp=1, cost=10.0),
    ]
    base_candidates[0]["sequence_parallel"] = False
    base_candidates[1]["sequence_parallel"] = True
    monkeypatch.setattr(
        segment_mod,
        "_base_stage_candidates",
        lambda **kwargs: base_candidates,
    )

    candidates = build_segment_stage_candidates(
        searcher=object(),
        space={},
        config=config,
        partition=partition,
        stage_index=0,
        max_segment_splits=1,
        max_segment_candidates=4,
    )

    assert candidates
    assert all(
        segment["sequence_parallel"] is True
        for segment in candidates[0]["segment_strategies"]
    )


def test_segment_stage_candidates_skip_mismatched_micro_batch_tuples(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path, cards=4, global_batch_size=8)
    partition = build_layer_count_balanced_partition(
        num_layers=10,
        pp_degree=2,
        world_size=4,
    )
    stage_range = (0, 4)
    device_group = (0, 1)
    tp1_mbs1 = _stage_candidate(0, stage_range, device_group, tp=1, dp=2, cost=8.0)
    tp1_mbs2 = _stage_candidate(0, stage_range, device_group, tp=1, dp=2, cost=8.5)
    tp2_mbs2 = _stage_candidate(0, stage_range, device_group, tp=2, dp=1, cost=9.0)
    tp2_mbs1 = _stage_candidate(0, stage_range, device_group, tp=2, dp=1, cost=10.0)
    tp1_mbs1["micro_batch_size"] = 1
    tp1_mbs2["micro_batch_size"] = 2
    tp2_mbs2["micro_batch_size"] = 2
    tp2_mbs1["micro_batch_size"] = 1
    monkeypatch.setattr(
        segment_mod,
        "_base_stage_candidates",
        lambda **kwargs: [tp1_mbs1, tp1_mbs2, tp2_mbs2, tp2_mbs1],
    )

    candidates = build_segment_stage_candidates(
        searcher=object(),
        space={},
        config=config,
        partition=partition,
        stage_index=0,
        max_segment_splits=1,
        max_segment_candidates=4,
    )

    assert candidates
    assert {
        segment["micro_batch_size"]
        for segment in candidates[0]["segment_strategies"]
    } == {2}


def test_segment_stage_candidates_skip_invalid_global_batch_contracts(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path, cards=4, global_batch_size=8)
    partition = build_layer_count_balanced_partition(
        num_layers=10,
        pp_degree=2,
        world_size=4,
    )
    stage_range = (0, 4)
    device_group = (0, 1)
    invalid_first = _stage_candidate(0, stage_range, device_group, tp=1, dp=1, cost=1.0)
    invalid_first["acc_step"] = 8
    invalid_second = _stage_candidate(0, stage_range, device_group, tp=1, dp=2, cost=2.0)
    invalid_second["acc_step"] = 2
    valid_first = _stage_candidate(0, stage_range, device_group, tp=1, dp=2, cost=3.0)
    valid_first["acc_step"] = 2
    valid_second = _stage_candidate(0, stage_range, device_group, tp=2, dp=1, cost=4.0)
    valid_second["acc_step"] = 4
    valid_second["sequence_parallel"] = True
    monkeypatch.setattr(
        segment_mod,
        "_base_stage_candidates",
        lambda **kwargs: [invalid_first, invalid_second, valid_first, valid_second],
    )

    candidates = build_segment_stage_candidates(
        searcher=object(),
        space={},
        config=config,
        partition=partition,
        stage_index=0,
        max_segment_splits=1,
        max_segment_candidates=4,
    )

    assert candidates
    assert all(
        segment["data_parallel_size"] * segment["tensor_model_parallel_size"] == 2
        for segment in candidates[0]["segment_strategies"]
    )


def test_segment_dp_assignment_materializes_multi_pp_stage_segment_plan(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path, cards=4, global_batch_size=8)
    config.experiment.auto_tuner.planner = {
        "assignment_solver": "segment_dp",
        "max_segment_splits_per_stage": 1,
        "max_segment_candidates_per_stage": 2,
        "max_dp_results_per_partition": 2,
    }
    partition = build_layer_count_balanced_partition(
        num_layers=10,
        pp_degree=2,
        world_size=4,
    )
    stage_candidates = [
        [
            _segment_stage_candidate(
                0,
                (0, 4),
                (0, 1),
                meshes=((2, 1), (1, 2)),
                cost=5.0,
            ),
        ],
        [
            _segment_stage_candidate(
                1,
                (5, 9),
                (2, 3),
                meshes=((1, 2), (2, 1)),
                cost=6.0,
            ),
        ],
    ]
    monkeypatch.setattr(
        segment_assignment_mod,
        "build_segment_stage_candidates",
        lambda **kwargs: stage_candidates[kwargs["stage_index"]],
    )

    strategies = generate_assignment_candidates(
        searcher=object(),
        space={},
        config=config,
        pp_degree=2,
        max_assignments=2,
        partition=partition,
    )

    strategy = strategies[0]
    plan = lower_strategy_to_plan(strategy, config)
    validation = validate_model_plan(plan)

    assert strategy["runtime_mode"] == "segment-executable"
    assert strategy["runtime_executable"] is True
    assert plan_kind(plan) == "segment-heterogeneous"
    assert validation.runtime_mode == "segment-executable"
    assert [len(stage.segments) for stage in plan.stages] == [2, 2]
    assert all("segment_strategies" in stage for stage in strategy["stage_strategies"])
