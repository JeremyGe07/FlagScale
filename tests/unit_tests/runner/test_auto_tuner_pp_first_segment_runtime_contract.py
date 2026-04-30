import flagscale.runner.auto_tuner.search.pp_first_segment_candidates as segment_mod
from flagscale.runner.auto_tuner.search.pp_first_partition import (
    build_layer_count_balanced_partition,
)
from flagscale.runner.auto_tuner.search.pp_first_segment_candidates import (
    build_segment_stage_candidates,
)
from tests.unit_tests.runner.test_auto_tuner_pp_first_dp_solver_helpers import (
    config as _config,
    stage_candidate,
)


def _stage_candidate(stage_range, device_group, *, tp, dp, sp):
    candidate = stage_candidate(dp=dp, tp=tp, pp=1, sp=sp)
    candidate.update(
        {
            "stage_index": 0,
            "stage_range": stage_range,
            "stage_device_group": device_group,
            "num_layers": stage_range[1] - stage_range[0] + 1,
            "stage_time_cost": 1.0,
            "stage_memory_model": 100.0,
            "time_cost": 1.0,
            "memory_model": 100.0,
            "runtime_mode": "stage-executable",
            "runtime_executable": True,
        }
    )
    return candidate


def test_segment_tp_switch_sets_top_level_sequence_parallel_contract(tmp_path, monkeypatch):
    config = _config(tmp_path, cards=4, global_batch_size=8)
    partition = build_layer_count_balanced_partition(
        num_layers=10,
        pp_degree=2,
        world_size=4,
    )
    stage_range = (0, 4)
    device_group = (0, 1)
    base_candidates = [
        _stage_candidate(stage_range, device_group, tp=1, dp=2, sp=False),
        _stage_candidate(stage_range, device_group, tp=2, dp=1, sp=True),
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
        stage_index=0,
        max_segment_splits=1,
        max_segment_candidates=4,
    )

    assert candidates[0]["sequence_parallel"] is True
