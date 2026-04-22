from types import SimpleNamespace

import pytest

from flagscale.train.hetero.parallel_context import ParallelContext
from flagscale.train.hetero.segment_runtime_args import parse_segment_runtime_args


def _segment_runtime_spec():
    return parse_segment_runtime_args(
        {
            "segment_runtime": {
                "hetero_stage_segment_splits": [[1, 1], [1, 1]],
                "hetero_stage_segment_meshes": [
                    [[2, 1, 1, 1, 1], [1, 1, 1, 2, 1]],
                    [[1, 1, 1, 2, 1], [2, 1, 1, 1, 1]],
                ],
                "hetero_stage_segment_transitions": [
                    {
                        "source_stage_id": 0,
                        "target_stage_id": 0,
                        "kind": "segment-redistribution",
                        "source_segment_index": 0,
                        "target_segment_index": 1,
                        "source_mesh": {"tp": 2, "cp": 1, "ep": 1, "dp": 1, "pp": 1},
                        "target_mesh": {"tp": 1, "cp": 1, "ep": 1, "dp": 2, "pp": 1},
                        "batch_unit": 2,
                        "redistribution_kind": "tp-dp",
                        "requires_sequence_parallel": True,
                    },
                    {
                        "source_stage_id": 1,
                        "target_stage_id": 1,
                        "kind": "segment-redistribution",
                        "source_segment_index": 0,
                        "target_segment_index": 1,
                        "source_mesh": {"tp": 1, "cp": 1, "ep": 1, "dp": 2, "pp": 1},
                        "target_mesh": {"tp": 2, "cp": 1, "ep": 1, "dp": 1, "pp": 1},
                        "batch_unit": 2,
                        "redistribution_kind": "tp-dp",
                        "requires_sequence_parallel": True,
                    },
                ],
            }
        }
    )


class _FakeProcessMesh:
    def __init__(self, stage_ranks, stage_groups, order="tp-cp-ep-dp-pp", offset=0):
        self._stage_ranks = tuple(stage_ranks)
        self._stage_groups = dict(stage_groups)
        if order is not None:
            self._order = order
        self._offset = offset
        self._world_size = len(stage_ranks)
        self._rank_mapper = _IdentityRankMapper()

    def get_parallel_size(self, token, is_expert=False):
        assert is_expert is False
        if token == "pp":
            return 1
        raise AssertionError(f"unexpected token: {token}")

    def get_process_group(self, token, is_expert=False, gloo=False, check_initialized=False):
        assert is_expert is False
        assert check_initialized is True
        key = f"{token}-gloo" if gloo else token
        return self._stage_groups[key]


class _IdentityRankMapper:
    @staticmethod
    def to_physical_ranks(logical_ranks):
        return list(logical_ranks)


def _make_parallel_context(*, current_rank, segment_runtime_spec=None):
    ctx = ParallelContext.__new__(ParallelContext)
    ctx._args = SimpleNamespace(
        segment_runtime_spec=segment_runtime_spec,
        use_tp_pp_dp_mapping=False,
    )
    ctx._rank = current_rank
    ctx._process_meshes = [
        _FakeProcessMesh(
            stage_ranks=(0, 1),
            stage_groups={"tp": "stage0-tp", "dp": "stage0-dp", "dp-gloo": "stage0-dp-gloo"},
            offset=0,
        ),
        _FakeProcessMesh(
            stage_ranks=(2, 3),
            stage_groups={"tp": "stage1-tp", "dp": "stage1-dp", "dp-gloo": "stage1-dp-gloo"},
            offset=2,
        ),
    ]
    ctx._current_process_mesh_index = 0
    ctx._segment_runtime_spec = None
    ctx._segment_mesh_lookup = {}
    ctx._segment_process_group_ranks = {}
    ctx._segment_all_process_group_ranks = {}
    if segment_runtime_spec is not None:
        ctx._build_segment_runtime_lookups()
    return ctx


def test_parallel_context_builds_segment_local_lookup_tables():
    spec = _segment_runtime_spec()
    ctx = _make_parallel_context(current_rank=0, segment_runtime_spec=spec)

    assert ctx.has_segment_runtime() is True
    assert ctx.get_segment_runtime_spec() is spec
    assert ctx.get_segment_mesh(0, 0).tensor_model_parallel_size == 2
    assert ctx.get_segment_mesh(0, 1).data_parallel_size == 2
    assert ctx.get_segment_mesh(1, 1).tensor_model_parallel_size == 2
    assert ctx.get_segment_process_group_ranks(0, 0, "tp") == (0, 1)
    assert ctx.get_segment_process_group_ranks(0, 1, "dp") == (0, 1)
    assert ctx.get_segment_process_group_world_size(0, 1, "tp") == 1
    assert ctx.get_segment_process_group_rank(0, 1, "dp") == 0
    assert ctx.get_segment_all_process_group_ranks(0, 1, "tp") == ((0,), (1,))
    assert ctx.get_segment_all_process_group_ranks(1, 0, "dp") == ((2, 3),)


def test_parallel_context_without_segment_runtime_keeps_stage_level_lookups():
    ctx = _make_parallel_context(current_rank=0)

    assert ctx.has_segment_runtime() is False
    assert ctx.get_segment_runtime_spec() is None
    assert ctx.get_tensor_model_parallel_group() == "stage0-tp"
    assert ctx.get_data_parallel_group() == "stage0-dp"
    assert ctx.get_data_parallel_group_gloo() == "stage0-dp-gloo"
    with pytest.raises(RuntimeError, match="segment runtime"):
        ctx.get_segment_mesh(0, 0)


def test_parallel_context_uses_megatron_order_for_multi_axis_segment_groups():
    spec = parse_segment_runtime_args(
        {
            "segment_runtime": {
                "hetero_stage_segment_splits": [[1]],
                "hetero_stage_segment_meshes": [[[2, 2, 1, 2, 1]]],
                "hetero_stage_segment_transitions": [],
            }
        }
    )
    ctx = ParallelContext.__new__(ParallelContext)
    ctx._args = SimpleNamespace(
        segment_runtime_spec=spec,
        use_tp_pp_dp_mapping=False,
    )
    ctx._rank = 5
    ctx._process_meshes = [
        _FakeProcessMesh(
            stage_ranks=tuple(range(8)),
            stage_groups={},
            order="tp-cp-ep-dp-pp",
        ),
    ]
    ctx._current_process_mesh_index = 0
    ctx._segment_runtime_spec = None
    ctx._segment_mesh_lookup = {}
    ctx._segment_process_group_ranks = {}
    ctx._segment_all_process_group_ranks = {}

    ctx._build_segment_runtime_lookups()

    assert ctx.get_segment_all_process_group_ranks(0, 0, "tp-dp") == (
        (0, 1, 4, 5),
        (2, 3, 6, 7),
    )
    assert ctx.get_segment_process_group_ranks(0, 0, "tp-dp") == (0, 1, 4, 5)
    assert ctx.get_segment_process_group_rank(0, 0, "tp-dp") == 3


def test_parallel_context_builds_segment_local_expert_groups():
    spec = parse_segment_runtime_args(
        {
            "segment_runtime": {
                "hetero_stage_segment_splits": [[1]],
                "hetero_stage_segment_meshes": [[[2, 1, 2, 1, 1]]],
                "hetero_stage_segment_transitions": [],
            }
        }
    )
    ctx = ParallelContext.__new__(ParallelContext)
    ctx._args = SimpleNamespace(
        segment_runtime_spec=spec,
        use_tp_pp_dp_mapping=False,
    )
    ctx._rank = 2
    ctx._process_meshes = [_FakeProcessMesh(stage_ranks=tuple(range(4)), stage_groups={})]
    ctx._current_process_mesh_index = 0
    ctx._segment_runtime_spec = None
    ctx._segment_mesh_lookup = {}
    ctx._segment_process_group_ranks = {}
    ctx._segment_all_process_group_ranks = {}

    ctx._build_segment_runtime_lookups()

    assert ctx.get_segment_all_process_group_ranks(0, 0, "ep", is_expert=True) == (
        (0, 2),
        (1, 3),
    )
    assert ctx.get_segment_process_group_ranks(0, 0, "ep", is_expert=True) == (0, 2)
    assert ctx.get_segment_process_group_rank(0, 0, "ep", is_expert=True) == 1
    assert ctx.get_segment_process_group_world_size(0, 0, "tp-ep", is_expert=True) == 4


def test_parallel_context_honors_tp_pp_dp_lookup_fallback_order():
    spec = parse_segment_runtime_args(
        {
            "segment_runtime": {
                "hetero_stage_segment_splits": [[1]],
                "hetero_stage_segment_meshes": [[[2, 1, 1, 2, 1]]],
                "hetero_stage_segment_transitions": [],
            }
        }
    )
    ctx = ParallelContext.__new__(ParallelContext)
    ctx._args = SimpleNamespace(
        segment_runtime_spec=spec,
        use_tp_pp_dp_mapping=True,
    )
    ctx._rank = 2
    ctx._process_meshes = [_FakeProcessMesh(stage_ranks=tuple(range(4)), stage_groups={}, order=None)]
    ctx._current_process_mesh_index = 0
    ctx._segment_runtime_spec = None
    ctx._segment_mesh_lookup = {}
    ctx._segment_process_group_ranks = {}
    ctx._segment_all_process_group_ranks = {}

    ctx._build_segment_runtime_lookups()

    assert ctx.get_segment_all_process_group_ranks(0, 0, "dp") == ((0, 2), (1, 3))
    assert ctx.get_segment_process_group_ranks(0, 0, "dp") == (0, 2)
