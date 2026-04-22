from copy import deepcopy

import pytest

from flagscale.train.hetero.segment_runtime import (
    SegmentRuntimeMeshSpec,
    SegmentRuntimeSpec,
    SegmentRuntimeStageSpec,
)
from flagscale.train.hetero.segment_runtime_args import parse_segment_runtime_args


def _segment_runtime_config():
    return {
        "train": {
            "system": {
                "hetero": {
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
            }
        }
    }


def test_parse_segment_runtime_args_returns_immutable_spec_without_mutating_input():
    config = _segment_runtime_config()
    original = deepcopy(config)

    spec = parse_segment_runtime_args(config)

    assert isinstance(spec, SegmentRuntimeSpec)
    assert config == original
    assert spec.to_runtime_dict() == original["train"]["system"]["hetero"]["segment_runtime"]
    assert isinstance(spec.stages, tuple)
    assert isinstance(spec.stages[0].segment_meshes, tuple)


@pytest.mark.parametrize(
    ("mesh_update", "expected_message"),
    [
        (
            [
                [[2, 1, 1, 1, 2], [1, 1, 1, 2, 1]],
                [[1, 1, 1, 2, 1], [2, 1, 1, 1, 1]],
            ],
            "pp_local",
        ),
        (
            [
                [[2, 1, 1, 1, True], [1, 1, 1, 2, 1]],
                [[1, 1, 1, 2, 1], [2, 1, 1, 1, 1]],
            ],
            "pp_local",
        ),
        (
            [
                [[0, 1, 1, 1, 1], [1, 1, 1, 2, 1]],
                [[1, 1, 1, 2, 1], [2, 1, 1, 1, 1]],
            ],
            "positive",
        ),
    ],
)
def test_parse_segment_runtime_args_rejects_invalid_mesh_values(mesh_update, expected_message):
    config = _segment_runtime_config()
    config["train"]["system"]["hetero"]["segment_runtime"]["hetero_stage_segment_meshes"] = mesh_update

    with pytest.raises(ValueError, match=expected_message):
        parse_segment_runtime_args(config)


@pytest.mark.parametrize(
    ("stage_update", "expected_message"),
    [
        (
            {
                "hetero_stage_segment_splits": [[1, 1, 1], [1, 1]],
                "hetero_stage_segment_meshes": [
                    [[2, 1, 1, 1, 1], [1, 1, 1, 2, 1]],
                    [[1, 1, 1, 2, 1], [2, 1, 1, 1, 1]],
                ],
            },
            "segment counts",
        ),
        (
            {
                "hetero_stage_segment_transitions": [
                    {
                        "source_stage_id": 0,
                        "target_stage_id": 0,
                        "kind": "segment-redistribution",
                        "source_segment_index": 0,
                        "target_segment_index": 2,
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
                ]
            },
            "transition indexes",
        ),
        (
            {
                "hetero_stage_segment_splits": [[]],
                "hetero_stage_segment_meshes": [[]],
                "hetero_stage_segment_transitions": [],
            },
            "must not be empty",
        ),
    ],
)
def test_parse_segment_runtime_args_rejects_inconsistent_segment_layout(stage_update, expected_message):
    config = _segment_runtime_config()
    config["train"]["system"]["hetero"]["segment_runtime"].update(stage_update)

    with pytest.raises(ValueError, match=expected_message):
        parse_segment_runtime_args(config)


def test_parse_segment_runtime_args_rejects_unknown_segment_runtime_keys():
    config = _segment_runtime_config()
    config["train"]["system"]["hetero"]["segment_runtime"]["unexpected"] = 1

    with pytest.raises(ValueError, match="unexpected"):
        parse_segment_runtime_args(config)


def test_parse_segment_runtime_args_normalizes_transition_order():
    config = _segment_runtime_config()
    config["train"]["system"]["hetero"]["segment_runtime"].update(
        {
            "hetero_stage_segment_splits": [[1, 1, 1]],
            "hetero_stage_segment_meshes": [[[2, 1, 1, 1, 1], [1, 1, 1, 2, 1], [1, 1, 1, 1, 1]]],
            "hetero_stage_segment_transitions": [
                {
                    "source_stage_id": 0,
                    "target_stage_id": 0,
                    "kind": "segment-redistribution",
                    "source_segment_index": 1,
                    "target_segment_index": 2,
                    "source_mesh": {"tp": 1, "cp": 1, "ep": 1, "dp": 2, "pp": 1},
                    "target_mesh": {"tp": 1, "cp": 1, "ep": 1, "dp": 1, "pp": 1},
                    "batch_unit": 2,
                    "redistribution_kind": "dp-only",
                    "requires_sequence_parallel": False,
                },
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
            ],
        }
    )

    spec = parse_segment_runtime_args(config)

    transitions = spec.to_runtime_dict()["hetero_stage_segment_transitions"]
    assert [(item["source_segment_index"], item["target_segment_index"]) for item in transitions] == [
        (0, 1),
        (1, 2),
    ]


def test_segment_runtime_public_specs_freeze_nested_sequences():
    splits = [1, 1]
    meshes = [
        SegmentRuntimeMeshSpec(2, 1, 1, 1, 1),
        SegmentRuntimeMeshSpec(1, 1, 1, 2, 1),
    ]
    stage = SegmentRuntimeStageSpec(segment_splits=splits, segment_meshes=meshes, transitions=[])
    spec = SegmentRuntimeSpec(stages=[stage])

    splits.append(3)
    meshes.append(SegmentRuntimeMeshSpec(1, 1, 1, 2, 1))

    assert spec.stages[0].segment_splits == (1, 1)
    assert len(spec.stages[0].segment_meshes) == 2


@pytest.mark.parametrize(
    ("transition_update", "expected_message"),
    [
        (
            {
                "batch_unit": 3,
            },
            "batch_unit",
        ),
        (
            {
                "redistribution_kind": "banana",
            },
            "redistribution_kind",
        ),
        (
            {
                "requires_sequence_parallel": False,
            },
            "requires_sequence_parallel",
        ),
    ],
)
def test_parse_segment_runtime_args_rejects_invalid_transition_semantics(
    transition_update,
    expected_message,
):
    config = _segment_runtime_config()
    config["train"]["system"]["hetero"]["segment_runtime"]["hetero_stage_segment_transitions"][0].update(
        transition_update
    )

    with pytest.raises(ValueError, match=expected_message):
        parse_segment_runtime_args(config)


def test_parse_segment_runtime_args_rejects_transition_mesh_mismatch():
    config = _segment_runtime_config()
    config["train"]["system"]["hetero"]["segment_runtime"]["hetero_stage_segment_transitions"][0][
        "source_mesh"
    ]["tp"] = 99

    with pytest.raises(ValueError, match="source_mesh"):
        parse_segment_runtime_args(config)


@pytest.mark.parametrize(
    ("mesh_index", "value", "expected_message"),
    [
        ((0, 1, 1), 2, "context_parallel_size"),
        ((0, 1, 2), 2, "expert_model_parallel_size"),
    ],
)
def test_parse_segment_runtime_args_rejects_stage_local_cp_ep_changes(
    mesh_index,
    value,
    expected_message,
):
    config = _segment_runtime_config()
    stage_id, segment_id, dim_index = mesh_index
    config["train"]["system"]["hetero"]["segment_runtime"]["hetero_stage_segment_meshes"][stage_id][
        segment_id
    ][dim_index] = value
    transition_key = "target_mesh" if segment_id == 1 else "source_mesh"
    mesh_field = "cp" if dim_index == 1 else "ep"
    config["train"]["system"]["hetero"]["segment_runtime"]["hetero_stage_segment_transitions"][
        stage_id
    ][transition_key][mesh_field] = value

    with pytest.raises(ValueError, match=expected_message):
        parse_segment_runtime_args(config)


def test_segment_runtime_mesh_spec_rejects_non_int_pp_local():
    with pytest.raises(ValueError, match="pp_local"):
        SegmentRuntimeMeshSpec(1, 1, 1, 1, 1.0)
