from copy import deepcopy

import pytest

from flagscale.train.hetero.segment_runtime import SegmentRuntimeSpec
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
                                "metadata": {
                                    "source_mesh": [2, 1, 1, 1, 1],
                                    "target_mesh": [1, 1, 1, 2, 1],
                                },
                            },
                            {
                                "source_stage_id": 1,
                                "target_stage_id": 1,
                                "kind": "segment-redistribution",
                                "source_segment_index": 0,
                                "target_segment_index": 1,
                                "metadata": {
                                    "source_mesh": [1, 1, 1, 2, 1],
                                    "target_mesh": [2, 1, 1, 1, 1],
                                },
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
    ],
)
def test_parse_segment_runtime_args_rejects_non_unit_pp_local(mesh_update, expected_message):
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
                        "metadata": {},
                    },
                    {
                        "source_stage_id": 1,
                        "target_stage_id": 1,
                        "kind": "segment-redistribution",
                        "source_segment_index": 0,
                        "target_segment_index": 1,
                        "metadata": {},
                    },
                ]
            },
            "transition indexes",
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
