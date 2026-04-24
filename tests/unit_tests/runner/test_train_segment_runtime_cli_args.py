import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from flagscale.runner.runner_train import _get_args_megatron
from flagscale.train.hetero.segment_runtime_args import parse_segment_runtime_args

MEGATRON_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "Megatron-LM"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))

from megatron.training.arguments import _add_hetero_args  # noqa: E402


def _transition():
    return {
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
    }


def test_megatron_parser_accepts_segment_runtime_cli_json():
    transitions = [_transition()]
    parser = _add_hetero_args(argparse.ArgumentParser())

    args = parser.parse_args(
        [
            "--hetero-stage-segment-splits",
            json.dumps([[1, 1]]),
            "--hetero-stage-segment-meshes",
            json.dumps([[[2, 1, 1, 1, 1], [1, 1, 1, 2, 1]]]),
            "--hetero-stage-segment-transitions",
            json.dumps(transitions),
        ]
    )
    spec = parse_segment_runtime_args(args)

    assert spec is not None
    assert spec.stages[0].segment_splits == (1, 1)
    assert spec.stages[0].segment_meshes[0].tensor_model_parallel_size == 2
    assert spec.stages[0].segment_meshes[1].data_parallel_size == 2


def test_megatron_parser_accepts_segment_runtime_args_generated_by_runner():
    runtime = {
        "hetero_stage_segment_splits": [[1, 1]],
        "hetero_stage_segment_meshes": [[[2, 1, 1, 1, 1], [1, 1, 1, 2, 1]]],
        "hetero_stage_segment_transitions": [_transition()],
    }
    config = OmegaConf.create(
        {
            "experiment": {"task": {"backend": "megatron"}},
            "train": {
                "system": {"hetero": {"segment_runtime": runtime}},
                "model": {},
                "data": {},
            },
        }
    )
    parser = _add_hetero_args(argparse.ArgumentParser())

    cli_args = _get_args_megatron(config)
    parsed = parser.parse_args(cli_args)
    spec = parse_segment_runtime_args(parsed)

    assert cli_args[cli_args.index("--hetero-stage-segment-splits") + 1] == json.dumps(
        runtime["hetero_stage_segment_splits"]
    )
    assert spec is not None
    assert spec.stages[0].segment_splits == (1, 1)
