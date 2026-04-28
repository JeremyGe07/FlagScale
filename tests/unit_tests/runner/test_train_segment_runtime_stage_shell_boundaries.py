from pathlib import Path
import sys
from types import SimpleNamespace

MEGATRON_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "Megatron-LM"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))

from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_layer import BaseTransformerLayer
import torch

from flagscale.train.hetero.segment_runtime import SegmentRuntimeMeshSpec
from flagscale.train.hetero.segment_runtime_args import parse_segment_runtime_args


def _segment_runtime_config():
    return {
        "train": {
            "system": {
                "hetero": {
                    "segment_runtime": {
                        "hetero_stage_segment_splits": [[1, 1]],
                        "hetero_stage_segment_meshes": [[[2, 1, 1, 1, 1], [1, 1, 1, 2, 1]]],
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
                            }
                        ],
                    }
                }
            }
        }
    }


class _DummyLayer(torch.nn.Module, BaseTransformerLayer):
    def __init__(self, config, layer_number=1, **_kwargs):
        super().__init__()
        self.config = config
        self.layer_number = layer_number

    def forward(self, hidden_states, attention_mask=None, context=None, **_kwargs):
        return hidden_states, context

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        return {}


class _FakeGroup:
    def __init__(self, name, size=1, rank=0):
        self.name = name
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _transformer_config(spec):
    return TransformerConfig(
        num_layers=2,
        hidden_size=8,
        ffn_hidden_size=16,
        num_attention_heads=2,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        sequence_parallel=True,
        segment_runtime_spec=spec,
        segment_stage_id=0,
    )


def _patch_segment_groups(monkeypatch):
    monkeypatch.setattr("megatron.core.transformer.transformer_block.get_pg_rank", lambda _group: 0)
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_block.parallel_state.get_segment_process_group",
        lambda stage_id, segment_index, token, is_expert=False: _FakeGroup(
            f"{stage_id}-{segment_index}-{token}",
            size=2 if token in {"tp", "dp"} else 1,
        ),
        raising=False,
    )


def _build_block(config):
    return TransformerBlock(
        config=config,
        spec=ModuleSpec(module=_DummyLayer),
        pg_collection=SimpleNamespace(pp=None, tp=None, dp=None, cp=None),
        post_layer_norm=False,
        post_process=False,
    )


def test_segment_runtime_adds_output_boundary_back_to_stage_shell(monkeypatch):
    spec = parse_segment_runtime_args(_segment_runtime_config())
    config = _transformer_config(spec)
    _patch_segment_groups(monkeypatch)

    block = _build_block(config)

    assert [type(layer).__name__ for layer in block.layers] == [
        "SegmentModuleBlock",
        "SegmentRedistributionBoundary",
        "SegmentModuleBlock",
        "SegmentRedistributionBoundary",
    ]
    output_boundary = block.layers[-1]
    assert output_boundary.source_mesh["dp"] == 2
    assert output_boundary.target_mesh["tp"] == 2
    assert output_boundary.target_mesh["dp"] == 1


def test_segment_runtime_adds_input_boundary_from_stage_shell(monkeypatch):
    mesh = SegmentRuntimeMeshSpec(1, 1, 1, 2, 1)
    stage = SimpleNamespace(segment_splits=(2,), segment_meshes=(mesh,), transitions=())
    config = _transformer_config(SimpleNamespace(stages=(stage,)))
    _patch_segment_groups(monkeypatch)

    block = _build_block(config)

    assert [type(layer).__name__ for layer in block.layers] == [
        "SegmentRedistributionBoundary",
        "SegmentModuleBlock",
        "SegmentRedistributionBoundary",
    ]
    input_boundary = block.layers[0]
    assert input_boundary.source_mesh["tp"] == 2
    assert input_boundary.target_mesh["dp"] == 2
