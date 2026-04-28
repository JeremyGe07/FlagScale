from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from flagscale.train.hetero.segment_runtime import SegmentRuntimeMeshSpec
from flagscale.train.hetero.segment_runtime_args import parse_segment_runtime_args

MEGATRON_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "Megatron-LM"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core import parallel_state
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_layer import BaseTransformerLayer
import gpt_builders


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


class _CountingLayer(torch.nn.Module, BaseTransformerLayer):
    def __init__(self, config, layer_number=1, **_kwargs):
        super().__init__()
        self.config = config
        self.layer_number = layer_number

    def forward(self, hidden_states, attention_mask=None, context=None, **_kwargs):
        return hidden_states + float(self.layer_number), context

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        return {}


class _RotaryAwareLayer(torch.nn.Module, BaseTransformerLayer):
    def __init__(self, config, layer_number=1, **_kwargs):
        super().__init__()
        self.config = config
        self.layer_number = layer_number

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        **_kwargs,
    ):
        if rotary_pos_cos is None or rotary_pos_sin is None:
            raise AssertionError("rotary-pos-cos-sin-required")
        return hidden_states, context

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        return {}


class _MicrobatchAwareLayer(torch.nn.Module, BaseTransformerLayer):
    def __init__(self, config, layer_number=1, **_kwargs):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.current_microbatch = -1

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


def _fake_stage_spec(*, pp_local=1, transitions=None):
    meshes = (
        SegmentRuntimeMeshSpec(2, 1, 1, 1, 1),
        SimpleNamespace(
            tensor_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            data_parallel_size=2,
            pp_local=pp_local,
        ),
    )
    return SimpleNamespace(
        segment_splits=(1, 1),
        segment_meshes=meshes,
        transitions=tuple(() if transitions is None else transitions),
    )


def _single_segment_runtime_spec():
    mesh = SegmentRuntimeMeshSpec(2, 1, 1, 1, 1)
    stage = SimpleNamespace(segment_splits=(2,), segment_meshes=(mesh,), transitions=())
    return SimpleNamespace(stages=(stage,))


def test_model_parallel_config_accepts_segment_runtime_fields():
    spec = parse_segment_runtime_args(_segment_runtime_config())

    config = ModelParallelConfig(segment_runtime_spec=spec, segment_stage_id=0)

    assert config.segment_runtime_spec is spec
    assert config.segment_stage_id == 0


def test_gpt_builder_threads_segment_runtime_spec_from_parallel_context(monkeypatch):
    spec = parse_segment_runtime_args(_segment_runtime_config())
    config = SimpleNamespace(segment_runtime_spec=None, segment_stage_id=None)
    para_ctx = SimpleNamespace(
        get_transformer_config=lambda: config,
        has_segment_runtime=lambda: True,
        get_segment_runtime_spec=lambda: spec,
    )
    args = SimpleNamespace(yaml_cfg=None, use_legacy_models=True)

    monkeypatch.setattr("flagscale.train.global_vars.get_parallel_context", lambda: para_ctx)
    monkeypatch.setattr(gpt_builders.parallel_state, "get_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        gpt_builders.megatron.legacy.model,
        "GPTModel",
        lambda model_config, **_kwargs: model_config,
    )

    result = gpt_builders.gpt_builder(args, pre_process=True, post_process=True)

    assert result is config
    assert config.segment_runtime_spec is spec
    assert config.segment_stage_id == 0


def test_gpt_builder_threads_segment_runtime_spec_for_yaml_config(monkeypatch):
    spec = parse_segment_runtime_args(_segment_runtime_config())
    config = SimpleNamespace(segment_runtime_spec=None, segment_stage_id=None)
    args = SimpleNamespace(yaml_cfg=object(), use_legacy_models=True, segment_runtime_spec=spec)

    monkeypatch.setattr(
        gpt_builders,
        "core_transformer_config_from_yaml",
        lambda passed_args, _key: config,
    )
    monkeypatch.setattr(gpt_builders.parallel_state, "get_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        gpt_builders.megatron.legacy.model,
        "GPTModel",
        lambda model_config, **_kwargs: model_config,
    )

    result = gpt_builders.gpt_builder(args, pre_process=True, post_process=True)

    assert result is config
    assert config.segment_runtime_spec is spec
    assert config.segment_stage_id == 0


def test_parallel_state_uses_parallel_context_tensor_and_context_rank(monkeypatch):
    para_ctx = SimpleNamespace(
        get_tensor_and_context_parallel_rank=lambda: 7,
        get_tensor_and_context_parallel_world_size=lambda: 8,
    )

    monkeypatch.setattr(parallel_state, "get_parallel_context", lambda: para_ctx)

    assert parallel_state.get_tensor_and_context_parallel_rank() == 7


def test_transformer_block_inserts_segment_redistribution_boundaries(monkeypatch):
    spec = parse_segment_runtime_args(_segment_runtime_config())
    config = _transformer_config(spec)

    monkeypatch.setattr("megatron.core.transformer.transformer_block.get_pg_rank", lambda _group: 0)
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_block.parallel_state.get_segment_process_group",
        lambda stage_id, segment_index, token, is_expert=False: _FakeGroup(
            f"{stage_id}-{segment_index}-{token}"
        ),
        raising=False,
    )
    block = TransformerBlock(
        config=config,
        spec=ModuleSpec(module=_DummyLayer),
        pg_collection=SimpleNamespace(pp=None, tp=None, dp=None, cp=None),
        post_layer_norm=False,
        post_process=False,
    )

    assert [type(layer).__name__ for layer in block.layers] == [
        "SegmentModuleBlock",
        "SegmentRedistributionBoundary",
        "SegmentModuleBlock",
        "SegmentRedistributionBoundary",
    ]


def test_transformer_block_materializes_segment_blocks_with_local_configs(monkeypatch):
    spec = parse_segment_runtime_args(_segment_runtime_config())
    config = _transformer_config(spec)

    monkeypatch.setattr("megatron.core.transformer.transformer_block.get_pg_rank", lambda _group: 0)
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_block.parallel_state.get_segment_process_group",
        lambda stage_id, segment_index, token, is_expert=False: _FakeGroup(
            f"{stage_id}-{segment_index}-{token}",
            size=2 if token == "tp" and segment_index == 0 else 1,
        ),
        raising=False,
    )
    block = TransformerBlock(
        config=config,
        spec=ModuleSpec(module=_DummyLayer),
        pg_collection=SimpleNamespace(pp=None, tp=None, dp=None, cp=None),
        post_layer_norm=False,
        post_process=False,
    )

    assert [type(layer).__name__ for layer in block.layers] == [
        "SegmentModuleBlock",
        "SegmentRedistributionBoundary",
        "SegmentModuleBlock",
        "SegmentRedistributionBoundary",
    ]
    assert block.layers[0].config.tensor_model_parallel_size == 2
    assert block.layers[0].config.data_parallel_size == 1
    assert block.layers[2].config.tensor_model_parallel_size == 1
    assert block.layers[2].config.data_parallel_size == 2


def test_checkpointed_forward_uses_segment_execution_nodes(monkeypatch):
    spec = parse_segment_runtime_args(_segment_runtime_config())
    config = _transformer_config(spec)
    config.recompute_granularity = "full"
    config.recompute_method = "uniform"
    config.recompute_num_layers = 1

    monkeypatch.setattr("megatron.core.transformer.transformer_block.get_pg_rank", lambda _group: 0)
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_block.parallel_state.get_segment_process_group",
        lambda stage_id, segment_index, token, is_expert=False: _FakeGroup(
            f"{stage_id}-{segment_index}-{token}"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_block.tensor_parallel.checkpoint",
        lambda forward_func, _distribute_saved_activations, *args: forward_func(*args),
    )
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_block._redistribute_segment_tensor",
        lambda tensor, *_args, **_kwargs: tensor,
    )
    block = TransformerBlock(
        config=config,
        spec=ModuleSpec(module=_CountingLayer),
        pg_collection=SimpleNamespace(pp=None, tp=None, dp=None, cp=None),
        post_layer_norm=False,
        post_process=False,
    )

    hidden_states = torch.zeros(2, 1, 8, requires_grad=True)
    output = block._checkpointed_forward(
        hidden_states=hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        attention_bias=None,
        packed_seq_params=None,
        use_inner_quantization_context=False,
    )

    assert torch.equal(output, torch.full_like(hidden_states, 3.0))


def test_checkpointed_forward_threads_rotary_pos_cos_sin(monkeypatch):
    spec = parse_segment_runtime_args(_segment_runtime_config())
    config = _transformer_config(spec)
    config.recompute_granularity = "full"
    config.recompute_method = "uniform"
    config.recompute_num_layers = 1

    monkeypatch.setattr("megatron.core.transformer.transformer_block.get_pg_rank", lambda _group: 0)
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_block.parallel_state.get_segment_process_group",
        lambda stage_id, segment_index, token, is_expert=False: _FakeGroup(
            f"{stage_id}-{segment_index}-{token}"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_block.tensor_parallel.checkpoint",
        lambda forward_func, _distribute_saved_activations, *args: forward_func(*args),
    )
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_block._redistribute_segment_tensor",
        lambda tensor, *_args, **_kwargs: tensor,
    )
    block = TransformerBlock(
        config=config,
        spec=ModuleSpec(module=_RotaryAwareLayer),
        pg_collection=SimpleNamespace(pp=None, tp=None, dp=None, cp=None),
        post_layer_norm=False,
        post_process=False,
    )

    hidden_states = torch.zeros(2, 1, 8, requires_grad=True)
    block._checkpointed_forward(
        hidden_states=hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=torch.ones_like(hidden_states),
        rotary_pos_sin=torch.ones_like(hidden_states),
        attention_bias=None,
        packed_seq_params=None,
        use_inner_quantization_context=False,
    )


def test_segment_module_block_wraps_inner_quantization_per_child(monkeypatch):
    config = _transformer_config(_single_segment_runtime_spec())
    config.fp8 = "hybrid"
    context_calls = []

    monkeypatch.setattr("megatron.core.transformer.transformer_block.get_pg_rank", lambda _group: 0)
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_block.parallel_state.get_segment_process_group",
        lambda stage_id, segment_index, token, is_expert=False: _FakeGroup(
            f"{stage_id}-{segment_index}-{token}"
        ),
        raising=False,
    )

    def fake_fp8_context(_config, layer_index=None, is_init=False):
        if not is_init:
            context_calls.append(layer_index)
        return nullcontext()

    monkeypatch.setattr(
        "megatron.core.transformer.transformer_block.get_fp8_context",
        fake_fp8_context,
    )
    block = TransformerBlock(
        config=config,
        spec=ModuleSpec(module=_DummyLayer),
        pg_collection=SimpleNamespace(pp=None, tp=None, dp=None, cp=None),
        post_layer_norm=False,
        post_process=False,
    )

    segment_block = block.layers[0]
    context_calls.clear()
    segment_block(
        hidden_states=torch.zeros(2, 1, 8, requires_grad=True),
        attention_mask=None,
        context=None,
        use_inner_quantization_context=True,
    )

    assert context_calls == [0, 1]


def test_segment_module_block_propagates_current_microbatch_to_child_layers(monkeypatch):
    config = _transformer_config(_single_segment_runtime_spec())

    monkeypatch.setattr("megatron.core.transformer.transformer_block.get_pg_rank", lambda _group: 0)
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_block.parallel_state.get_segment_process_group",
        lambda stage_id, segment_index, token, is_expert=False: _FakeGroup(
            f"{stage_id}-{segment_index}-{token}"
        ),
        raising=False,
    )
    block = TransformerBlock(
        config=config,
        spec=ModuleSpec(module=_MicrobatchAwareLayer),
        pg_collection=SimpleNamespace(pp=None, tp=None, dp=None, cp=None),
        post_layer_norm=False,
        post_process=False,
    )

    segment_block = block.layers[0]
    segment_block.current_microbatch = 3
    segment_block(
        hidden_states=torch.zeros(2, 1, 8, requires_grad=True),
        attention_mask=None,
        context=None,
    )

    assert [layer.current_microbatch for layer in segment_block.layers] == [3, 3]


def test_transformer_block_uses_virtual_pipeline_recompute_override_without_typo(monkeypatch):
    spec = parse_segment_runtime_args(_segment_runtime_config())
    config = _transformer_config(spec)
    config.sequence_parallel = False
    config.virtual_pipeline_model_parallel_size = 1
    config.recompute_method_per_stage_micro_batch = [[1]]

    monkeypatch.setattr("megatron.core.transformer.transformer_block.get_pg_rank", lambda _group: 0)
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_block.parallel_state.get_segment_process_group",
        lambda stage_id, segment_index, token, is_expert=False: _FakeGroup(
            f"{stage_id}-{segment_index}-{token}"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_block.parallel_state.get_virtual_pipeline_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_block.parallel_state.get_pipeline_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_block._redistribute_segment_tensor",
        lambda tensor, *_args, **_kwargs: tensor,
    )
    block = TransformerBlock(
        config=config,
        spec=ModuleSpec(module=_DummyLayer),
        pg_collection=SimpleNamespace(pp=None, tp=None, dp=None, cp=None),
        post_layer_norm=False,
        post_process=False,
    )

    output = block(
        hidden_states=torch.zeros(2, 1, 8, requires_grad=True),
        attention_mask=None,
        context=None,
    )

    assert output.shape == (2, 1, 8)
    assert config.recompute_method == "block"


def test_segment_boundary_carries_transition_metadata_and_segment_groups(monkeypatch):
    spec = parse_segment_runtime_args(_segment_runtime_config())
    config = _transformer_config(spec)

    monkeypatch.setattr("megatron.core.transformer.transformer_block.get_pg_rank", lambda _group: 0)
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_block.parallel_state.get_segment_process_group",
        lambda stage_id, segment_index, token, is_expert=False: _FakeGroup(
            f"{stage_id}-{segment_index}-{token}",
            size=2 if token in {"tp", "dp"} else 1,
        ),
        raising=False,
    )
    block = TransformerBlock(
        config=config,
        spec=ModuleSpec(module=_DummyLayer),
        pg_collection=SimpleNamespace(pp=None, tp=None, dp=None, cp=None),
        post_layer_norm=False,
        post_process=False,
    )

    boundary = block.layers[1]
    assert boundary.redistribution_kind == "tp-dp"
    assert boundary.requires_sequence_parallel is True
    assert boundary.source_mesh["tp"] == 2
    assert boundary.target_mesh["dp"] == 2
    assert boundary.source_pg_collection.tp.name == "0-0-tp"
    assert boundary.target_pg_collection.dp.name == "0-1-dp"


@pytest.mark.parametrize(
    ("config_update", "expected_message"),
    [
        ("bad_pp_local", "pp_local"),
        ("missing_transition", "transition"),
    ],
)
def test_transformer_block_rejects_invalid_segment_runtime_contract(config_update, expected_message):
    valid_spec = parse_segment_runtime_args(_segment_runtime_config())
    stage_spec = _fake_stage_spec(pp_local=2) if config_update == "bad_pp_local" else _fake_stage_spec()
    spec = SimpleNamespace(stages=(stage_spec,))
    config = _transformer_config(spec)
    config.sequence_parallel = valid_spec.stages[0].transitions[0].requires_sequence_parallel

    with pytest.raises(ValueError, match=expected_message):
        TransformerBlock(
            config=config,
            spec=ModuleSpec(module=_DummyLayer),
            pg_collection=SimpleNamespace(pp=None, tp=None),
        )
