from pathlib import Path
from types import SimpleNamespace
import sys

import torch

MEGATRON_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "Megatron-LM"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))

from megatron.core.transformer import transformer_block


class _RotaryAwareLayer(torch.nn.Module):
    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        sequence_len_offset=None,
        **_kwargs,
    ):
        assert rotary_pos_cos is not None
        assert rotary_pos_sin is not None
        assert sequence_len_offset is not None
        return hidden_states + float(self.layer_number), context


class _FakeBlock:
    def __init__(self):
        self.config = SimpleNamespace(
            fp8=False,
            fp4=False,
            distribute_saved_activations=False,
            recompute_method="block",
            recompute_num_layers=1,
        )
        self.layers = [_RotaryAwareLayer(1), _RotaryAwareLayer(2)]

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _execution_layer_count(self):
        return len(self.layers)


def test_block_recompute_threads_rotary_args_into_uncheckpointed_layers(monkeypatch):
    monkeypatch.setattr(
        transformer_block.tensor_parallel,
        "checkpoint",
        lambda forward_func, _distribute_saved_activations, *args: forward_func(*args),
    )
    hidden_states = torch.zeros(2, 1, 8, requires_grad=True)

    output = transformer_block.TransformerBlock._checkpointed_forward(
        _FakeBlock(),
        hidden_states=hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=torch.ones_like(hidden_states),
        rotary_pos_sin=torch.ones_like(hidden_states),
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=torch.tensor(0),
        use_inner_quantization_context=False,
    )

    assert torch.equal(output, torch.full_like(hidden_states, 3.0))
