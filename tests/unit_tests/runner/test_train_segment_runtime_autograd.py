from pathlib import Path
from types import SimpleNamespace
import sys

import torch
import torch.distributed.nn.functional as dist_nn_functional

MEGATRON_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "Megatron-LM"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))

from megatron.core.transformer import transformer_block


class _FakeGroup:
    def __init__(self, size=1, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def test_segment_gather_preserves_autograd_chain(monkeypatch):
    group = _FakeGroup(size=2)
    source_pg = SimpleNamespace(tp=group, dp=_FakeGroup())
    target_pg = SimpleNamespace(tp=_FakeGroup(), dp=_FakeGroup())
    source_mesh = {"tp": 2, "cp": 1, "ep": 1, "dp": 1, "pp": 1}
    target_mesh = {"tp": 1, "cp": 1, "ep": 1, "dp": 1, "pp": 1}

    def fail_non_autograd_all_gather(*_args, **_kwargs):
        raise AssertionError("segment redistribution must use autograd-aware all_gather")

    def fake_autograd_all_gather(tensor, group=None):
        assert group is source_pg.tp
        return tensor, tensor * 2

    monkeypatch.setattr(torch.distributed, "all_gather", fail_non_autograd_all_gather)
    monkeypatch.setattr(dist_nn_functional, "all_gather", fake_autograd_all_gather)

    hidden_states = torch.ones(2, 2, requires_grad=True)

    output = transformer_block._redistribute_segment_tensor(
        hidden_states,
        source_mesh,
        target_mesh,
        source_pg,
        target_pg,
        requires_sequence_parallel=True,
    )

    assert output.requires_grad
    output.sum().backward()
    assert torch.equal(hidden_states.grad, torch.full_like(hidden_states, 3.0))
