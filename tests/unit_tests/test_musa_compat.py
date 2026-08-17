from types import SimpleNamespace

from flagscale.train.musa_compat import (
    _map_cuda_device,
    _patch_device_queries,
    _patch_real_mccl,
    _sdpa_dot_product_attention_forward,
)


class _Device:
    def __init__(self, device_type, index=None):
        self.type = device_type
        self.index = index


def test_map_cuda_device_preserves_non_cuda_devices():
    cpu_device = _Device("cpu", 0)

    assert _map_cuda_device("cuda") == "musa"
    assert _map_cuda_device("cuda:3") == "musa:3"
    assert _map_cuda_device(_Device("cuda")) == "musa"
    assert _map_cuda_device(_Device("cuda", 2)) == "musa:2"
    assert _map_cuda_device("cpu") == "cpu"
    assert _map_cuda_device(cpu_device) is cpu_device


def test_real_mccl_patch_translates_legacy_backend_spelling():
    calls = []

    def init_process_group(backend=None, *args, **kwargs):
        calls.append((backend, args, kwargs))
        return backend

    torch = SimpleNamespace(
        distributed=SimpleNamespace(
            init_process_group=init_process_group,
            is_initialized=lambda: True,
            get_world_size=lambda: 1,
        )
    )

    _patch_real_mccl(torch)

    assert torch.distributed.init_process_group("nccl", rank=0) == "mccl"
    assert (
        torch.distributed.init_process_group("cpu:gloo,cuda:nccl")
        == "cpu:gloo,musa:mccl"
    )
    assert calls == [
        ("mccl", (), {"rank": 0}),
        ("cpu:gloo,musa:mccl", (), {}),
    ]


def test_real_mccl_patch_eagerly_initializes_default_communicator():
    calls = []
    warmup_tensor = object()

    def init_process_group(backend=None, *args, **kwargs):
        calls.append(("init", backend, args, kwargs))

    def all_reduce(tensor):
        calls.append(("all_reduce", tensor))

    def synchronize():
        calls.append(("synchronize",))

    torch = SimpleNamespace(
        distributed=SimpleNamespace(
            init_process_group=init_process_group,
            is_initialized=lambda: True,
            get_world_size=lambda: 8,
            get_rank=lambda: 1,
            all_reduce=all_reduce,
        ),
        musa=SimpleNamespace(current_device=lambda: 3, synchronize=synchronize),
        ones=lambda *args, **kwargs: (
            calls.append(("ones", args, kwargs)) or warmup_tensor
        ),
    )

    _patch_real_mccl(torch)

    assert torch.distributed.init_process_group("nccl", rank=1) is None
    assert calls == [
        ("init", "mccl", (), {"rank": 1}),
        ("ones", (1,), {"device": "musa:3"}),
        ("all_reduce", warmup_tensor),
        ("synchronize",),
    ]


def test_device_query_patch_maps_cuda_device_objects_and_is_idempotent():
    calls = []

    def get_device_properties(device=None):
        calls.append(device)
        return device

    torch = SimpleNamespace(
        musa=SimpleNamespace(get_device_properties=get_device_properties)
    )

    _patch_device_queries(torch)
    _patch_device_queries(torch)

    assert torch.musa.get_device_properties(_Device("cuda", 1)) == "musa:1"
    assert calls == ["musa:1"]


def test_sdpa_attention_preserves_megatron_mask_and_gqa_semantics():
    import torch

    sequence_length = 8
    batch_size = 2
    query_heads = 4
    query_groups = 2
    head_dim = 4
    scale = 0.25

    torch.manual_seed(17)
    query = torch.randn(sequence_length, batch_size, query_heads, head_dim)
    key = torch.randn(sequence_length, batch_size, query_groups, head_dim)
    value = torch.randn(sequence_length, batch_size, query_groups, head_dim)

    # Megatron boolean masks use True for masked positions. Include a reset
    # boundary so this checks more than a plain triangular causal mask.
    attention_mask = torch.ones(
        batch_size, 1, sequence_length, sequence_length, dtype=torch.bool
    ).triu(diagonal=1)
    attention_mask[:, :, 4:, :4] = True

    attention = SimpleNamespace(
        num_attention_heads_per_partition=query_heads,
        num_query_groups_per_partition=query_groups,
        hidden_size_per_partition=query_heads * head_dim,
        softmax_scale=scale,
        attention_dropout=torch.nn.Dropout(0.0),
        training=True,
        config=SimpleNamespace(sequence_parallel=True),
        attn_mask_type=SimpleNamespace(name="causal"),
    )
    actual = _sdpa_dot_product_attention_forward(
        attention, query, key, value, attention_mask
    )

    expanded_key = key.repeat_interleave(query_heads // query_groups, dim=2)
    expanded_value = value.repeat_interleave(query_heads // query_groups, dim=2)
    query_bhsd = query.permute(1, 2, 0, 3)
    key_bhsd = expanded_key.permute(1, 2, 0, 3)
    value_bhsd = expanded_value.permute(1, 2, 0, 3)
    scores = torch.matmul(query_bhsd, key_bhsd.transpose(-1, -2)) * scale
    probabilities = torch.softmax(
        scores.masked_fill(attention_mask, float("-inf")), dim=-1
    )
    expected = torch.matmul(probabilities, value_bhsd)
    expected = expected.permute(2, 0, 1, 3).contiguous().view(
        sequence_length, batch_size, query_heads * head_dim
    )

    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)
