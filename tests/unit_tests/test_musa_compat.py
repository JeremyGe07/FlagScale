from types import SimpleNamespace

from flagscale.train.musa_compat import (
    _map_cuda_device,
    _patch_device_queries,
    _patch_real_mccl,
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
        distributed=SimpleNamespace(init_process_group=init_process_group)
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
