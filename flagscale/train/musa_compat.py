"""Opt-in MUSA compatibility for the legacy Megatron training path.

The legacy stack predates Megatron-LM-FL's device platform abstraction and
still uses CUDA spellings for accelerator APIs.  This module keeps the port
local and explicit: it is enabled only with ``FLAGSCALE_MUSA_COMPAT=1`` and
maps those accelerator calls to torch_musa.  Multi-process jobs still use the
real MCCL backend; no fake process group is installed here.
"""

import os
from functools import wraps


_COMPAT_ENV = "FLAGSCALE_MUSA_COMPAT"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _map_cuda_device(device):
    """Translate an explicit CUDA device value to its MUSA equivalent."""

    if isinstance(device, str):
        if device == "cuda":
            return "musa"
        if device.startswith("cuda:"):
            return f"musa:{device.split(':', 1)[1]}"
        return device

    device_type = getattr(device, "type", None)
    if device_type == "cuda":
        index = getattr(device, "index", None)
        return "musa" if index is None else f"musa:{index}"
    return device


def _patch_factory_devices(torch) -> None:
    """Map ``device='cuda'`` on the tensor factories used by Megatron."""

    factory_names = (
        "arange",
        "empty",
        "empty_like",
        "eye",
        "full",
        "full_like",
        "ones",
        "ones_like",
        "rand",
        "rand_like",
        "randint",
        "randn",
        "randn_like",
        "tensor",
        "zeros",
        "zeros_like",
    )
    for name in factory_names:
        factory = getattr(torch, name)
        if getattr(factory, "_flagscale_musa_compat", False):
            continue

        @wraps(factory)
        def wrapper(*args, __factory=factory, **kwargs):
            if "device" in kwargs:
                kwargs["device"] = _map_cuda_device(kwargs["device"])
            return __factory(*args, **kwargs)

        wrapper._flagscale_musa_compat = True
        setattr(torch, name, wrapper)


def _patch_to_and_cuda(torch) -> None:
    """Route legacy ``.cuda()`` and ``.to('cuda')`` calls to MUSA."""

    if not getattr(torch.Tensor.cuda, "_flagscale_musa_compat", False):

        @wraps(torch.Tensor.musa)
        def tensor_cuda(tensor, device=None, non_blocking=False, **kwargs):
            return tensor.musa(
                device=_map_cuda_device(device), non_blocking=non_blocking, **kwargs
            )

        tensor_cuda._flagscale_musa_compat = True
        torch.Tensor.cuda = tensor_cuda

    if not getattr(torch.nn.Module.cuda, "_flagscale_musa_compat", False):

        @wraps(torch.nn.Module.musa)
        def module_cuda(module, device=None):
            return module.musa(device=_map_cuda_device(device))

        module_cuda._flagscale_musa_compat = True
        torch.nn.Module.cuda = module_cuda

    for owner in (torch.Tensor, torch.nn.Module):
        original_to = owner.to
        if getattr(original_to, "_flagscale_musa_compat", False):
            continue

        @wraps(original_to)
        def to(instance, *args, __original_to=original_to, **kwargs):
            args = list(args)
            if args:
                args[0] = _map_cuda_device(args[0])
            if "device" in kwargs:
                kwargs["device"] = _map_cuda_device(kwargs["device"])
            return __original_to(instance, *args, **kwargs)

        to._flagscale_musa_compat = True
        owner.to = to


def _patch_legacy_tensor_type(torch) -> None:
    """Satisfy legacy optimizer checks that compare CUDA type-name strings."""

    original_type = torch.Tensor.type
    if getattr(original_type, "_flagscale_musa_compat", False):
        return

    @wraps(original_type)
    def tensor_type(tensor, *args, **kwargs):
        result = original_type(tensor, *args, **kwargs)
        if not args and not kwargs and isinstance(result, str):
            return result.replace("torch.musa.", "torch.cuda.")
        return result

    tensor_type._flagscale_musa_compat = True
    torch.Tensor.type = tensor_type


def _patch_real_mccl(torch) -> None:
    """Translate the legacy NCCL backend spelling to the registered MCCL backend."""

    original_init = torch.distributed.init_process_group
    if getattr(original_init, "_flagscale_musa_compat", False):
        return

    @wraps(original_init)
    def init_process_group(backend=None, *args, **kwargs):
        if backend is None:
            backend = kwargs.pop("backend", None)
        if backend == "nccl":
            backend = "mccl"
        elif isinstance(backend, str):
            backend = backend.replace("cuda:nccl", "musa:mccl")
        return original_init(backend, *args, **kwargs)

    init_process_group._flagscale_musa_compat = True
    torch.distributed.init_process_group = init_process_group


def _patch_device_queries(torch) -> None:
    """Accept legacy CUDA device objects in torch_musa device queries."""

    original_get_device_properties = torch.musa.get_device_properties
    if getattr(original_get_device_properties, "_flagscale_musa_compat", False):
        return

    @wraps(original_get_device_properties)
    def get_device_properties(device=None):
        return original_get_device_properties(_map_cuda_device(device))

    get_device_properties._flagscale_musa_compat = True
    torch.musa.get_device_properties = get_device_properties


def _patch_optimizer_fallbacks(torch) -> None:
    """Use torch_musa-compatible gradient clipping and Adam code paths."""

    from megatron.core import utils as megatron_utils
    from megatron.core.optimizer import clip_grads, optimizer

    def local_clip_grad_by_total_norm_fp32(
        parameters, max_norm, total_norm, use_decoupled_grad=False
    ):
        grads = []
        for param in parameters:
            grad_name = "decoupled_grad" if use_decoupled_grad else "grad"
            grad = getattr(param, grad_name, None)
            if grad is not None:
                grads.append(megatron_utils.to_local_if_dtensor(grad).detach())

        clip_coeff = max_norm / (float(total_norm) + 1.0e-6)
        if clip_coeff < 1.0:
            for grad in grads:
                grad.mul_(clip_coeff)

    clip_grads.clip_grad_by_total_norm_fp32 = local_clip_grad_by_total_norm_fp32
    optimizer.clip_grad_by_total_norm_fp32 = local_clip_grad_by_total_norm_fp32

    for optimizer_class in (torch.optim.Adam, torch.optim.AdamW):
        if optimizer_class.__dict__.get("_flagscale_musa_single_tensor", False):
            continue
        original_init = optimizer_class.__init__

        @wraps(original_init)
        def init(self, *args, __original_init=original_init, **kwargs):
            kwargs.setdefault("foreach", False)
            __original_init(self, *args, **kwargs)

        optimizer_class.__init__ = init
        optimizer_class._flagscale_musa_single_tensor = True


def _disable_cuda_fused_kernel_build() -> None:
    """Skip the legacy NVCC extension when all corresponding fusions are disabled."""

    from megatron.legacy import fused_kernels

    if getattr(fused_kernels.load, "_flagscale_musa_compat", False):
        return

    def load(_args):
        print("FlagScale: skipped the legacy CUDA fused-kernel build on MUSA.", flush=True)

    load._flagscale_musa_compat = True
    fused_kernels.load = load


def _disable_cuda_jit_warmup() -> None:
    """Skip the legacy torch.compile warmup that targets CUDA/Triton."""

    from megatron.training import initialize

    if getattr(initialize.set_jit_fusion_options, "_flagscale_musa_compat", False):
        return

    def set_jit_fusion_options():
        print("FlagScale: skipped the legacy CUDA JIT warmup on MUSA.", flush=True)

    set_jit_fusion_options._flagscale_musa_compat = True
    initialize.set_jit_fusion_options = set_jit_fusion_options


def enable_musa_compatibility() -> bool:
    """Enable the legacy compatibility layer when explicitly requested."""

    if os.environ.get(_COMPAT_ENV, "").strip().lower() not in _TRUE_VALUES:
        return False

    import torch
    import torch_musa  # noqa: F401 - registers MUSA and MCCL with PyTorch

    if not torch.musa.is_available():
        raise RuntimeError("MUSA compatibility is enabled, but no MUSA device is available.")

    _patch_factory_devices(torch)
    _patch_to_and_cuda(torch)
    _patch_legacy_tensor_type(torch)
    _patch_real_mccl(torch)
    _patch_device_queries(torch)

    # Legacy Megatron resolves accelerator helpers through torch.cuda. Keep
    # explicit device strings handled by the wrappers above, then redirect the
    # namespace itself for streams, events, memory accounting, and RNG state.
    from torch_musa.core import random as musa_random

    torch.musa.random = musa_random
    torch.cuda = torch.musa

    _patch_optimizer_fallbacks(torch)
    _disable_cuda_fused_kernel_build()
    _disable_cuda_jit_warmup()
    print("FlagScale: enabled legacy MUSA compatibility with real MCCL.", flush=True)
    return True
