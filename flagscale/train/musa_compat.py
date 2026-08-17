"""Opt-in MUSA compatibility for the legacy Megatron training path.

The legacy stack predates Megatron-LM-FL's device platform abstraction and
still uses CUDA spellings for accelerator APIs.  This module keeps the port
local and explicit: it is enabled only with ``FLAGSCALE_MUSA_COMPAT=1`` and
maps those accelerator calls to torch_musa.  Multi-process jobs still use the
real MCCL backend; no fake process group is installed here.
"""

import os
from contextlib import nullcontext
from functools import wraps


_COMPAT_ENV = "FLAGSCALE_MUSA_COMPAT"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_MUSA_SDPA_LOGGED = False


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
    """Translate NCCL spelling and eagerly initialize the default MCCL group.

    Legacy Megatron's first accelerator collective is a tensor-free barrier in
    ``_compile_dependencies``.  ProcessGroupMCCL must otherwise choose a device
    while lazily creating its communicator, which is unreliable on the S5000
    runtime.  A one-element all-reduce on the rank's current MUSA device binds
    the communicator before any tensor-free barrier is reached.
    """

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
        result = original_init(backend, *args, **kwargs)

        if (
            "mccl" in str(backend).lower()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            device = torch.musa.current_device()
            warmup = torch.ones(1, device=f"musa:{device}")
            torch.distributed.all_reduce(warmup)
            torch.musa.synchronize()
            if torch.distributed.get_rank() == 0:
                print(
                    "FlagScale: eagerly initialized the default MCCL communicator on MUSA.",
                    flush=True,
                )

        return result

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


def _sdpa_dot_product_attention_forward(
    attention,
    query,
    key,
    value,
    attention_mask,
    attn_mask_type=None,
):
    """Run Megatron's local attention layout through PyTorch SDPA.

    ``torch_musa`` dispatches this API to its native FlashAttention forward
    and backward kernels.  Megatron boolean masks use ``True`` for positions
    that must be masked, while PyTorch SDPA uses ``True`` for positions that
    may participate, so boolean masks are inverted here.
    """

    import torch
    import torch.nn.functional as F

    global _MUSA_SDPA_LOGGED
    if not _MUSA_SDPA_LOGGED:
        print(
            "FlagScale: local DotProductAttention is using torch_musa Flash SDPA.",
            flush=True,
        )
        _MUSA_SDPA_LOGGED = True

    # [sk, b, ng, hn] -> [sk, b, np, hn]. Keep the same GQA expansion as
    # Megatron's local implementation because the MUSA SDPA GQA contract is
    # not exposed independently by the legacy configuration.
    heads_per_group = (
        attention.num_attention_heads_per_partition
        // attention.num_query_groups_per_partition
    )
    if heads_per_group > 1:
        key = key.repeat_interleave(heads_per_group, dim=2)
        value = value.repeat_interleave(heads_per_group, dim=2)

    # Megatron: [s, b, h, d]. PyTorch SDPA: [b, h, s, d].
    query = query.permute(1, 2, 0, 3)
    key = key.permute(1, 2, 0, 3)
    value = value.permute(1, 2, 0, 3)

    sdpa_mask = attention_mask
    if sdpa_mask is not None and sdpa_mask.dtype == torch.bool:
        sdpa_mask = ~sdpa_mask

    effective_mask_type = attn_mask_type or attention.attn_mask_type
    is_causal = sdpa_mask is None and effective_mask_type.name == "causal"
    dropout_p = (
        attention.attention_dropout.p if getattr(attention, "training", True) else 0.0
    )

    if attention.config.sequence_parallel or dropout_p == 0.0:
        rng_context = nullcontext()
    else:
        from megatron.core import tensor_parallel

        rng_context = tensor_parallel.get_cuda_rng_tracker().fork()

    with rng_context:
        context = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=sdpa_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=attention.softmax_scale,
        )

    # [b, h, sq, hn] -> [sq, b, hp].
    context = context.permute(2, 0, 1, 3).contiguous()
    return context.view(
        context.size(0), context.size(1), attention.hidden_size_per_partition
    )


def _patch_musa_flash_attention(torch) -> None:
    """Use torch_musa's native Flash SDPA for supported local attention."""

    from megatron.core.transformer.dot_product_attention import DotProductAttention
    from megatron.core.transformer.enums import AttnMaskType

    original_forward = DotProductAttention.forward
    if getattr(original_forward, "_flagscale_musa_compat", False):
        return

    supported_mask_types = {
        AttnMaskType.padding,
        AttnMaskType.causal,
        AttnMaskType.no_mask,
        AttnMaskType.padding_causal,
        AttnMaskType.arbitrary,
    }

    @wraps(original_forward)
    def forward(
        attention,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
    ):
        effective_mask_type = attn_mask_type or attention.attn_mask_type
        can_use_sdpa = (
            query.device.type == "musa"
            and packed_seq_params is None
            and attention_bias is None
            and attention.softmax_offset is None
            and attention.config.window_size is None
            and effective_mask_type in supported_mask_types
        )
        if can_use_sdpa:
            return _sdpa_dot_product_attention_forward(
                attention,
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=effective_mask_type,
            )
        return original_forward(
            attention,
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )

    forward._flagscale_musa_compat = True
    forward._flagscale_original_forward = original_forward
    DotProductAttention.forward = forward


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
    _patch_musa_flash_attention(torch)
    _disable_cuda_fused_kernel_build()
    _disable_cuda_jit_warmup()
    print(
        "FlagScale: enabled legacy MUSA compatibility with real MCCL and native Flash SDPA.",
        flush=True,
    )
    return True
