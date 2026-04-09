import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

BITS_PER_BYTE = 8.0
GIGA = 1e9
MICROSECONDS = 1e6
DEFAULT_WARMUP_ITERS = 10
DEFAULT_ITERS = 20
DEFAULT_SMALL_BYTES = 8
DEFAULT_LARGE_BYTES = 64 * 1024 * 1024
CUDA_DEVICE = "cuda"
P2P_COLLECTIVE = "p2p"
ALL_REDUCE_COLLECTIVE = "all_reduce"
NCCL_BACKEND = "nccl"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Measure collective latency and bandwidth.")
    parser.add_argument(
        "--collective",
        required=True,
        choices=[P2P_COLLECTIVE, ALL_REDUCE_COLLECTIVE],
    )
    parser.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP_ITERS)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--small-bytes", type=int, default=DEFAULT_SMALL_BYTES)
    parser.add_argument("--large-bytes", type=int, default=DEFAULT_LARGE_BYTES)
    return parser.parse_args(argv)


def main(argv=None, stdout=None):
    args = parse_args(argv)
    metrics = measure_collective(args)
    write_metrics(metrics, stdout or sys.stdout)
    return 0


def measure_collective(args):
    _init_nccl()
    try:
        if args.collective == P2P_COLLECTIVE:
            return _measure_p2p(args)
        return _measure_all_reduce(args)
    finally:
        _destroy_process_group()


def write_metrics(metrics, stdout):
    for key in ["bandwidth_gbps", "latency_us"]:
        stdout.write(f"{key}={metrics[key]}\n")


def _init_nccl():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for torch NCCL benchmark.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=NCCL_BACKEND)
    torch.cuda.synchronize()


def _destroy_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()


def _measure_p2p(args):
    _require_world_size(2, P2P_COLLECTIVE)
    latency_s = _average_seconds(_run_p2p_once, args.small_bytes, args.warmup_iters, args.iters)
    bandwidth_s = _average_seconds(_run_p2p_once, args.large_bytes, args.warmup_iters, args.iters)
    return {
        "bandwidth_gbps": _bytes_to_gbps(args.large_bytes, bandwidth_s),
        "latency_us": latency_s * MICROSECONDS,
    }


def _measure_all_reduce(args):
    latency_s = _average_seconds(
        _run_all_reduce_once,
        args.small_bytes,
        args.warmup_iters,
        args.iters,
    )
    bandwidth_s = _average_seconds(
        _run_all_reduce_once,
        args.large_bytes,
        args.warmup_iters,
        args.iters,
    )
    return {
        "bandwidth_gbps": _all_reduce_bandwidth(args.large_bytes, bandwidth_s),
        "latency_us": latency_s * MICROSECONDS,
    }


def _average_seconds(step, size_bytes, warmup_iters, measure_iters):
    _run_iters(step, size_bytes, warmup_iters)
    dist.barrier()
    torch.cuda.synchronize()
    start = time.perf_counter()
    _run_iters(step, size_bytes, measure_iters)
    torch.cuda.synchronize()
    avg_seconds = (time.perf_counter() - start) / measure_iters
    return _max_seconds_across_ranks(avg_seconds)


def _run_iters(step, size_bytes, iters):
    for _ in range(iters):
        step(size_bytes)


def _run_p2p_once(size_bytes):
    tensor = torch.empty(size_bytes, dtype=torch.uint8, device=CUDA_DEVICE)
    peer = 1 - dist.get_rank()
    ops = [_build_p2p_op(tensor, peer)]
    for request in dist.batch_isend_irecv(ops):
        request.wait()


def _build_p2p_op(tensor, peer):
    if dist.get_rank() == 0:
        return dist.P2POp(dist.isend, tensor, peer)
    return dist.P2POp(dist.irecv, tensor, peer)


def _run_all_reduce_once(size_bytes):
    tensor = torch.ones(size_bytes, dtype=torch.uint8, device=CUDA_DEVICE)
    dist.all_reduce(tensor)


def _max_seconds_across_ranks(avg_seconds):
    value = torch.tensor([avg_seconds], dtype=torch.float64, device=CUDA_DEVICE)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def _bytes_to_gbps(size_bytes, avg_seconds):
    return (size_bytes * BITS_PER_BYTE) / avg_seconds / GIGA


def _all_reduce_bandwidth(size_bytes, avg_seconds):
    world_size = dist.get_world_size()
    effective_bytes = 2.0 * (world_size - 1) / world_size * size_bytes
    return _bytes_to_gbps(effective_bytes, avg_seconds)


def _require_world_size(expected, collective):
    world_size = dist.get_world_size()
    if world_size == expected:
        return
    raise RuntimeError(f"{collective} benchmark requires world_size={expected}, got {world_size}.")


if __name__ == "__main__":
    raise SystemExit(main())
