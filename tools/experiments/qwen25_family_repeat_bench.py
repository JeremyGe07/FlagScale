#!/usr/bin/env python3
"""Repeat fixed Qwen2.5-1.5B strategy runs for A800 validation."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REPEATS = 10
TRAIN_ITERS = 3
NPROC_PER_NODE = 4
GLOBAL_BATCH_SIZE = 32
ITER_RE = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9.]+)")
MAX_RESERVED_RE = re.compile(r"max reserved:\s*([0-9.]+)")


@dataclass(frozen=True)
class StrategySpec:
    label: str
    tp: int
    pp: int
    dist_opt: bool
    sequence_parallel: bool
    micro_batch_size: int
    recompute_method: str | None = None
    recompute_granularity: str | None = None
    recompute_num_layers: int | None = None


def strategy_specs() -> tuple[StrategySpec, ...]:
    return (
        StrategySpec("dp4_tp1_pp1", 1, 1, True, False, 4),
        StrategySpec("dp2_tp1_pp2", 1, 2, True, False, 4),
        StrategySpec("dp1_tp2_pp2_fast", 2, 2, False, True, 8),
        StrategySpec("dp1_tp2_pp2_uniform14", 2, 2, False, True, 16, "uniform", "full", 14),
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


def _common_train_args(run_dir: Path) -> list[str]:
    return [
        "--make-vocab-size-divisible-by", "128", "--disable-bias-linear",
        "--use-flash-attn", "--distributed-timeout-minutes", "60",
        "--bf16", "--attention-softmax-in-fp32", "--accumulate-allreduce-grads-in-fp32",
        "--log-interval", "1", "--tensorboard-log-interval", "1",
        "--wandb-project", "train-qwen2.5-1.5B", "--wandb-exp-name", "train-qwen2.5-1.5B",
        "--tensorboard-dir", str(run_dir / "tensorboard"),
        "--wandb-save-dir", str(run_dir / "wandb"), "--save-interval", "2000",
        "--save", str(run_dir / "checkpoints"), "--load", str(run_dir / "checkpoints"),
        "--context-parallel-size", "1", "--expert-model-parallel-size", "1", "--auto-tune",
        "--use-mcore-models", "--num-layers", "28", "--hidden-size", "1536",
        "--num-attention-heads", "12", "--num-query-groups", "2",
        "--group-query-attention", "--ffn-hidden-size", "8960", "--seq-length", "4096",
        "--max-position-embeddings", "4096", "--norm-epsilon", "1e-06",
        "--norm-init-weight", "0.02", "--use-rotary-position-embeddings",
        "--rotary-base", "1000000", "--no-position-embedding", "--reset-position-ids",
        "--add-qkv-bias", "--reset-attention-mask", "--swiglu", "--normalization", "RMSNorm",
        "--init-method-std", "0.02", "--attention-dropout", "0.0", "--hidden-dropout", "0.0",
        "--weight-decay", "0.0", "--clip-grad", "1.0", "--train-iters", str(TRAIN_ITERS),
        "--eval-iters", "0", "--eval-interval", "100", "--global-batch-size",
        str(GLOBAL_BATCH_SIZE), "--transformer-impl", "transformer_engine", "--seed", "42",
        "--adam-beta1", "0.9", "--adam-beta2", "0.95", "--lr", "1e-05", "--min-lr", "0",
        "--lr-warmup-iters", "2", "--lr-decay-style", "cosine", "--data-path",
        "/home/user-a800/GuoChanZhiSuan/data/pile_wikipedia_demo/pile_wikipedia_demo",
        "--split", "1", "--apply-sft-dataset-separated-loss-mask-if-existed",
        "--legacy-tokenizer", "--tokenizer-type", "Qwen2TokenizerFS", "--tokenizer-path",
        "/home/user-a800/GuoChanZhiSuan/models/Qwen2.5-1.5B", "--vocab-size", "151936",
    ]


def build_train_args(spec: StrategySpec, run_dir: Path) -> list[str]:
    args = [
        "./flagscale/train/train_gpt.py",
        "--tensor-model-parallel-size", str(spec.tp),
        "--pipeline-model-parallel-size", str(spec.pp),
    ]
    if spec.sequence_parallel:
        args.append("--sequence-parallel")
    if spec.dist_opt:
        args.append("--use-distributed-optimizer")
    if spec.recompute_method is not None:
        args += [
            "--recompute-method", spec.recompute_method,
            "--recompute-granularity", str(spec.recompute_granularity),
            "--recompute-num-layers", str(spec.recompute_num_layers),
        ]
    return args + _common_train_args(run_dir) + ["--micro-batch-size", str(spec.micro_batch_size)]


def _torchrun_args(spec: StrategySpec, run_dir: Path) -> list[str]:
    return [
        "torchrun", "--nnodes", "1", "--nproc_per_node", str(NPROC_PER_NODE),
        "--tee", "3", "--redirects", "3", "--rdzv_id", f"{spec.label}_{int(time.time())}",
        "--node_rank", "0", "--rdzv_backend", "c10d",
        "--rdzv_endpoint", f"localhost:{_free_port()}",
        "--log_dir", str(run_dir / "logs" / "details"),
    ] + build_train_args(spec, run_dir)


def _run_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0,1,2,3", "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_SOCKET_IFNAME": "lo", "NCCL_IB_DISABLE": "1", "NCCL_IB_CUDA_SUPPORT": "1",
        "NCCL_IB_GID_INDEX": "0", "NCCL_DEBUG": "INFO", "OMP_NUM_THREADS": "4",
        "GLOO_SOCKET_IFNAME": "lo", "NCCL_IB_HCA": "mlx5_2,mlx5_5", "WANDB_MODE": "offline",
    })
    py_path = f"{repo_root}:{repo_root / 'third_party/Megatron-LM'}"
    env["PYTHONPATH"] = f"{py_path}:{env.get('PYTHONPATH', '')}"
    return env


def _read_text(path: Path) -> str:
    return path.read_text(errors="replace")


def _log_paths(root: Path) -> list[Path]:
    stdout_logs = sorted(root.rglob("stdout.log"))
    if stdout_logs:
        return stdout_logs
    return [path for path in sorted(root.rglob("*")) if path.suffix in {".log", ".output"}]


def _scan_logs(root: Path, pattern: re.Pattern[str]) -> list[float]:
    values = []
    for path in _log_paths(root):
        if path.is_file():
            values.extend(float(match) for match in pattern.findall(_read_text(path)))
    return values


def parse_iteration_performance(log_root: Path) -> float | None:
    timings = _scan_logs(log_root, ITER_RE)
    if len(timings) < 2:
        return None
    return round(sum(timings[1:]) / (len(timings) - 1), 2)


def parse_max_reserved_memory(log_root: Path) -> float | None:
    values = _scan_logs(log_root, MAX_RESERVED_RE)
    return max(values) if values else None


def _shell_command(command: list[str]) -> str:
    quoted = " ".join(shlex.quote(part) for part in command)
    conda_sh = "/home/user-a800/miniconda3/etc/profile.d/conda.sh"
    return f"source {conda_sh} && conda activate flagscale-train && {quoted}"


def _prepare_run_dir(run_dir: Path) -> None:
    for name in ("checkpoints", "logs", "tensorboard", "wandb"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def run_once(spec: StrategySpec, repeat: int, output_root: Path, repo_root: Path) -> dict[str, str]:
    run_dir = output_root / spec.label / f"repeat_{repeat:02d}"
    _prepare_run_dir(run_dir)
    started = time.time()
    command = _torchrun_args(spec, run_dir)
    with (run_dir / "run.log").open("w") as log:
        proc = subprocess.run(
            ["bash", "-lc", _shell_command(command)],
            cwd=repo_root, env=_run_env(repo_root), stdout=log, stderr=subprocess.STDOUT,
        )
    elapsed = round(time.time() - started, 2)
    return {
        "label": spec.label, "repeat": str(repeat), "returncode": str(proc.returncode),
        "performance_ms": str(parse_iteration_performance(run_dir) or ""),
        "max_mem_mb": str(parse_max_reserved_memory(run_dir) or ""),
        "elapsed_s": str(elapsed), "run_dir": str(run_dir),
    }


def write_row(csv_path: Path, row: dict[str, str]) -> None:
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "results.csv"
    for repeat in range(1, args.repeats + 1):
        for spec in strategy_specs():
            row = run_once(spec, repeat, args.output_dir, repo_root)
            write_row(csv_path, row)
            print(row, flush=True)


if __name__ == "__main__":
    main()
