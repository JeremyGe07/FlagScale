#!/usr/bin/env python3
"""Replay MoE time-cost ranking against historical AutoTune rows."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

FAMILY_DIMS = (
    "data_parallel_size",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "sequence_parallel",
    "use_distributed_optimizer",
)
FULL_DIMS = (
    "data_parallel_size",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "expert_model_parallel_size",
    "context_parallel_size",
    "decoder_first_pipeline_num_layers",
    "decoder_last_pipeline_num_layers",
    "use_distributed_optimizer",
    "sequence_parallel",
    "acc_step",
    "micro_batch_size",
    "num_layers_per_virtual_pipeline_stage",
    "use_recompute",
    "recompute_method",
    "recompute_granularity",
    "recompute_num_layers",
)
DEFAULT_TOPKS = (4, 8, 9, 16, 32)


def normalize_history_value(value):
    if isinstance(value, bool) or value is None:
        return value
    if value in ("", "None"):
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() else number


def best_performance(rows):
    valid = [row for row in rows if _is_success(row)]
    if not valid:
        return None
    return min(valid, key=lambda row: float(row["performance"]))


def topk_recall(rows, topks):
    result = {}
    for topk in topks:
        kept = [
            float(row["performance"])
            for row in rows
            if _is_success(row) and int(row["rank"]) <= topk
        ]
        result[topk] = {
            "successes_kept": len(kept),
            "best_kept": min(kept) if kept else None,
        }
    return result


def read_history(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def match_history_to_ranks(rows, rank_by_key):
    matched = []
    for row in rows:
        rank_info = rank_by_key.get(_row_key(row, FULL_DIMS))
        if rank_info is None:
            continue
        matched.append({**row, **rank_info})
    return matched


def rank_strategy_families(strategies, gpu_memory):
    groups = defaultdict(list)
    for strategy in strategies:
        if not _is_memory_feasible(strategy, gpu_memory):
            continue
        groups[_strategy_key(strategy, FAMILY_DIMS)].append(strategy)
    return _rank_grouped_strategies(groups)


def fastest_ep_candidate(strategies, gpu_memory):
    candidates = [
        strategy
        for strategy in strategies
        if strategy.get("expert_model_parallel_size", 1) > 1
        and _is_memory_feasible(strategy, gpu_memory)
    ]
    if not candidates:
        return None
    return min(candidates, key=_time_cost_sort_key)


def parse_topks(raw_value):
    if raw_value is None:
        return DEFAULT_TOPKS
    values = tuple(int(item.strip()) for item in raw_value.split(",") if item.strip())
    if not values or any(value < 1 for value in values):
        raise ValueError("--topks must contain positive integers.")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--history-csv", required=True)
    parser.add_argument("--chip-profile-path")
    parser.add_argument("--moe-time-cost-model")
    parser.add_argument("--moe-time-cost-option", action="append", default=[])
    parser.add_argument("--topks")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    topks = parse_topks(args.topks)
    config = _compose_config(args)
    searcher = _build_searcher(config)
    rows = read_history(args.history_csv)
    gpu_memory = _gpu_memory(config)
    rank_by_key = rank_strategy_families(searcher.strategies, gpu_memory)
    matched = match_history_to_ranks(rows, rank_by_key)
    _print_summary(searcher.strategies, rows, matched, topks, gpu_memory)
    return 0


def _is_success(row):
    return bool(row.get("performance")) and not row.get("error")


def _row_key(row, dims):
    return tuple((dim, normalize_history_value(row.get(dim, ""))) for dim in dims)


def _strategy_key(strategy, dims):
    return tuple((dim, normalize_history_value(strategy.get(dim))) for dim in dims)


def _is_memory_feasible(strategy, gpu_memory):
    memory = strategy.get("memory_model")
    if memory is None:
        return False
    return gpu_memory is None or memory <= gpu_memory


def _rank_grouped_strategies(groups):
    rank_by_key = {}
    for family, strategies in groups.items():
        ranked = sorted(strategies, key=_time_cost_sort_key)
        for rank, strategy in enumerate(ranked, start=1):
            rank_by_key[_strategy_key(strategy, FULL_DIMS)] = {
                "rank": rank,
                "family_size": len(ranked),
                "family": dict(family),
                "time_cost": strategy["time_cost"],
            }
    return rank_by_key


def _time_cost_sort_key(strategy):
    return (
        strategy["time_cost"],
        -strategy.get("chip_score", float("-inf")),
        -strategy.get("memory_model", float("-inf")),
    )


def _compose_config(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    overrides = hydra_overrides(args)
    config_dir = os.path.abspath(args.config_path)
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name=args.config_name, overrides=overrides)
    OmegaConf.set_struct(config, False)
    return config


def hydra_overrides(args):
    overrides = []
    if args.chip_profile_path:
        overrides.append(
            f"experiment.auto_tuner.chip_profile.path={args.chip_profile_path}"
        )
    if args.moe_time_cost_model:
        overrides.append(
            f"++experiment.auto_tuner.algo.moe_time_cost_model={args.moe_time_cost_model}"
        )
    overrides.extend(_option_overrides(args.moe_time_cost_option))
    return overrides


def _option_overrides(raw_options):
    overrides = []
    for raw_option in raw_options:
        key, value = _split_key_value(raw_option)
        overrides.append(
            "++experiment.auto_tuner.algo.moe_time_cost_model_options."
            f"{key}={value}"
        )
    return overrides


def _split_key_value(raw_value):
    if "=" not in raw_value:
        raise ValueError("--moe-time-cost-option entries must use key=value.")
    key, value = raw_value.split("=", 1)
    if not key or not value:
        raise ValueError("--moe-time-cost-option entries must use key=value.")
    return key, value


def _build_searcher(config):
    from flagscale.runner.auto_tuner.search.searcher import Searcher

    return Searcher(config)


def _gpu_memory(config):
    memory_model = config.experiment.auto_tuner.get("memory_model", {})
    return memory_model.get("gpu_memory")


def _print_summary(strategies, rows, matched, topks, gpu_memory):
    best = best_performance(matched)
    print(f"strategies={len(strategies)}")
    print(f"history_rows={len(rows)}")
    print(f"matched_rows={len(matched)}")
    print(f"success_rows={sum(1 for row in matched if _is_success(row))}")
    _print_best(best)
    _print_topk_recall(matched, topks)
    _print_fastest_ep(strategies, gpu_memory)


def _print_best(best):
    if best is None:
        print("best_performance=None")
        return
    print(f"best_performance={best['performance']}")
    print(f"best_rank={best['rank']}/{best['family_size']}")
    print(f"best_time_cost={float(best['time_cost']):.3f}")
    print(f"best_family={json.dumps(best['family'], sort_keys=True)}")


def _print_topk_recall(rows, topks):
    for topk, item in topk_recall(rows, topks).items():
        print(
            f"topk={topk} successes_kept={item['successes_kept']} "
            f"best_kept={item['best_kept']}"
        )


def _print_fastest_ep(strategies, gpu_memory):
    candidate = fastest_ep_candidate(strategies, gpu_memory)
    if candidate is None:
        print("fastest_ep_candidate=None")
        return
    print(f"fastest_ep_time_cost={candidate['time_cost']:.3f}")
    print(f"fastest_ep_strategy={_strategy_summary(candidate)}")


def _strategy_summary(strategy):
    fields = ("data_parallel_size", "tensor_model_parallel_size", "pipeline_model_parallel_size")
    base = ",".join(f"{field}={strategy[field]}" for field in fields)
    return f"{base},expert_model_parallel_size={strategy['expert_model_parallel_size']}"


if __name__ == "__main__":
    raise SystemExit(main())
