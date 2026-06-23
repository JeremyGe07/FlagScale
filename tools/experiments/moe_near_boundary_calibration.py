import argparse
import csv

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

SUCCESS_STATUS = "success"
OOM_STATUS = "oom"
OTHER_STATUS = "other"
OOM_MARKERS = (
    "out of memory",
    "cuda out of memory",
    "ncclunhandledcudaerror",
    "cuda failure 2",
)
DEFAULT_SAMPLE_COUNT = 12
DEFAULT_PER_FAMILY_LIMIT = 2


@dataclass(frozen=True)
class HistoryCandidate:
    idx: int
    dp: int
    tp: int
    pp: int
    ep: int
    cp: int
    use_dist: bool
    sp: bool
    mbs: int
    use_recompute: bool
    recompute_method: str | None
    recompute_granularity: str | None
    recompute_num_layers: int | None
    memory_model: float
    max_mem: float | None
    status: str


def main(argv=None):
    args = _parse_args(argv)
    rows = _load_history_rows(args.history_csv)
    template = build_near_boundary_template(
        rows,
        template_name=args.template_name,
        sample_count=args.sample_count,
        gpu_memory_mb=args.gpu_memory_mb,
        per_family_limit=args.per_family_limit,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(template), f=output)
    print(f"Wrote {len(template['tasks'])} calibration tasks to {output}")
    return 0


def build_near_boundary_template(
    rows,
    *,
    template_name: str,
    sample_count: int,
    gpu_memory_mb: float,
    per_family_limit: int = DEFAULT_PER_FAMILY_LIMIT,
):
    candidates = _valid_candidates(rows)
    selected = _select_candidates(
        candidates,
        sample_count=sample_count,
        gpu_memory_mb=gpu_memory_mb,
        per_family_limit=per_family_limit,
    )
    return {"name": template_name, "tasks": [_candidate_to_task(item) for item in selected]}


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Build a near-boundary calibration template.")
    parser.add_argument("--history-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--template-name", required=True)
    parser.add_argument("--gpu-memory-mb", type=float, required=True)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--per-family-limit", type=int, default=DEFAULT_PER_FAMILY_LIMIT)
    return parser.parse_args(argv)


def _load_history_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _valid_candidates(rows) -> list[HistoryCandidate]:
    candidates = []
    for row in rows:
        candidate = _candidate_from_row(row)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _candidate_from_row(row) -> HistoryCandidate | None:
    try:
        return HistoryCandidate(
            idx=_read_int(row, "idx"),
            dp=_read_int(row, "data_parallel_size"),
            tp=_read_int(row, "tensor_model_parallel_size"),
            pp=_read_int(row, "pipeline_model_parallel_size"),
            ep=_read_int(row, "expert_model_parallel_size", default=1),
            cp=_read_int(row, "context_parallel_size", default=1),
            use_dist=_read_bool(row, "use_distributed_optimizer"),
            sp=_read_bool(row, "sequence_parallel"),
            mbs=_read_int(row, "micro_batch_size"),
            use_recompute=_read_bool(row, "use_recompute"),
            recompute_method=_read_optional_str(row, "recompute_method"),
            recompute_granularity=_read_optional_str(row, "recompute_granularity"),
            recompute_num_layers=_read_optional_int(row, "recompute_num_layers"),
            memory_model=_read_float(row, "memory_model"),
            max_mem=_read_optional_float(row, "max_mem"),
            status=_classify_status(row),
        )
    except (TypeError, ValueError):
        return None


def _select_candidates(candidates, *, sample_count, gpu_memory_mb, per_family_limit):
    selected = []
    informative = [item for item in candidates if item.status in (SUCCESS_STATUS, OOM_STATUS)]
    selected.extend(
        _select_from_ranked_families(
            informative,
            sample_count=sample_count,
            gpu_memory_mb=gpu_memory_mb,
            per_family_limit=per_family_limit,
        )
    )
    if len(selected) >= sample_count:
        return selected[:sample_count]
    selected_ids = {item.idx for item in selected}
    fallback = [item for item in candidates if item.idx not in selected_ids]
    selected.extend(
        _select_from_ranked_families(
            fallback,
            sample_count=sample_count - len(selected),
            gpu_memory_mb=gpu_memory_mb,
            per_family_limit=per_family_limit,
        )
    )
    return selected


def _select_from_ranked_families(candidates, *, sample_count, gpu_memory_mb, per_family_limit):
    selected = []
    for family in _ranked_families(candidates, gpu_memory_mb):
        for candidate in family[:per_family_limit]:
            selected.append(candidate)
            if len(selected) >= sample_count:
                return selected
    return selected


def _ranked_families(candidates, gpu_memory_mb):
    families = {}
    for candidate in candidates:
        families.setdefault(_family_key(candidate), []).append(candidate)
    ranked = [
        sorted(items, key=lambda item: (_memory_distance(item, gpu_memory_mb), item.idx))
        for items in families.values()
    ]
    return sorted(ranked, key=lambda items: (_memory_distance(items[0], gpu_memory_mb), items[0].idx))


def _candidate_to_task(candidate: HistoryCandidate):
    return {
        "name": f"history-{candidate.idx}",
        "branch": _family_key(candidate),
        "dp": candidate.dp,
        "tp": candidate.tp,
        "pp": candidate.pp,
        "expert_model_parallel_size": candidate.ep,
        "context_parallel_size": candidate.cp,
        "sequence_parallel": candidate.sp,
        "use_distributed_optimizer": candidate.use_dist,
        "micro_batch_size": candidate.mbs,
        "use_recompute": candidate.use_recompute,
        "recompute_method": candidate.recompute_method,
        "recompute_granularity": candidate.recompute_granularity,
        "recompute_num_layers": candidate.recompute_num_layers,
        "source_history_idx": candidate.idx,
        "historical_status": candidate.status,
        "historical_memory_model_mb": candidate.memory_model,
        "historical_max_mem_mb": candidate.max_mem,
    }


def _family_key(candidate: HistoryCandidate):
    sp = str(candidate.sp).lower()
    dist = str(candidate.use_dist).lower()
    return f"dp{candidate.dp}_tp{candidate.tp}_pp{candidate.pp}_ep{candidate.ep}_sp{sp}_dist{dist}"


def _memory_distance(candidate: HistoryCandidate, gpu_memory_mb: float):
    boundary_memory = max(candidate.memory_model, candidate.max_mem or 0.0)
    return abs(gpu_memory_mb - boundary_memory)


def _classify_status(row):
    if _read_optional_float(row, "performance") is not None:
        return SUCCESS_STATUS
    error = str(row.get("error") or "").lower()
    if any(marker in error for marker in OOM_MARKERS):
        return OOM_STATUS
    return OTHER_STATUS


def _read_int(row, key, default=None):
    value = row.get(key, default)
    if value in (None, ""):
        raise ValueError(f"Missing integer field: {key}")
    return int(float(value))


def _read_float(row, key):
    value = row.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing float field: {key}")
    return float(value)


def _read_optional_float(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def _read_optional_int(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    return int(float(value))


def _read_optional_str(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    return str(value)


def _read_bool(row, key):
    value = str(row.get(key) or "").strip().lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    raise ValueError(f"Invalid boolean field {key}: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
