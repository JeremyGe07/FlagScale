import csv
import json
import re

from flagscale.runner.auto_tuner.profile_acquisition.models import (
    ERROR_STATUS,
    OOM_STATUS,
    SUCCESS_STATUS,
)

OOM_PREFIX = "OOM"
ALLOCATED_MB_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?) GiB is allocated by PyTorch")
RESERVED_MB_PATTERN = re.compile(
    r"([0-9]+(?:\.[0-9]+)?) GiB is reserved by PyTorch but unallocated"
)
TASK_ERROR_PATTERN = re.compile(r"task_(\d+) error: (.*)")
BYTES_PER_GIB_MB = 1024.0


def build_acquisition_dataset(history_csv_path, tuner_log_path):
    oom_details = _parse_tuner_oom_details(tuner_log_path)
    records = []
    for row in _read_history_rows(history_csv_path):
        records.append(_build_record(row, oom_details.get(int(row["idx"]))))
    return records


def _read_history_rows(history_csv_path):
    with open(history_csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _build_record(row, oom_detail):
    memory_model_mb = _read_float(row.get("memory_model"))
    max_mem_mb = _read_float(row.get("max_mem"))
    performance_ms = _read_float(row.get("performance"))
    error_text = str(row.get("error") or "")
    status = _infer_status(max_mem_mb, performance_ms, error_text, oom_detail)
    return {
        "strategy_idx": int(row["idx"]),
        "status": status,
        "memory_model_mb": memory_model_mb,
        "max_mem_mb": max_mem_mb,
        "performance_ms": performance_ms,
        "memory_breakdown": _parse_json_field(row.get("memory_breakdown")),
        "error": error_text,
        "oom_detail": oom_detail or {},
    }


def _infer_status(max_mem_mb, performance_ms, error_text, oom_detail):
    if oom_detail or "OOM" in error_text:
        return OOM_STATUS
    if max_mem_mb is not None and performance_ms is not None:
        return SUCCESS_STATUS
    return ERROR_STATUS


def _parse_json_field(value):
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def _read_float(value):
    if value in (None, "", "OOM"):
        return None
    return float(value)


def _parse_tuner_oom_details(tuner_log_path):
    details = {}
    with open(tuner_log_path) as handle:
        for line in handle:
            parsed = _parse_task_error_line(line)
            if parsed is None:
                continue
            task_idx, detail = parsed
            if detail is not None:
                details[task_idx] = detail
    return details


def _parse_task_error_line(line):
    match = TASK_ERROR_PATTERN.search(line)
    if match is None:
        return None
    task_idx = int(match.group(1))
    error_text = match.group(2)
    if OOM_PREFIX not in error_text:
        return task_idx, None
    return task_idx, _extract_oom_metrics(error_text)


def _extract_oom_metrics(error_text):
    detail = {"raw_error": error_text}
    allocated_mb = _extract_gib_to_mb(ALLOCATED_MB_PATTERN, error_text)
    reserved_mb = _extract_gib_to_mb(RESERVED_MB_PATTERN, error_text)
    if allocated_mb is not None:
        detail["allocated_mb"] = allocated_mb
    if reserved_mb is not None:
        detail["reserved_unallocated_mb"] = reserved_mb
    return detail


def _extract_gib_to_mb(pattern, text):
    match = pattern.search(text)
    if match is None:
        return None
    return round(float(match.group(1)) * BYTES_PER_GIB_MB, 2)
