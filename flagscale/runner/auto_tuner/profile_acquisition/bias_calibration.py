from flagscale.runner.auto_tuner.profile_acquisition.models import (
    MemoryBiasFitResult,
    OOM_STATUS,
    SUCCESS_STATUS,
)

EPSILON_MB = 1.0


def fit_memory_bias(records, gpu_memory_mb, max_false_prunes=0):
    candidates = _build_bias_candidates(records, gpu_memory_mb)
    best = None
    for bias in candidates:
        summary = _evaluate_bias(records, gpu_memory_mb, bias)
        if summary["false_prune_count"] > max_false_prunes:
            continue
        best = _pick_better_result(best, bias, summary)
    if best is None:
        best = (0.0, _evaluate_bias(records, gpu_memory_mb, 0.0))
    bias, summary = best
    peak_bias = _fit_peak_activation_bias(records, reserved_bias_mb=bias)
    return MemoryBiasFitResult(
        reserved_memory_bias_mb=bias,
        peak_activation_bias_mb=peak_bias,
        summary=summary,
    )


def _build_bias_candidates(records, gpu_memory_mb):
    candidates = {0.0}
    for record in records:
        if record["status"] != OOM_STATUS:
            continue
        required = gpu_memory_mb - record["memory_model_mb"] + EPSILON_MB
        candidates.add(max(required, 0.0))
    return sorted(candidates)


def _evaluate_bias(records, gpu_memory_mb, reserved_bias_mb):
    oom_total = 0
    oom_pruned = 0
    false_prunes = 0
    for record in records:
        should_prune = record["memory_model_mb"] + reserved_bias_mb > gpu_memory_mb
        if record["status"] == OOM_STATUS:
            oom_total += 1
            oom_pruned += int(should_prune)
        elif record["status"] == SUCCESS_STATUS and should_prune:
            false_prunes += 1
    oom_recall = float(oom_pruned / oom_total) if oom_total else 0.0
    return {
        "sample_count": len(records),
        "oom_count": oom_total,
        "oom_recall": oom_recall,
        "false_prune_count": false_prunes,
    }


def _pick_better_result(current, bias, summary):
    if current is None:
        return bias, summary
    current_bias, current_summary = current
    current_key = _score_candidate(current_bias, current_summary)
    next_key = _score_candidate(bias, summary)
    if next_key > current_key:
        return bias, summary
    return current


def _score_candidate(bias, summary):
    return (
        summary["oom_recall"],
        -summary["false_prune_count"],
        -bias,
    )


def _fit_peak_activation_bias(records, reserved_bias_mb):
    peak_bias = 0.0
    for record in records:
        if record["status"] != SUCCESS_STATUS:
            continue
        max_mem_mb = record.get("max_mem_mb")
        if max_mem_mb is None:
            continue
        extra = max(max_mem_mb - record["memory_model_mb"] - reserved_bias_mb, 0.0)
        peak_bias = max(peak_bias, extra)
    return peak_bias
