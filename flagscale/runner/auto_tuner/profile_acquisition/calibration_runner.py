from collections.abc import Callable, Mapping
from types import MappingProxyType

from flagscale.runner.auto_tuner.profile_acquisition.bias_calibration import (
    fit_memory_bias,
)
from flagscale.runner.auto_tuner.profile_acquisition.models import (
    CalibrationRecord,
    CalibrationResult,
    CalibrationStrategy,
    CalibrationSummary,
    ERROR_STATUS,
    MemoryBiasFitResult,
    OTHER_FAILURE_STATUS,
    OOM_STATUS,
    SUCCESS_STATUS,
)
from flagscale.runner.auto_tuner.profile_acquisition.templates import (
    CalibrationTask,
    CalibrationTemplate,
)

TaskExecutor = Callable[[CalibrationStrategy], Mapping[str, object]]
ALLOWED_INPUT_STATUSES = frozenset(
    {SUCCESS_STATUS, OOM_STATUS, ERROR_STATUS, OTHER_FAILURE_STATUS}
)


def expand_calibration_template(
    template: CalibrationTemplate,
) -> tuple[CalibrationStrategy, ...]:
    return tuple(
        _build_strategy(strategy_idx=index, task=task)
        for index, task in enumerate(template.tasks, start=1)
    )


def run_calibration(
    *,
    template: CalibrationTemplate,
    execute_task: TaskExecutor,
    gpu_memory_mb: float,
    max_false_prunes: int = 0,
) -> CalibrationResult:
    strategies = expand_calibration_template(template)
    records = tuple(_execute_strategy(strategy, execute_task) for strategy in strategies)
    bias_fit = fit_memory_bias(
        _to_bias_records(records),
        gpu_memory_mb=gpu_memory_mb,
        max_false_prunes=max_false_prunes,
    )
    summary = _build_summary(template.name, records, bias_fit)
    return CalibrationResult(
        template_name=template.name,
        strategies=strategies,
        records=records,
        summary=summary,
        profile_patch=_build_profile_patch(summary),
    )


def _build_strategy(*, strategy_idx: int, task: CalibrationTask) -> CalibrationStrategy:
    return CalibrationStrategy(
        strategy_idx=strategy_idx,
        task_name=task.name,
        branch=task.branch,
        dp=task.dp,
        tp=task.tp,
        pp=task.pp,
        micro_batch_size=task.micro_batch_size,
        use_recompute=task.use_recompute,
        expert_model_parallel_size=task.expert_model_parallel_size,
        recompute_method=task.recompute_method,
        recompute_granularity=task.recompute_granularity,
        recompute_num_layers=task.recompute_num_layers,
    )


def _execute_strategy(
    strategy: CalibrationStrategy, execute_task: TaskExecutor
) -> CalibrationRecord:
    payload = execute_task(strategy)
    return CalibrationRecord(
        strategy_idx=strategy.strategy_idx,
        task_name=strategy.task_name,
        branch=strategy.branch,
        status=_normalize_status(str(payload["status"])),
        memory_model_mb=float(payload["memory_model_mb"]),
        max_mem_mb=_read_float(payload.get("max_mem_mb")),
        performance_ms=_read_float(payload.get("performance_ms")),
        log_path=str(payload.get("log_path") or ""),
        error=str(payload.get("error") or ""),
    )


def _normalize_status(status: str) -> str:
    if status not in ALLOWED_INPUT_STATUSES:
        raise ValueError(f"Unknown calibration task status: {status}")
    if status == ERROR_STATUS:
        return OTHER_FAILURE_STATUS
    return status


def _read_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _to_bias_records(records: tuple[CalibrationRecord, ...]) -> list[dict]:
    return [
        {
            "strategy_idx": record.strategy_idx,
            "status": record.status,
            "memory_model_mb": record.memory_model_mb,
            "max_mem_mb": record.max_mem_mb,
            "performance_ms": record.performance_ms,
            "error": record.error,
        }
        for record in records
    ]


def _build_summary(
    template_name: str,
    records: tuple[CalibrationRecord, ...],
    bias_fit: MemoryBiasFitResult,
) -> CalibrationSummary:
    fit_summary = bias_fit.summary
    return CalibrationSummary(
        template_name=template_name,
        sample_count=len(records),
        success_count=_count_records(records, SUCCESS_STATUS),
        oom_count=_count_records(records, OOM_STATUS),
        other_failure_count=_count_records(records, OTHER_FAILURE_STATUS),
        oom_recall=float(fit_summary["oom_recall"]),
        false_prune_count=int(fit_summary["false_prune_count"]),
        reserved_memory_bias_mb=bias_fit.reserved_memory_bias_mb,
        peak_activation_bias_mb=bias_fit.peak_activation_bias_mb,
    )


def _count_records(records: tuple[CalibrationRecord, ...], status: str) -> int:
    return sum(record.status == status for record in records)


def _build_profile_patch(summary: CalibrationSummary) -> Mapping[str, float]:
    return MappingProxyType(
        {
            "cost_model.reserved_memory_bias_mb": summary.reserved_memory_bias_mb,
            "cost_model.peak_activation_bias_mb": summary.peak_activation_bias_mb,
        }
    )
