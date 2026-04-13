from dataclasses import dataclass, field
from typing import Mapping

OOM_STATUS = "oom"
SUCCESS_STATUS = "success"
ERROR_STATUS = "error"
OTHER_FAILURE_STATUS = "other_failure"


@dataclass(frozen=True)
class AcquisitionMeasurement:
    kind: str
    source: str
    metrics: dict
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryBiasFitResult:
    reserved_memory_bias_mb: float
    peak_activation_bias_mb: float
    summary: dict


@dataclass(frozen=True)
class CalibrationStrategy:
    strategy_idx: int
    task_name: str
    branch: str
    dp: int
    tp: int
    pp: int
    micro_batch_size: int
    use_recompute: bool
    expert_model_parallel_size: int = 1
    recompute_method: str | None = None
    recompute_granularity: str | None = None
    recompute_num_layers: int | None = None


@dataclass(frozen=True)
class CalibrationRecord:
    strategy_idx: int
    task_name: str
    branch: str
    status: str
    memory_model_mb: float
    max_mem_mb: float | None
    performance_ms: float | None
    log_path: str
    error: str = ""


@dataclass(frozen=True)
class CalibrationSummary:
    template_name: str
    sample_count: int
    success_count: int
    oom_count: int
    other_failure_count: int
    oom_recall: float
    false_prune_count: int
    reserved_memory_bias_mb: float
    peak_activation_bias_mb: float


@dataclass(frozen=True)
class CalibrationResult:
    template_name: str
    strategies: tuple[CalibrationStrategy, ...]
    records: tuple[CalibrationRecord, ...]
    summary: CalibrationSummary
    profile_patch: Mapping[str, float]
