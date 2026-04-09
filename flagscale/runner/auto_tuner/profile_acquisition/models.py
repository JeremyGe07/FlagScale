from dataclasses import dataclass, field

OOM_STATUS = "oom"
SUCCESS_STATUS = "success"
ERROR_STATUS = "error"


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
