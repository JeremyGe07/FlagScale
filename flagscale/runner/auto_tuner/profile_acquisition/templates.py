from dataclasses import dataclass
from pathlib import Path
from typing import Final

from omegaconf import OmegaConf

LOW_MICRO_BATCH_SIZE: Final[int] = 1
HIGH_MICRO_BATCH_SIZE: Final[int] = 8
RECOMPUTE_METHOD: Final[str] = "uniform"
RECOMPUTE_GRANULARITY: Final[str] = "full"
RECOMPUTE_NUM_LAYERS: Final[int] = 1
SHARED_BRANCH_SPECS: Final[tuple[tuple[str, str, tuple[int, int, int], int], ...]] = (
    ("dp2-mbs1", "dp2", (2, 1, 1), LOW_MICRO_BATCH_SIZE),
    ("dp2-mbs8", "dp2", (2, 1, 1), HIGH_MICRO_BATCH_SIZE),
    ("tp2-mbs1", "tp2", (1, 2, 1), LOW_MICRO_BATCH_SIZE),
    ("tp2-mbs8", "tp2", (1, 2, 1), HIGH_MICRO_BATCH_SIZE),
    ("pp2-mbs1", "pp2", (1, 1, 2), LOW_MICRO_BATCH_SIZE),
    ("pp2-mbs8", "pp2", (1, 1, 2), HIGH_MICRO_BATCH_SIZE),
)


@dataclass(frozen=True)
class CalibrationTask:
    name: str
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
    sequence_parallel: bool | None = None
    use_distributed_optimizer: bool | None = None
    context_parallel_size: int | None = None


@dataclass(frozen=True)
class CalibrationTemplate:
    name: str
    tasks: tuple[CalibrationTask, ...]


def _make_task(
    name: str,
    branch: str,
    *,
    dp: int,
    tp: int,
    pp: int,
    micro_batch_size: int,
    use_recompute: bool = False,
    expert_model_parallel_size: int = 1,
) -> CalibrationTask:
    recompute_method = None
    recompute_granularity = None
    recompute_num_layers = None
    if use_recompute:
        recompute_method = RECOMPUTE_METHOD
        recompute_granularity = RECOMPUTE_GRANULARITY
        recompute_num_layers = RECOMPUTE_NUM_LAYERS
    return CalibrationTask(
        name=name,
        branch=branch,
        dp=dp,
        tp=tp,
        pp=pp,
        micro_batch_size=micro_batch_size,
        use_recompute=use_recompute,
        expert_model_parallel_size=expert_model_parallel_size,
        recompute_method=recompute_method,
        recompute_granularity=recompute_granularity,
        recompute_num_layers=recompute_num_layers,
    )


def _shared_branch_tasks() -> tuple[CalibrationTask, ...]:
    return tuple(
        _make_task(
            name,
            branch,
            dp=parallelism[0],
            tp=parallelism[1],
            pp=parallelism[2],
            micro_batch_size=micro_batch_size,
        )
        for name, branch, parallelism, micro_batch_size in SHARED_BRANCH_SPECS
    )


def _build_dense_template() -> CalibrationTemplate:
    return CalibrationTemplate(
        name="dense-8",
        tasks=_shared_branch_tasks()
        + (
            _make_task(
                "dp2-mbs8-recompute",
                "dp2",
                dp=2,
                tp=1,
                pp=1,
                micro_batch_size=HIGH_MICRO_BATCH_SIZE,
                use_recompute=True,
            ),
            _make_task(
                "tp2-mbs8-recompute",
                "tp2",
                dp=1,
                tp=2,
                pp=1,
                micro_batch_size=HIGH_MICRO_BATCH_SIZE,
                use_recompute=True,
            ),
        ),
    )


def _build_moe_template() -> CalibrationTemplate:
    return CalibrationTemplate(
        name="moe-8",
        tasks=_shared_branch_tasks()
        + (
            _make_task(
                "ep2-mbs1",
                "ep2",
                dp=1,
                tp=1,
                pp=1,
                micro_batch_size=LOW_MICRO_BATCH_SIZE,
                expert_model_parallel_size=2,
            ),
            _make_task(
                "ep2-mbs8",
                "ep2",
                dp=1,
                tp=1,
                pp=1,
                micro_batch_size=HIGH_MICRO_BATCH_SIZE,
                expert_model_parallel_size=2,
            ),
        ),
    )


BUILTIN_TEMPLATES: Final[dict[str, CalibrationTemplate]] = {
    "dense-8": _build_dense_template(),
    "moe-8": _build_moe_template(),
}


def get_calibration_template(name: str) -> CalibrationTemplate:
    try:
        return BUILTIN_TEMPLATES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown calibration template: {name}") from exc


def load_calibration_template_file(path: str | Path) -> CalibrationTemplate:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("Calibration template file must contain a mapping.")
    name = _read_required(payload, "name", source="template")
    tasks = _read_required(payload, "tasks", source="template")
    if not isinstance(tasks, list):
        raise ValueError("Calibration template field tasks must be a list.")
    return CalibrationTemplate(
        name=str(name),
        tasks=tuple(_task_from_payload(task, index) for index, task in enumerate(tasks, 1)),
    )


def _task_from_payload(payload: object, index: int) -> CalibrationTask:
    if not isinstance(payload, dict):
        raise ValueError(f"Calibration task {index} must be a mapping.")
    return CalibrationTask(
        name=str(_read_required(payload, "name", source=f"task {index}")),
        branch=str(_read_required(payload, "branch", source=f"task {index}")),
        dp=_read_int(payload, "dp", source=f"task {index}"),
        tp=_read_int(payload, "tp", source=f"task {index}"),
        pp=_read_int(payload, "pp", source=f"task {index}"),
        micro_batch_size=_read_int(payload, "micro_batch_size", source=f"task {index}"),
        use_recompute=bool(_read_required(payload, "use_recompute", source=f"task {index}")),
        expert_model_parallel_size=_read_optional_int(payload, "expert_model_parallel_size", 1),
        recompute_method=_read_optional_str(payload, "recompute_method"),
        recompute_granularity=_read_optional_str(payload, "recompute_granularity"),
        recompute_num_layers=_read_optional_int(payload, "recompute_num_layers"),
        sequence_parallel=_read_optional_bool(payload, "sequence_parallel"),
        use_distributed_optimizer=_read_optional_bool(payload, "use_distributed_optimizer"),
        context_parallel_size=_read_optional_int(payload, "context_parallel_size"),
    )


def _read_required(payload: dict, key: str, *, source: str) -> object:
    if key not in payload:
        raise ValueError(f"Calibration {source} is missing required field: {key}")
    return payload[key]


def _read_int(payload: dict, key: str, *, source: str) -> int:
    return int(_read_required(payload, key, source=source))


def _read_optional_int(payload: dict, key: str, default: int | None = None) -> int | None:
    value = payload.get(key, default)
    if value is None:
        return None
    return int(value)


def _read_optional_str(payload: dict, key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    return str(value)


def _read_optional_bool(payload: dict, key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    return bool(value)


__all__ = [
    "CalibrationTask",
    "CalibrationTemplate",
    "get_calibration_template",
    "load_calibration_template_file",
]
