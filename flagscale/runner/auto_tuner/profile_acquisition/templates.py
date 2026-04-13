from dataclasses import dataclass
from typing import Final

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


__all__ = [
    "CalibrationTask",
    "CalibrationTemplate",
    "get_calibration_template",
]
