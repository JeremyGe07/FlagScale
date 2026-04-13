from dataclasses import FrozenInstanceError

import pytest

from flagscale.runner.auto_tuner.profile_acquisition.calibration_runner import (
    expand_calibration_template,
    run_calibration,
)
from flagscale.runner.auto_tuner.profile_acquisition.models import (
    OOM_STATUS,
    OTHER_FAILURE_STATUS,
    SUCCESS_STATUS,
)
from flagscale.runner.auto_tuner.profile_acquisition.templates import (
    get_calibration_template,
)


def test_expand_calibration_template_builds_indexed_readonly_strategies():
    template = get_calibration_template("dense-8")

    strategies = expand_calibration_template(template)

    assert [strategy.strategy_idx for strategy in strategies] == list(range(1, 9))
    assert strategies[0].task_name == "dp2-mbs1"
    assert strategies[-1].task_name == "tp2-mbs8-recompute"
    assert strategies[-1].use_recompute is True

    with pytest.raises(FrozenInstanceError):
        strategies[0].micro_batch_size = 2


def test_run_calibration_returns_summary_patch_and_records(tmp_path):
    template = get_calibration_template("dense-8")
    log_root = tmp_path / "logs"
    expected_order = [task.name for task in template.tasks]
    calls = []
    task_results = {
        "dp2-mbs1": {
            "status": SUCCESS_STATUS,
            "memory_model_mb": 28414.0,
            "max_mem_mb": 33450.0,
            "performance_ms": 9699.35,
            "log_path": str(log_root / "dp2-mbs1.log"),
        },
        "dp2-mbs8": {
            "status": OOM_STATUS,
            "memory_model_mb": 44101.0,
            "max_mem_mb": None,
            "performance_ms": None,
            "log_path": str(log_root / "dp2-mbs8.log"),
            "error": "OOM",
        },
        "tp2-mbs1": {
            "status": SUCCESS_STATUS,
            "memory_model_mb": 37248.0,
            "max_mem_mb": 42388.0,
            "performance_ms": 9829.85,
            "log_path": str(log_root / "tp2-mbs1.log"),
        },
        "tp2-mbs8": {
            "status": SUCCESS_STATUS,
            "memory_model_mb": 43000.0,
            "max_mem_mb": 44150.0,
            "performance_ms": 10010.0,
            "log_path": str(log_root / "tp2-mbs8.log"),
        },
        "pp2-mbs1": {
            "status": SUCCESS_STATUS,
            "memory_model_mb": 20000.0,
            "max_mem_mb": 22000.0,
            "performance_ms": 5100.0,
            "log_path": str(log_root / "pp2-mbs1.log"),
        },
        "pp2-mbs8": {
            "status": OTHER_FAILURE_STATUS,
            "memory_model_mb": 32000.0,
            "max_mem_mb": None,
            "performance_ms": None,
            "log_path": str(log_root / "pp2-mbs8.log"),
            "error": "runner failed",
        },
        "dp2-mbs8-recompute": {
            "status": SUCCESS_STATUS,
            "memory_model_mb": 41000.0,
            "max_mem_mb": 43800.0,
            "performance_ms": 9600.0,
            "log_path": str(log_root / "dp2-mbs8-recompute.log"),
        },
        "tp2-mbs8-recompute": {
            "status": SUCCESS_STATUS,
            "memory_model_mb": 39500.0,
            "max_mem_mb": 42000.0,
            "performance_ms": 9400.0,
            "log_path": str(log_root / "tp2-mbs8-recompute.log"),
        },
    }

    def execute_task(strategy):
        calls.append(strategy.task_name)
        return task_results[strategy.task_name]

    result = run_calibration(
        template=template,
        execute_task=execute_task,
        gpu_memory_mb=46000,
    )

    assert calls == expected_order
    assert len(result.strategies) == 8
    assert len(result.records) == 8
    assert result.summary.template_name == "dense-8"
    assert result.summary.sample_count == 8
    assert result.summary.success_count == 6
    assert result.summary.oom_count == 1
    assert result.summary.other_failure_count == 1
    assert result.summary.oom_recall == 1.0
    assert result.summary.false_prune_count == 0
    assert result.summary.reserved_memory_bias_mb > 1800.0
    assert result.summary.peak_activation_bias_mb > 3000.0
    assert result.records[1].status == OOM_STATUS
    assert result.records[1].log_path.endswith("dp2-mbs8.log")
    assert result.records[5].status == OTHER_FAILURE_STATUS
    assert (
        result.profile_patch["cost_model.reserved_memory_bias_mb"]
        == result.summary.reserved_memory_bias_mb
    )
    assert (
        result.profile_patch["cost_model.peak_activation_bias_mb"]
        == result.summary.peak_activation_bias_mb
    )

    with pytest.raises(FrozenInstanceError):
        result.records[0].status = OOM_STATUS
