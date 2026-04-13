from dataclasses import FrozenInstanceError

import pytest

from flagscale.runner.auto_tuner.profile_acquisition.templates import (
    get_calibration_template,
)


def test_get_builtin_dense_template_returns_expected_tasks():
    template = get_calibration_template("dense-8")

    assert template.name == "dense-8"
    assert len(template.tasks) == 8
    assert {(task.dp, task.tp, task.pp) for task in template.tasks} == {
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
    }
    assert {task.micro_batch_size for task in template.tasks} == {1, 8}
    assert {task.use_recompute for task in template.tasks} == {False, True}


def test_template_and_tasks_are_readonly():
    template = get_calibration_template("dense-8")

    with pytest.raises(FrozenInstanceError):
        template.name = "other"

    with pytest.raises(FrozenInstanceError):
        template.tasks[0].micro_batch_size = 2

    with pytest.raises(AttributeError):
        template.tasks.append(template.tasks[0])


def test_get_builtin_moe_template_reuses_shared_branches_and_adds_ep_tasks():
    template = get_calibration_template("moe-8")

    assert template.name == "moe-8"
    assert len(template.tasks) == 8
    assert {task.branch for task in template.tasks} == {
        "dp2",
        "tp2",
        "pp2",
        "ep2",
    }
    assert {task.expert_model_parallel_size for task in template.tasks} == {1, 2}
    assert any(task.expert_model_parallel_size > 1 for task in template.tasks)
    assert any(task.expert_model_parallel_size == 1 for task in template.tasks)
    assert any(task.branch == "ep2" and task.micro_batch_size == 8 for task in template.tasks)


def test_get_calibration_template_rejects_unknown_template_name():
    with pytest.raises(KeyError, match="unknown-template"):
        get_calibration_template("unknown-template")
