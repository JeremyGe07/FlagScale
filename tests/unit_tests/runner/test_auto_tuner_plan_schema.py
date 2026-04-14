from dataclasses import FrozenInstanceError
import importlib
import sys

import pytest


def _import_plan_module():
    sys.modules.pop("flagscale.runner.auto_tuner.plan", None)
    sys.modules.pop("flagscale.runner.auto_tuner", None)
    sys.modules.pop("flagscale.runner.auto_tuner.tuner", None)
    return importlib.import_module("flagscale.runner.auto_tuner.plan")


def _load_plan_types():
    plan_module = _import_plan_module()
    return (
        plan_module.ExecutionContract,
        plan_module.ModelPlan,
        plan_module.SegmentPlan,
        plan_module.StagePlan,
        plan_module.TransitionPlan,
    )


def test_importing_plan_module_does_not_load_tuner():
    _import_plan_module()

    assert "flagscale.runner.auto_tuner.tuner" not in sys.modules


def test_model_plan_accepts_multi_segment_stage():
    _, ModelPlan, SegmentPlan, StagePlan, _ = _load_plan_types()
    plan = ModelPlan(
        stages=[
            StagePlan(
                stage_id=0,
                segments=[
                    SegmentPlan(start=0, end=3, strategy={"tp": 1, "dp": 2}),
                    SegmentPlan(start=4, end=7, strategy={"tp": 2, "dp": 1}),
                ],
            )
        ]
    )

    assert len(plan.stages) == 1
    assert len(plan.stages[0].segments) == 2
    assert plan.stages[0].segments[1].strategy["tp"] == 2


def test_plan_schema_is_readonly_after_construction():
    _, ModelPlan, SegmentPlan, StagePlan, _ = _load_plan_types()
    plan = ModelPlan(
        stages=[
            StagePlan(
                stage_id=0,
                segments=[SegmentPlan(start=0, end=1, strategy={"tp": 1, "dp": 1})],
            )
        ]
    )

    with pytest.raises(FrozenInstanceError):
        plan.stages[0].stage_id = 1

    with pytest.raises(AttributeError):
        plan.stages.append(plan.stages[0])


def test_plan_schema_deep_freezes_nested_strategy_and_metadata():
    ExecutionContract, ModelPlan, SegmentPlan, StagePlan, TransitionPlan = _load_plan_types()
    plan = ModelPlan(
        stages=[
            StagePlan(
                stage_id=0,
                segments=[
                    SegmentPlan(
                        start=0,
                        end=1,
                        strategy={
                            "mesh": {"tp_ranks": [0, 1]},
                            "replicas": [{"dp_rank": 0}, {"dp_rank": 1}],
                        },
                    )
                ],
            )
        ],
        transitions=[
            TransitionPlan(
                source_stage_id=0,
                target_stage_id=1,
                kind="pipeline",
                metadata={"links": [{"src": 0, "dst": 1}]},
            )
        ],
        contract=ExecutionContract(
            world_size=4,
            micro_batch_size=2,
            gradient_accumulation_steps=8,
        ),
    )

    with pytest.raises(TypeError):
        plan.stages[0].segments[0].strategy["mesh"]["tp_ranks"][0] = 99

    with pytest.raises(TypeError):
        plan.transitions[0].metadata["links"][0]["src"] = 9


def test_segment_plan_docstring_declares_inclusive_bounds():
    _, _, SegmentPlan, _, _ = _load_plan_types()

    assert SegmentPlan.__doc__ is not None
    assert "inclusive" in SegmentPlan.__doc__.lower()
    assert "[start, end]" in SegmentPlan.__doc__


def test_model_plan_keeps_transitions_and_execution_contract():
    ExecutionContract, ModelPlan, SegmentPlan, StagePlan, TransitionPlan = _load_plan_types()
    plan = ModelPlan(
        stages=[
            StagePlan(
                stage_id=0,
                segments=[SegmentPlan(start=0, end=1, strategy={"tp": 1, "dp": 2})],
            ),
            StagePlan(
                stage_id=1,
                segments=[SegmentPlan(start=2, end=3, strategy={"tp": 2, "dp": 1})],
            ),
        ],
        transitions=[
            TransitionPlan(
                source_stage_id=0,
                target_stage_id=1,
                kind="pipeline",
                metadata={"buffer_layers": 1},
            )
        ],
        contract=ExecutionContract(
            world_size=4,
            micro_batch_size=2,
            gradient_accumulation_steps=8,
        ),
    )

    assert plan.transitions[0].kind == "pipeline"
    assert plan.transitions[0].metadata["buffer_layers"] == 1
    assert plan.contract.world_size == 4
    assert plan.contract.gradient_accumulation_steps == 8
