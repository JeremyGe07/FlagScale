from dataclasses import FrozenInstanceError

import pytest

from flagscale.runner.auto_tuner.plan import (
    ExecutionContract,
    ModelPlan,
    SegmentPlan,
    StagePlan,
    TransitionPlan,
)


def test_model_plan_accepts_multi_segment_stage():
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


def test_model_plan_keeps_transitions_and_execution_contract():
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
