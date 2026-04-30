import pytest

from flagscale.runner.auto_tuner.plan.schema import (
    ExecutionContract,
    ModelPlan,
    SegmentPlan,
    StagePlan,
    TransitionPlan,
)
from flagscale.runner.auto_tuner.plan.validator import validate_model_plan


def _strategy(*, tp: int, dp: int) -> dict[str, object]:
    return {
        "context_parallel_size": 1,
        "data_parallel_size": dp,
        "device_type": "nvidia_l20",
        "expert_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "pp_local": 1,
        "sequence_parallel": True,
        "tensor_model_parallel_size": tp,
        "use_distributed_optimizer": False,
    }


def _mesh(*, tp: int, dp: int) -> dict[str, int]:
    return {"tp": tp, "cp": 1, "ep": 1, "dp": dp, "pp": 1}


def test_validate_model_plan_rejects_tp_segment_after_tp1_stage_shell():
    plan = ModelPlan(
        total_layers=2,
        contract=ExecutionContract(
            world_size=2,
            micro_batch_size=1,
            gradient_accumulation_steps=16,
            global_batch_size=32,
        ),
        stages=(
            StagePlan(
                stage_id=0,
                device_group=(0, 1),
                segments=(
                    SegmentPlan(start=0, end=0, strategy=_strategy(tp=1, dp=2)),
                    SegmentPlan(start=1, end=1, strategy=_strategy(tp=2, dp=1)),
                ),
            ),
        ),
        transitions=(
            TransitionPlan(
                source_stage_id=0,
                target_stage_id=0,
                kind="segment-redistribution",
                source_segment_index=0,
                target_segment_index=1,
                metadata={"source_mesh": _mesh(tp=1, dp=2), "target_mesh": _mesh(tp=2, dp=1)},
            ),
        ),
    )

    with pytest.raises(ValueError, match="stage-shell"):
        validate_model_plan(plan)


def test_validate_model_plan_rejects_non_divisible_segment_dp_batch_axis():
    plan = ModelPlan(
        total_layers=2,
        contract=ExecutionContract(
            world_size=2,
            micro_batch_size=1,
            gradient_accumulation_steps=16,
            global_batch_size=32,
        ),
        stages=(
            StagePlan(
                stage_id=0,
                device_group=(0, 1),
                segments=(
                    SegmentPlan(start=0, end=0, strategy=_strategy(tp=2, dp=1)),
                    SegmentPlan(start=1, end=1, strategy=_strategy(tp=1, dp=2)),
                ),
            ),
        ),
        transitions=(
            TransitionPlan(
                source_stage_id=0,
                target_stage_id=0,
                kind="segment-redistribution",
                source_segment_index=0,
                target_segment_index=1,
                metadata={"source_mesh": _mesh(tp=2, dp=1), "target_mesh": _mesh(tp=1, dp=2)},
            ),
        ),
    )

    with pytest.raises(ValueError, match="batch axis"):
        validate_model_plan(plan)
