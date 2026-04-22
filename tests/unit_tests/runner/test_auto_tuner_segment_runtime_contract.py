import pytest

from flagscale.runner.auto_tuner.plan.lowering import lower_strategy_to_plan
from flagscale.runner.auto_tuner.plan.runtime import build_segment_hetero_runtime_overrides
from flagscale.runner.auto_tuner.plan.schema import (
    ExecutionContract,
    ModelPlan,
    SegmentPlan,
    StagePlan,
    TransitionPlan,
)
from flagscale.runner.auto_tuner.plan.segment_runtime_contract import (
    build_segment_runtime_contract,
)
from tests.unit_tests.runner.segment_runtime_test_utils import (
    segment_runtime_config,
    segment_runtime_strategy,
)


def test_build_segment_runtime_overrides_materializes_exact_contract(tmp_path):
    config = segment_runtime_config(tmp_path, with_num_layers=True)
    strategy = segment_runtime_strategy()
    plan = lower_strategy_to_plan(strategy, config)

    overrides = build_segment_hetero_runtime_overrides(strategy, config)
    runtime = overrides["hetero"]["segment_runtime"]

    assert runtime["hetero_stage_segment_splits"] == [[1, 1], [1, 1]]
    assert runtime["hetero_stage_segment_meshes"][0][0] == [2, 1, 1, 1, 1]
    assert runtime["hetero_stage_segment_transitions"][0]["kind"] == "segment-redistribution"
    assert runtime["hetero_stage_segment_transitions"][0]["source_mesh"] == {
        "tp": 2,
        "cp": 1,
        "ep": 1,
        "dp": 1,
        "pp": 1,
    }
    assert runtime["hetero_stage_segment_transitions"][0]["redistribution_kind"] == "tp-dp"
    assert runtime["hetero_stage_segment_transitions"][0]["requires_sequence_parallel"] is True
    assert set(runtime) == {
        "hetero_stage_segment_splits",
        "hetero_stage_segment_meshes",
        "hetero_stage_segment_transitions",
    }
    assert len(runtime["hetero_stage_segment_transitions"]) == len(plan.transitions)
    assert set(plan.transitions[0].metadata) == {
        "source_mesh",
        "target_mesh",
        "batch_unit",
        "redistribution_kind",
        "requires_sequence_parallel",
    }
    assert plan.transitions[0].metadata["source_mesh"] == {
        "tp": 2,
        "cp": 1,
        "ep": 1,
        "dp": 1,
        "pp": 1,
    }


def test_build_segment_runtime_contract_keeps_only_segment_redistribution_transitions():
    plan = ModelPlan(
        stages=(
            StagePlan(
                stage_id=0,
                segments=(
                    SegmentPlan(
                        start=0,
                        end=0,
                        strategy={
                            "data_parallel_size": 1,
                            "tensor_model_parallel_size": 2,
                            "context_parallel_size": 1,
                            "expert_model_parallel_size": 1,
                            "pp_local": 1,
                        },
                    ),
                    SegmentPlan(
                        start=1,
                        end=1,
                        strategy={
                            "data_parallel_size": 2,
                            "tensor_model_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_model_parallel_size": 1,
                            "pp_local": 1,
                        },
                    ),
                ),
            ),
        ),
        contract=ExecutionContract(
            world_size=2,
            micro_batch_size=1,
            gradient_accumulation_steps=2,
            global_batch_size=4,
        ),
        transitions=(
            TransitionPlan(
                source_stage_id=0,
                target_stage_id=0,
                kind="segment-redistribution",
                source_segment_index=0,
                target_segment_index=1,
                metadata={
                    "source_mesh": {
                        "tensor_model_parallel_size": 2,
                        "context_parallel_size": 1,
                        "expert_model_parallel_size": 1,
                        "data_parallel_size": 1,
                        "pp_local": 1,
                    },
                    "target_mesh": {
                        "tensor_model_parallel_size": 1,
                        "context_parallel_size": 1,
                        "expert_model_parallel_size": 1,
                        "data_parallel_size": 2,
                        "pp_local": 1,
                    },
                },
            ),
            TransitionPlan(
                source_stage_id=0,
                target_stage_id=1,
                kind="pipeline",
                metadata={"buffer_layers": 1},
            ),
        ),
    )

    contract = build_segment_runtime_contract(plan)

    assert len(contract["hetero_stage_segment_transitions"]) == 1
    assert contract["hetero_stage_segment_transitions"][0]["kind"] == "segment-redistribution"
    assert contract["hetero_stage_segment_transitions"][0]["source_mesh"] == {
        "tp": 2,
        "cp": 1,
        "ep": 1,
        "dp": 1,
        "pp": 1,
    }
    assert contract["hetero_stage_segment_transitions"][0]["target_mesh"] == {
        "tp": 1,
        "cp": 1,
        "ep": 1,
        "dp": 2,
        "pp": 1,
    }
    assert contract["hetero_stage_segment_transitions"][0]["batch_unit"] == 2


def test_build_segment_runtime_contract_rejects_same_mesh_transition():
    plan = ModelPlan(
        stages=(
            StagePlan(
                stage_id=0,
                segments=(
                    SegmentPlan(
                        start=0,
                        end=0,
                        strategy={
                            "data_parallel_size": 1,
                            "tensor_model_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_model_parallel_size": 1,
                            "pp_local": 1,
                        },
                    ),
                    SegmentPlan(
                        start=1,
                        end=1,
                        strategy={
                            "data_parallel_size": 1,
                            "tensor_model_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_model_parallel_size": 1,
                            "pp_local": 1,
                        },
                    ),
                ),
            ),
        ),
        contract=ExecutionContract(
            world_size=1,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            global_batch_size=1,
        ),
        transitions=(
            TransitionPlan(
                source_stage_id=0,
                target_stage_id=0,
                kind="segment-redistribution",
                source_segment_index=0,
                target_segment_index=1,
                metadata={
                    "source_mesh": {
                        "tensor_model_parallel_size": 1,
                        "context_parallel_size": 1,
                        "expert_model_parallel_size": 1,
                        "data_parallel_size": 1,
                        "pp_local": 1,
                    },
                    "target_mesh": {
                        "tensor_model_parallel_size": 1,
                        "context_parallel_size": 1,
                        "expert_model_parallel_size": 1,
                        "data_parallel_size": 1,
                        "pp_local": 1,
                    },
                },
            ),
        ),
    )

    with pytest.raises(ValueError, match="redistribution_kind"):
        build_segment_runtime_contract(plan)


def test_build_segment_runtime_contract_uses_stage_segment_meshes_not_tampered_metadata():
    plan = ModelPlan(
        stages=(
            StagePlan(
                stage_id=0,
                segments=(
                    SegmentPlan(
                        start=0,
                        end=0,
                        strategy={
                            "data_parallel_size": 1,
                            "tensor_model_parallel_size": 2,
                            "context_parallel_size": 1,
                            "expert_model_parallel_size": 1,
                            "pp_local": 1,
                        },
                    ),
                    SegmentPlan(
                        start=1,
                        end=1,
                        strategy={
                            "data_parallel_size": 2,
                            "tensor_model_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_model_parallel_size": 1,
                            "pp_local": 1,
                        },
                    ),
                ),
            ),
        ),
        contract=ExecutionContract(
            world_size=2,
            micro_batch_size=1,
            gradient_accumulation_steps=2,
            global_batch_size=4,
        ),
        transitions=(
            TransitionPlan(
                source_stage_id=0,
                target_stage_id=0,
                kind="segment-redistribution",
                source_segment_index=0,
                target_segment_index=1,
                metadata={
                    "source_mesh": {
                        "tensor_model_parallel_size": 99,
                        "context_parallel_size": 1,
                        "expert_model_parallel_size": 1,
                        "data_parallel_size": 1,
                        "pp_local": 1,
                    },
                    "target_mesh": {
                        "tensor_model_parallel_size": 1,
                        "context_parallel_size": 1,
                        "expert_model_parallel_size": 1,
                        "data_parallel_size": 2,
                        "pp_local": 1,
                    },
                },
            ),
        ),
    )

    contract = build_segment_runtime_contract(plan)

    assert contract["hetero_stage_segment_transitions"][0]["source_mesh"]["tp"] == 2


def test_build_segment_runtime_contract_rejects_non_unit_pp_local():
    plan = ModelPlan(
        stages=(
            StagePlan(
                stage_id=0,
                segments=(
                    SegmentPlan(
                        start=0,
                        end=0,
                        strategy={
                            "data_parallel_size": 1,
                            "tensor_model_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_model_parallel_size": 1,
                            "pp_local": 2,
                        },
                    ),
                    SegmentPlan(
                        start=1,
                        end=1,
                        strategy={
                            "data_parallel_size": 2,
                            "tensor_model_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_model_parallel_size": 1,
                            "pp_local": 1,
                        },
                    ),
                ),
            ),
        ),
        contract=ExecutionContract(
            world_size=2,
            micro_batch_size=1,
            gradient_accumulation_steps=2,
            global_batch_size=4,
        ),
        transitions=(
            TransitionPlan(
                source_stage_id=0,
                target_stage_id=0,
                kind="segment-redistribution",
                source_segment_index=0,
                target_segment_index=1,
            ),
        ),
    )

    with pytest.raises(ValueError, match="pp"):
        build_segment_runtime_contract(plan)


def test_build_segment_runtime_contract_rejects_non_divisible_batch_unit():
    plan = ModelPlan(
        stages=(
            StagePlan(
                stage_id=0,
                segments=(
                    SegmentPlan(
                        start=0,
                        end=0,
                        strategy={
                            "data_parallel_size": 1,
                            "tensor_model_parallel_size": 2,
                            "context_parallel_size": 1,
                            "expert_model_parallel_size": 1,
                            "pp_local": 1,
                        },
                    ),
                    SegmentPlan(
                        start=1,
                        end=1,
                        strategy={
                            "data_parallel_size": 2,
                            "tensor_model_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_model_parallel_size": 1,
                            "pp_local": 1,
                        },
                    ),
                ),
            ),
        ),
        contract=ExecutionContract(
            world_size=2,
            micro_batch_size=1,
            gradient_accumulation_steps=2,
            global_batch_size=6,
        ),
        transitions=(
            TransitionPlan(
                source_stage_id=0,
                target_stage_id=0,
                kind="segment-redistribution",
                source_segment_index=0,
                target_segment_index=1,
            ),
        ),
    )

    with pytest.raises(ValueError, match="batch_unit"):
        build_segment_runtime_contract(plan)
