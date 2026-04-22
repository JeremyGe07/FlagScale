import pytest

from flagscale.runner.auto_tuner.plan.schema import (
    ExecutionContract,
    ModelPlan,
    SegmentPlan,
    StagePlan,
    TransitionPlan,
)
from flagscale.runner.auto_tuner.plan.summary import is_runtime_executable_plan
from flagscale.runner.auto_tuner.plan.validator import validate_model_plan


def _strategy(**overrides):
    strategy = {
        "context_parallel_size": 1,
        "cp": 1,
        "data_parallel_size": 1,
        "dp": 1,
        "device_type": "mlu290",
        "expert_model_parallel_size": 1,
        "ep": 1,
        "pipeline_model_parallel_size": 1,
        "pp": 1,
        "pp_local": 1,
        "sequence_parallel": True,
        "tensor_model_parallel_size": 1,
        "tp": 1,
        "use_distributed_optimizer": False,
    }
    strategy.update(overrides)
    for alias, long_name in (
        ("tp", "tensor_model_parallel_size"),
        ("cp", "context_parallel_size"),
        ("ep", "expert_model_parallel_size"),
        ("dp", "data_parallel_size"),
        ("pp", "pipeline_model_parallel_size"),
        ("pp_local", "pipeline_model_parallel_size"),
    ):
        if alias in overrides:
            strategy[long_name] = overrides[alias]
    for long_name, alias in (
        ("tensor_model_parallel_size", "tp"),
        ("context_parallel_size", "cp"),
        ("expert_model_parallel_size", "ep"),
        ("data_parallel_size", "dp"),
        ("pipeline_model_parallel_size", "pp"),
    ):
        if long_name in overrides:
            strategy[alias] = overrides[long_name]
    return strategy


def _contract(contract_gbs=8, acc_step=4, world_size=4):
    return ExecutionContract(
        world_size=world_size,
        micro_batch_size=2,
        gradient_accumulation_steps=acc_step,
        global_batch_size=contract_gbs,
    )


def _segment(start, end, **overrides):
    return SegmentPlan(start=start, end=end, strategy=_strategy(**overrides))


def _stage(stage_id, segments, device_group):
    return StagePlan(stage_id=stage_id, segments=segments, device_group=device_group)


def _mesh(**overrides):
    mesh = {
        "tp": 1,
        "cp": 1,
        "ep": 1,
        "dp": 1,
        "pp_local": 1,
        "pp": 1,
        "pipeline_model_parallel_size": 1,
        "device_type": "mlu290",
        "sequence_parallel": True,
        "use_distributed_optimizer": False,
    }
    mesh.update(overrides)
    for alias, long_name in (
        ("tp", "tensor_model_parallel_size"),
        ("cp", "context_parallel_size"),
        ("ep", "expert_model_parallel_size"),
        ("dp", "data_parallel_size"),
        ("pp", "pipeline_model_parallel_size"),
        ("pp_local", "pipeline_model_parallel_size"),
    ):
        if alias in overrides:
            mesh[long_name] = overrides[alias]
    return mesh


def _transition(stage_id, mesh_from, mesh_to):
    return TransitionPlan(
        source_stage_id=stage_id,
        target_stage_id=stage_id,
        kind="segment-redistribution",
        source_segment_index=0,
        target_segment_index=1,
        metadata={"source_mesh": mesh_from, "target_mesh": mesh_to},
    )


def _segment_plan(
    *,
    contract_gbs,
    acc_step,
    stage0_meshes,
    stage1_meshes,
    stage0_device_group=(0, 1),
    stage1_device_group=(2, 3),
    world_size=4,
):
    stage0_meshes = tuple(dict(mesh) for mesh in stage0_meshes)
    stage1_meshes = tuple(dict(mesh) for mesh in stage1_meshes)
    stages = (
        _stage(
            0,
            (
                _segment(0, 0, **_strategy(**stage0_meshes[0])),
                _segment(1, 1, **_strategy(**stage0_meshes[1])),
            ),
            stage0_device_group,
        ),
        _stage(
            1,
            (
                _segment(2, 2, **_strategy(**stage1_meshes[0])),
                _segment(3, 3, **_strategy(**stage1_meshes[1])),
            ),
            stage1_device_group,
        ),
    )
    return ModelPlan(
        total_layers=4,
        contract=_contract(contract_gbs=contract_gbs, acc_step=acc_step, world_size=world_size),
        stages=stages,
        transitions=(
            _transition(0, _mesh(**stage0_meshes[0]), _mesh(**stage0_meshes[1])),
            _transition(1, _mesh(**stage1_meshes[0]), _mesh(**stage1_meshes[1])),
        ),
    )


def test_validate_model_plan_accepts_segment_executable_subset():
    plan = _segment_plan(
        contract_gbs=8,
        acc_step=4,
        stage0_meshes=(
            {"tp": 2, "cp": 1, "ep": 1, "dp": 1, "pp_local": 1},
            {"tp": 1, "cp": 1, "ep": 1, "dp": 2, "pp_local": 1},
        ),
        stage1_meshes=(
            {"tp": 1, "cp": 1, "ep": 1, "dp": 2, "pp_local": 1},
            {"tp": 2, "cp": 1, "ep": 1, "dp": 1, "pp_local": 1},
        ),
    )

    result = validate_model_plan(plan)

    assert result.runtime_mode == "segment-executable"
    assert is_runtime_executable_plan(plan) is True


def test_validate_model_plan_rejects_missing_segment_redistribution_transition():
    plan = _segment_plan(
        contract_gbs=8,
        acc_step=4,
        stage0_meshes=(
            {"tp": 2, "cp": 1, "ep": 1, "dp": 1, "pp_local": 1},
            {"tp": 1, "cp": 1, "ep": 1, "dp": 2, "pp_local": 1},
        ),
        stage1_meshes=(
            {"tp": 1, "cp": 1, "ep": 1, "dp": 2, "pp_local": 1},
            {"tp": 2, "cp": 1, "ep": 1, "dp": 1, "pp_local": 1},
        ),
    )
    plan = ModelPlan(
        stages=plan.stages,
        contract=plan.contract,
        total_layers=plan.total_layers,
    )

    with pytest.raises(ValueError, match="segment-redistribution"):
        validate_model_plan(plan)


def test_validate_model_plan_rejects_segment_with_invalid_dp_divisibility():
    plan = _segment_plan(
        contract_gbs=8,
        acc_step=4,
        stage0_device_group=(0, 1, 2, 3),
        stage1_device_group=(4, 5, 6, 7),
        world_size=8,
        stage0_meshes=(
            {"tp": 1, "cp": 1, "ep": 1, "dp": 4, "pp_local": 1},
            {"tp": 1, "cp": 1, "ep": 1, "dp": 4, "pp_local": 1},
        ),
        stage1_meshes=(
            {"tp": 2, "cp": 1, "ep": 1, "dp": 2, "pp_local": 1},
            {"tp": 2, "cp": 1, "ep": 1, "dp": 2, "pp_local": 1},
        ),
    )

    with pytest.raises(ValueError, match="divide"):
        validate_model_plan(plan)


def test_validate_model_plan_rejects_tp_changes_without_sequence_parallel():
    plan = _segment_plan(
        contract_gbs=8,
        acc_step=4,
        stage0_meshes=(
            {"tp": 2, "cp": 1, "ep": 1, "dp": 1, "pp_local": 1, "sequence_parallel": False},
            {"tp": 1, "cp": 1, "ep": 1, "dp": 2, "pp_local": 1, "sequence_parallel": False},
        ),
        stage1_meshes=(
            {"tp": 1, "cp": 1, "ep": 1, "dp": 2, "pp_local": 1, "sequence_parallel": False},
            {"tp": 2, "cp": 1, "ep": 1, "dp": 1, "pp_local": 1, "sequence_parallel": False},
        ),
    )

    with pytest.raises(ValueError, match="sequence_parallel"):
        validate_model_plan(plan)


@pytest.mark.parametrize(
    ("_field", "mesh_pair", "message"),
    [
        (
            "cp",
            (
                {"tp": 2, "cp": 1, "ep": 1, "dp": 2, "pp_local": 1},
                {"tp": 1, "cp": 2, "ep": 1, "dp": 2, "pp_local": 1},
            ),
            "context_parallel_size",
        ),
        (
            "ep",
            (
                {"tp": 2, "cp": 1, "ep": 1, "dp": 2, "pp_local": 1},
                {"tp": 1, "cp": 1, "ep": 2, "dp": 2, "pp_local": 1},
            ),
            "expert_model_parallel_size",
        ),
        (
            "use_distributed_optimizer",
            (
                {"tp": 2, "cp": 1, "ep": 1, "dp": 2, "pp_local": 1, "use_distributed_optimizer": False},
                {"tp": 2, "cp": 1, "ep": 1, "dp": 2, "pp_local": 1, "use_distributed_optimizer": True},
            ),
            "use_distributed_optimizer",
        ),
        (
            "device_type",
            (
                {"tp": 2, "cp": 1, "ep": 1, "dp": 2, "pp_local": 1, "device_type": "mlu290"},
                {"tp": 2, "cp": 1, "ep": 1, "dp": 2, "pp_local": 1, "device_type": "mlu500"},
            ),
            "device_type",
        ),
    ],
)
def test_validate_model_plan_rejects_stage_internal_mesh_changes(_field, mesh_pair, message):
    plan = _segment_plan(
        contract_gbs=8,
        acc_step=4,
        stage0_meshes=mesh_pair,
        stage0_device_group=(0, 1, 2, 3),
        stage1_device_group=(4, 5, 6, 7),
        world_size=8,
        stage1_meshes=(
            {"tp": 2, "cp": 1, "ep": 1, "dp": 2, "pp_local": 1},
            {"tp": 2, "cp": 1, "ep": 1, "dp": 2, "pp_local": 1},
        ),
    )

    with pytest.raises(ValueError, match=message):
        validate_model_plan(plan)
