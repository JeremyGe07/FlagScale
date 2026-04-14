import importlib

import pytest


def _load_plan_validator_types():
    schema_module = importlib.import_module("flagscale.runner.auto_tuner.plan.schema")
    validator_module = importlib.import_module("flagscale.runner.auto_tuner.plan.validator")
    return (
        schema_module.ExecutionContract,
        schema_module.ModelPlan,
        schema_module.SegmentPlan,
        schema_module.StagePlan,
        validator_module.validate_model_plan,
    )


def _strategy(**overrides):
    strategy = {
        "context_parallel_size": 1,
        "data_parallel_size": 2,
        "expert_model_parallel_size": 1,
        "pipeline_model_parallel_size": 2,
        "tensor_model_parallel_size": 1,
    }
    strategy.update(overrides)
    return strategy


def _contract(gbs=16, world_size=4):
    ExecutionContract, _, _, _, _ = _load_plan_validator_types()
    return ExecutionContract(
        world_size=world_size,
        micro_batch_size=2,
        gradient_accumulation_steps=4,
        global_batch_size=gbs,
    )


def _segment(start, end, **strategy_overrides):
    _, _, SegmentPlan, _, _ = _load_plan_validator_types()
    return SegmentPlan(start=start, end=end, strategy=_strategy(**strategy_overrides))


def _stage(stage_id, segments, device_group=()):
    _, _, _, StagePlan, _ = _load_plan_validator_types()
    return StagePlan(stage_id=stage_id, segments=segments, device_group=device_group)


def _validate(plan):
    _, _, _, _, validate_model_plan = _load_plan_validator_types()
    return validate_model_plan(plan)


def _build_stage_executable_plan(**overrides):
    _, ModelPlan, _, _, _ = _load_plan_validator_types()
    plan_kwargs = {
        "total_layers": 4,
        "contract": _contract(),
        "stages": (
            _stage(0, (_segment(0, 1),), device_group=(0, 1)),
            _stage(1, (_segment(2, 3),), device_group=(2, 3)),
        ),
    }
    plan_kwargs.update(overrides)
    return ModelPlan(**plan_kwargs)


def test_validate_model_plan_accepts_stage_executable_plan():
    result = _validate(_build_stage_executable_plan())

    assert result.is_valid is True
    assert result.runtime_mode == "stage-executable"
    assert result.errors == ()


def test_validate_model_plan_marks_multi_segment_stage_as_analysis_only():
    _, ModelPlan, _, _, _ = _load_plan_validator_types()
    plan = ModelPlan(
        total_layers=4,
        stages=(_stage(0, (_segment(0, 1), _segment(2, 3))),),
        contract=_contract(),
    )

    result = _validate(plan)

    assert result.is_valid is True
    assert result.runtime_mode == "analysis-only"
    assert result.errors == ()


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        ((_segment(0, 0), _segment(2, 3)), "gap"),
        ((_segment(0, 2), _segment(2, 3)), "overlap"),
        ((_segment(0, 4),), "out of bounds"),
    ],
)
def test_validate_model_plan_rejects_invalid_layer_spans(segments, message):
    _, ModelPlan, _, _, _ = _load_plan_validator_types()
    plan = ModelPlan(
        total_layers=4,
        stages=(_stage(0, segments),),
        contract=_contract(),
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert any(message in error for error in result.errors)


def test_validate_model_plan_rejects_non_positive_parallelism():
    plan = _build_stage_executable_plan(
        stages=(
            _stage(
                0,
                (_segment(0, 1, data_parallel_size=0),),
                device_group=(0, 1),
            ),
            _stage(1, (_segment(2, 3),), device_group=(2, 3)),
        )
    )

    result = _validate(plan)

    assert result.is_valid is False
    assert any("positive" in error for error in result.errors)


def test_validate_model_plan_rejects_global_batch_size_mismatch():
    plan = _build_stage_executable_plan(contract=_contract(gbs=8))

    result = _validate(plan)

    assert result.is_valid is False
    assert any("global batch size" in error for error in result.errors)


@pytest.mark.parametrize(
    ("stages", "message"),
    [
        (
            (
                _stage(0, (_segment(0, 1),)),
                _stage(1, (_segment(2, 3),), device_group=(2, 3)),
            ),
            "device_group",
        ),
        (
            (
                _stage(0, (_segment(0, 1),), device_group=(0, 0)),
                _stage(1, (_segment(2, 3),), device_group=(2, 3)),
            ),
            "duplicate",
        ),
        (
            (
                _stage(0, (_segment(0, 1),), device_group=(0, 1)),
                _stage(1, (_segment(2, 3),), device_group=(1, 2)),
            ),
            "overlap",
        ),
        (
            (
                _stage(0, (_segment(0, 1),), device_group=(0, 1)),
                _stage(1, (_segment(2, 3),), device_group=(2,)),
            ),
            "world_size",
        ),
    ],
)
def test_validate_model_plan_rejects_invalid_runtime_device_groups(stages, message):
    plan = _build_stage_executable_plan(stages=stages)

    result = _validate(plan)

    assert result.is_valid is False
    assert result.runtime_mode == "stage-executable"
    assert any(message in error for error in result.errors)
