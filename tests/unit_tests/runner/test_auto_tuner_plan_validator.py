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


def _build_analysis_only_plan(**overrides):
    _, ModelPlan, _, _, _ = _load_plan_validator_types()
    plan_kwargs = {
        "total_layers": 6,
        "contract": _contract(),
        "stages": (
            _stage(0, (_segment(0, 1), _segment(2, 3))),
            _stage(1, (_segment(4, 5),)),
        ),
    }
    plan_kwargs.update(overrides)
    return ModelPlan(**plan_kwargs)


def test_validate_model_plan_accepts_stage_executable_plan():
    result = _validate(_build_stage_executable_plan())

    assert result.runtime_mode == "stage-executable"


def test_validate_model_plan_accepts_analysis_only_plan():
    result = _validate(_build_analysis_only_plan())

    assert result.runtime_mode == "analysis-only"


def test_validate_model_plan_accepts_analysis_only_plan_with_valid_explicit_device_groups():
    plan = _build_analysis_only_plan(
        stages=(
            _stage(0, (_segment(0, 1), _segment(2, 3)), device_group=(0, 1)),
            _stage(1, (_segment(4, 5),), device_group=(2, 3)),
        )
    )

    result = _validate(plan)

    assert result.runtime_mode == "analysis-only"


@pytest.mark.parametrize(
    ("device_group", "contract", "message"),
    [
        ((-1, 0), _contract(), ">= 0"),
        (("a", "b"), _contract(), "int"),
        ((0, 4), _contract(world_size=4), "world_size"),
    ],
)
def test_validate_model_plan_rejects_invalid_explicit_device_group_ranks(
    device_group,
    contract,
    message,
):
    plan = _build_analysis_only_plan(
        contract=contract,
        stages=(
            _stage(0, (_segment(0, 1), _segment(2, 3)), device_group=device_group),
            _stage(1, (_segment(4, 5),)),
        ),
    )

    with pytest.raises(ValueError, match=message):
        _validate(plan)


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        (
            _build_analysis_only_plan(
                stages=(
                    _stage(0, (_segment(0, 1), _segment(4, 5))),
                    _stage(1, (_segment(2, 3),)),
                )
            ),
            "stage 0.*gap",
        ),
        (
            _build_analysis_only_plan(
                stages=(
                    _stage(0, (_segment(0, 2), _segment(2, 3))),
                    _stage(1, (_segment(4, 5),)),
                )
            ),
            "stage 0.*overlap",
        ),
        (
            _build_analysis_only_plan(
                total_layers=5,
                stages=(
                    _stage(0, (_segment(0, 1),)),
                    _stage(1, (_segment(2, 5),)),
                ),
            ),
            "out of bounds",
        ),
    ],
)
def test_validate_model_plan_rejects_invalid_global_stage_coverage(plan, message):
    with pytest.raises(ValueError, match=message):
        _validate(plan)


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        ((_segment(0, 1), _segment(3, 4)), "stage 0.*gap"),
        ((_segment(0, 2), _segment(2, 3)), "stage 0.*overlap"),
        ((_segment(2, 3), _segment(0, 1)), "stage 0.*order"),
    ],
)
def test_validate_model_plan_rejects_invalid_per_stage_segment_coverage(segments, message):
    _, ModelPlan, _, _, _ = _load_plan_validator_types()
    plan = ModelPlan(
        total_layers=6,
        contract=_contract(),
        stages=(
            _stage(0, segments),
            _stage(1, (_segment(4, 5),)),
        ),
    )

    with pytest.raises(ValueError, match=message):
        _validate(plan)


@pytest.mark.parametrize(
    ("stages", "message"),
    [
        (
            (
                _stage(7, (_segment(0, 1),)),
                _stage(7, (_segment(2, 3),)),
            ),
            "unique",
        ),
        (
            (
                _stage(1, (_segment(0, 1),)),
                _stage(2, (_segment(2, 3),)),
            ),
            "from 0",
        ),
        (
            (
                _stage(0, (_segment(0, 1),)),
                _stage(2, (_segment(2, 3),)),
            ),
            "continuous",
        ),
    ],
)
def test_validate_model_plan_rejects_invalid_stage_ids(stages, message):
    plan = _build_stage_executable_plan(stages=stages)

    with pytest.raises(ValueError, match=message):
        _validate(plan)


def test_validate_model_plan_rejects_non_positive_parallelism():
    plan = _build_stage_executable_plan(
        stages=(
            _stage(0, (_segment(0, 1, data_parallel_size=0),), device_group=(0, 1)),
            _stage(1, (_segment(2, 3),), device_group=(2, 3)),
        )
    )

    with pytest.raises(ValueError, match="positive"):
        _validate(plan)


def test_validate_model_plan_rejects_global_batch_size_mismatch():
    plan = _build_stage_executable_plan(contract=_contract(gbs=8))

    with pytest.raises(ValueError, match="global batch size"):
        _validate(plan)


def test_validate_model_plan_rejects_missing_data_parallel_size_for_gbs_check():
    plan = _build_stage_executable_plan(
        stages=(
            _stage(0, (_segment(0, 1, data_parallel_size=None, dp=None),), device_group=(0, 1)),
            _stage(1, (_segment(2, 3),), device_group=(2, 3)),
        )
    )

    with pytest.raises(ValueError, match="data_parallel_size"):
        _validate(plan)


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
                _stage(1, (_segment(2, 3),), device_group=(2, 3)),
            ),
            "world_size",
        ),
    ],
)
def test_validate_model_plan_rejects_invalid_runtime_device_groups(stages, message):
    contract = _contract(world_size=5) if message == "world_size" else _contract()
    plan = _build_stage_executable_plan(stages=stages, contract=contract)

    with pytest.raises(ValueError, match=message):
        _validate(plan)


def test_validate_model_plan_rejects_pipeline_stage_count_mismatch():
    plan = _build_stage_executable_plan(
        stages=(
            _stage(0, (_segment(0, 1, pipeline_model_parallel_size=3),), device_group=(0, 1)),
            _stage(1, (_segment(2, 3, pipeline_model_parallel_size=3),), device_group=(2, 3)),
        )
    )

    with pytest.raises(ValueError, match="pipeline_model_parallel_size"):
        _validate(plan)


def test_validate_model_plan_rejects_parallelism_that_exceeds_stage_device_group():
    plan = _build_stage_executable_plan(
        stages=(
            _stage(0, (_segment(0, 1, tensor_model_parallel_size=8),), device_group=(0, 1)),
            _stage(1, (_segment(2, 3),), device_group=(2, 3)),
        )
    )

    with pytest.raises(ValueError, match="device_group"):
        _validate(plan)


def test_validate_model_plan_rejects_data_parallelism_that_exceeds_stage_device_group():
    plan = _build_stage_executable_plan(
        stages=(
            _stage(0, (_segment(0, 1, data_parallel_size=2),), device_group=(0,)),
            _stage(1, (_segment(2, 3, data_parallel_size=2),), device_group=(1, 2, 3)),
        ),
        contract=_contract(world_size=4),
    )

    with pytest.raises(ValueError, match="device_group"):
        _validate(plan)


def test_validate_model_plan_rejects_inconsistent_data_parallelism_across_stages():
    plan = _build_stage_executable_plan(
        contract=_contract(gbs=None),
        stages=(
            _stage(0, (_segment(0, 1, data_parallel_size=1),), device_group=(0,)),
            _stage(1, (_segment(2, 3, data_parallel_size=2),), device_group=(1, 2)),
        ),
    )

    with pytest.raises(ValueError, match="data_parallel_size"):
        _validate(plan)


@pytest.mark.parametrize(
    ("stages", "message"),
    [
        (
            (
                _stage(0, (_segment(0, 1), _segment(2, 3)), device_group=(0, 0)),
                _stage(1, (_segment(4, 5),), device_group=(2, 3)),
            ),
            "duplicate",
        ),
        (
            (
                _stage(0, (_segment(0, 1), _segment(2, 3)), device_group=(0, 1)),
                _stage(1, (_segment(4, 5),), device_group=(1, 2)),
            ),
            "overlap",
        ),
    ],
)
def test_validate_model_plan_rejects_invalid_explicit_device_groups_in_analysis_only(stages, message):
    plan = _build_analysis_only_plan(stages=stages)

    with pytest.raises(ValueError, match=message):
        _validate(plan)
