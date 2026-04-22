from dataclasses import FrozenInstanceError
import importlib
import sys
import types

import pytest


def _restore_modules(previous_modules):
    ordered_names = sorted(previous_modules, key=lambda name: name.count("."))

    for name in ordered_names:
        module = previous_modules[name]
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module

        parent_name, _, child_name = name.rpartition(".")
        if not parent_name:
            continue

        parent_module = sys.modules.get(parent_name)
        if parent_module is None:
            continue

        if module is None:
            if hasattr(parent_module, child_name):
                delattr(parent_module, child_name)
            continue

        setattr(parent_module, child_name, module)


def _import_plan_module():
    module_names = (
        "flagscale.runner.auto_tuner.plan",
        "flagscale.runner.auto_tuner",
        "flagscale.runner.auto_tuner.tuner",
    )
    previous_modules = {name: sys.modules.get(name) for name in module_names}

    try:
        for name in module_names:
            sys.modules.pop(name, None)
        return importlib.import_module("flagscale.runner.auto_tuner.plan")
    finally:
        _restore_modules(previous_modules)


def _load_plan_types():
    plan_module = _import_plan_module()
    return (
        plan_module.ExecutionContract,
        plan_module.ModelPlan,
        plan_module.SegmentPlan,
        plan_module.StagePlan,
        plan_module.TransitionPlan,
    )


def _import_auto_tuner_module():
    module_name = "flagscale.runner.auto_tuner"
    previous_modules = {module_name: sys.modules.get(module_name)}

    try:
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)
    finally:
        _restore_modules(previous_modules)


def test_importing_plan_module_does_not_load_tuner():
    module_names = (
        "flagscale.runner.auto_tuner",
        "flagscale.runner.auto_tuner.tuner",
    )
    previous_modules = {name: sys.modules.get(name) for name in module_names}

    try:
        sys.modules.pop("flagscale.runner.auto_tuner.tuner", None)
        _import_plan_module()
        assert "flagscale.runner.auto_tuner.tuner" not in sys.modules
    finally:
        _restore_modules(previous_modules)


def test_import_plan_module_restores_existing_dummy_tuner_module(monkeypatch):
    dummy_auto_tuner = types.ModuleType("flagscale.runner.auto_tuner")
    dummy_tuner = types.ModuleType("flagscale.runner.auto_tuner.tuner")
    dummy_auto_tuner.tuner = dummy_tuner
    monkeypatch.setitem(sys.modules, "flagscale.runner.auto_tuner", dummy_auto_tuner)
    monkeypatch.setitem(sys.modules, "flagscale.runner.auto_tuner.tuner", dummy_tuner)

    _import_plan_module()

    assert sys.modules["flagscale.runner.auto_tuner"] is dummy_auto_tuner
    assert sys.modules["flagscale.runner.auto_tuner.tuner"] is dummy_tuner
    assert sys.modules["flagscale.runner.auto_tuner"].tuner is dummy_tuner


def test_auto_tuner_dir_does_not_duplicate_lazy_exports(monkeypatch):
    tuner_module = types.ModuleType("flagscale.runner.auto_tuner.tuner")
    tuner_module.AutoTuner = object()
    tuner_module.ServeAutoTunner = object()
    monkeypatch.setitem(sys.modules, "flagscale.runner.auto_tuner.tuner", tuner_module)

    auto_tuner_module = _import_auto_tuner_module()
    auto_tuner_module.AutoTuner

    exported_names = dir(auto_tuner_module)
    assert exported_names.count("AutoTuner") == 1
    assert exported_names.count("ServeAutoTunner") == 1


def test_from_auto_tuner_imports_lazy_exports(monkeypatch):
    previous_modules = {
        name: sys.modules.get(name)
        for name in (
            "flagscale.runner.auto_tuner",
            "flagscale.runner.auto_tuner.tuner",
        )
    }
    tuner_module = types.ModuleType("flagscale.runner.auto_tuner.tuner")
    auto_tuner = object()
    serve_auto_tunner = object()
    tuner_module.AutoTuner = auto_tuner
    tuner_module.ServeAutoTunner = serve_auto_tunner
    monkeypatch.setitem(sys.modules, "flagscale.runner.auto_tuner.tuner", tuner_module)

    try:
        sys.modules.pop("flagscale.runner.auto_tuner", None)

        namespace: dict[str, object] = {}
        exec(
            "from flagscale.runner.auto_tuner import AutoTuner, ServeAutoTunner",
            namespace,
        )

        assert namespace["AutoTuner"] is auto_tuner
        assert namespace["ServeAutoTunner"] is serve_auto_tunner
    finally:
        _restore_modules(previous_modules)


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


def test_summarize_plan_preserves_transition_metadata():
    _, ModelPlan, SegmentPlan, StagePlan, TransitionPlan = _load_plan_types()
    plan = ModelPlan(
        stages=[
            StagePlan(
                stage_id=0,
                segments=[SegmentPlan(start=0, end=1, strategy={"tp": 1, "dp": 1})],
            )
        ],
        transitions=[
            TransitionPlan(
                source_stage_id=0,
                target_stage_id=0,
                kind="segment-redistribution",
                source_segment_index=0,
                target_segment_index=0,
                metadata={
                    "source_mesh": {"tensor_model_parallel_size": 2, "pp_local": 1},
                    "target_mesh": {"tensor_model_parallel_size": 1, "pp_local": 1},
                },
            )
        ],
    )

    summary = importlib.import_module("flagscale.runner.auto_tuner.plan.summary").summarize_plan(
        plan
    )

    assert summary["transitions"][0]["kind"] == "segment-redistribution"
    assert summary["transitions"][0]["metadata"]["source_mesh"]["tensor_model_parallel_size"] == 2
