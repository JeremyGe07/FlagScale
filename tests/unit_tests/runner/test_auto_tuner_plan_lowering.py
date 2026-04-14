import importlib
import json

from omegaconf import OmegaConf


def _load_lowering_types():
    lowering_module = importlib.import_module("flagscale.runner.auto_tuner.plan.lowering")
    validator_module = importlib.import_module("flagscale.runner.auto_tuner.plan.validator")
    recorder_module = importlib.import_module("flagscale.runner.auto_tuner.record.recorder")
    searcher_module = importlib.import_module("flagscale.runner.auto_tuner.search.searcher")
    return (
        lowering_module.build_execution_contract,
        lowering_module.lower_strategy_to_plan,
        lowering_module.summarize_plan,
        validator_module.validate_model_plan,
        recorder_module.Recorder,
        searcher_module.Searcher,
    )


def _make_config(tmp_path, *, num_layers=8, global_batch_size=16, cards=4, space_overrides=None):
    space = {
        "data_parallel_size": [cards],
        "use_distributed_optimizer": [False],
        "tensor_model_parallel_size": [1],
        "sequence_parallel": [False],
        "pipeline_model_parallel_size": [1],
        "num_layers_per_virtual_pipeline_stage": [0],
        "use_recompute": [False],
        "recompute_method": ["uniform"],
        "recompute_granularity": ["full"],
        "recompute_num_layers": [1],
        "micro_batch_size": [2],
        "context_parallel_size": [1],
        "expert_model_parallel_size": [1],
    }
    if space_overrides:
        space.update(space_overrides)
    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {"nnodes": 1, "nproc_per_node": cards},
                "auto_tuner": {
                    "cards": cards,
                    "nnodes": 1,
                    "nproc_per_node": cards,
                    "platform": {},
                    "algo": {"name": "grid"},
                    "space": space,
                },
            },
            "train": {
                "model": {
                    "num_layers": num_layers,
                    "global_batch_size": global_batch_size,
                    "hidden_size": 8,
                    "num_attention_heads": 4,
                    "seq_length": 16,
                },
                "system": {},
            },
        }
    )


def _strategy(**overrides):
    strategy = {
        "data_parallel_size": 2,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "micro_batch_size": 2,
        "acc_step": 4,
        "use_distributed_optimizer": False,
        "sequence_parallel": False,
        "num_layers_per_virtual_pipeline_stage": None,
        "use_recompute": False,
        "recompute_method": None,
        "recompute_granularity": None,
        "recompute_num_layers": None,
        "decoder_first_pipeline_num_layers": None,
        "decoder_last_pipeline_num_layers": None,
    }
    strategy.update(overrides)
    return strategy


def test_build_execution_contract_uses_strategy_and_config(tmp_path):
    build_execution_contract, _, _, _, _, _ = _load_lowering_types()
    config = _make_config(tmp_path, global_batch_size=32, cards=8)

    contract = build_execution_contract(
        _strategy(micro_batch_size=4, acc_step=8),
        config,
    )

    assert contract.world_size == 8
    assert contract.micro_batch_size == 4
    assert contract.gradient_accumulation_steps == 8
    assert contract.global_batch_size == 32


def test_lower_strategy_to_plan_builds_single_stage_plan_for_pp1(tmp_path):
    _, lower_strategy_to_plan, _, validate_model_plan, _, _ = _load_lowering_types()
    config = _make_config(tmp_path, num_layers=8, cards=2)

    plan = lower_strategy_to_plan(_strategy(data_parallel_size=2), config)
    result = validate_model_plan(plan)

    assert len(plan.stages) == 1
    assert len(plan.stages[0].segments) == 1
    assert plan.stages[0].segments[0].start == 0
    assert plan.stages[0].segments[0].end == 7
    assert plan.stages[0].device_group == (0, 1)
    assert result.runtime_mode == "stage-executable"


def test_lower_strategy_to_plan_builds_two_stage_plan_for_pp2(tmp_path):
    _, lower_strategy_to_plan, _, validate_model_plan, _, _ = _load_lowering_types()
    config = _make_config(tmp_path, num_layers=8, cards=4)

    plan = lower_strategy_to_plan(_strategy(pipeline_model_parallel_size=2), config)
    result = validate_model_plan(plan)

    assert len(plan.stages) == 2
    assert plan.stages[0].segments[0].start == 0
    assert plan.stages[0].segments[0].end == 3
    assert plan.stages[1].segments[0].start == 4
    assert plan.stages[1].segments[0].end == 7
    assert plan.stages[0].device_group == (0, 1)
    assert plan.stages[1].device_group == (2, 3)
    assert result.runtime_mode == "stage-executable"


def test_lower_strategy_to_plan_handles_odd_pp2_without_explicit_edge_sizes(tmp_path):
    _, lower_strategy_to_plan, _, validate_model_plan, _, _ = _load_lowering_types()
    config = _make_config(tmp_path, num_layers=9, global_batch_size=8, cards=4)

    plan = lower_strategy_to_plan(
        _strategy(
            pipeline_model_parallel_size=2,
            data_parallel_size=1,
        ),
        config,
    )
    result = validate_model_plan(plan)

    spans = [(stage.segments[0].start, stage.segments[0].end) for stage in plan.stages]

    assert spans == [(0, 3), (4, 8)]
    assert result.runtime_mode == "stage-executable"


def test_lower_strategy_to_plan_honors_decoder_first_and_last_stage_sizes(tmp_path):
    _, lower_strategy_to_plan, _, validate_model_plan, _, _ = _load_lowering_types()
    config = _make_config(tmp_path, num_layers=10, global_batch_size=8, cards=3)

    plan = lower_strategy_to_plan(
        _strategy(
            pipeline_model_parallel_size=3,
            decoder_first_pipeline_num_layers=2,
            decoder_last_pipeline_num_layers=4,
            data_parallel_size=1,
        ),
        config,
    )
    result = validate_model_plan(plan)

    spans = [(stage.segments[0].start, stage.segments[0].end) for stage in plan.stages]

    assert spans == [(0, 1), (2, 5), (6, 9)]
    assert result.runtime_mode == "stage-executable"


def test_lower_strategy_to_plan_builds_interleaved_segments_for_vpp(tmp_path):
    _, lower_strategy_to_plan, summarize_plan, validate_model_plan, _, _ = _load_lowering_types()
    config = _make_config(tmp_path, num_layers=18, global_batch_size=12, cards=3)

    plan = lower_strategy_to_plan(
        _strategy(
            data_parallel_size=1,
            pipeline_model_parallel_size=3,
            num_layers_per_virtual_pipeline_stage=2,
            micro_batch_size=2,
            acc_step=6,
        ),
        config,
    )
    result = validate_model_plan(plan)
    summary = summarize_plan(plan)

    assert [len(stage.segments) for stage in plan.stages] == [3, 3, 3]
    assert [(segment.start, segment.end) for segment in plan.stages[0].segments] == [
        (0, 1),
        (6, 7),
        (12, 13),
    ]
    assert result.runtime_mode == "analysis-only"
    assert summary["vpp_stage_segment_counts"] == [3, 3, 3]


def test_summarize_plan_is_json_safe_for_frozen_nested_values(tmp_path):
    _, lower_strategy_to_plan, summarize_plan, _, _, _ = _load_lowering_types()
    config = _make_config(tmp_path, num_layers=8, cards=2)
    plan = lower_strategy_to_plan(
        _strategy(
            data_parallel_size=2,
            nested={"labels": {"a", "b"}, "attrs": {"mode": "stage"}},
        ),
        config,
    )

    stage = plan.stages[0]
    plan_with_metadata = plan.__class__(
        stages=plan.stages,
        transitions=(
            importlib.import_module("flagscale.runner.auto_tuner.plan.schema").TransitionPlan(
                source_stage_id=0,
                target_stage_id=0,
                kind="note",
                metadata={"labels": {"a", "b"}, "attrs": {"mode": "stage"}},
            ),
        ),
        contract=plan.contract,
        total_layers=plan.total_layers,
    )

    summary = summarize_plan(plan_with_metadata)

    assert stage.segments[0].strategy["data_parallel_size"] == 2
    assert summary["stages"][0]["segments"][0]["strategy"]["nested"]["attrs"]["mode"] == "stage"
    assert summary["transitions"][0]["metadata"]["attrs"]["mode"] == "stage"
    json.dumps(summary)


def test_lower_strategy_to_plan_accepts_reasonable_expert_parallel_strategy(tmp_path):
    _, lower_strategy_to_plan, _, validate_model_plan, _, _ = _load_lowering_types()
    config = _make_config(tmp_path, num_layers=8, cards=2)

    plan = lower_strategy_to_plan(
        _strategy(
            data_parallel_size=2,
            expert_model_parallel_size=2,
        ),
        config,
    )
    result = validate_model_plan(plan)

    assert result.runtime_mode == "stage-executable"


def test_searcher_injects_serializable_plan_summary_and_runtime_metadata(tmp_path):
    _, _, _, _, Recorder, Searcher = _load_lowering_types()
    config = _make_config(tmp_path, num_layers=8, cards=2)

    searcher = Searcher(config)

    assert searcher.strategies
    strategy = searcher.strategies[0]
    assert "plan" not in strategy
    assert "plan_summary" in strategy
    assert strategy["runtime_mode"] == "stage-executable"
    assert strategy["runtime_executable"] is True
    assert strategy["plan_summary"]["contract"]["global_batch_size"] == 16

    (tmp_path / "auto_tuner").mkdir()
    recorder = Recorder(config)
    recorder.save([dict(strategy, idx=1, performance=1.0)])
    history = recorder.read()

    assert history[0]["plan_summary"]["stage_count"] == 1


def test_searcher_marks_vpp_strategy_as_analysis_only_and_not_runtime_executable(tmp_path):
    _, _, _, _, _, Searcher = _load_lowering_types()
    config = _make_config(
        tmp_path,
        num_layers=18,
        global_batch_size=12,
        cards=3,
        space_overrides={
            "data_parallel_size": [1],
            "pipeline_model_parallel_size": [3],
            "num_layers_per_virtual_pipeline_stage": [2],
            "micro_batch_size": [2],
            "expert_model_parallel_size": [1],
        },
    )

    searcher = Searcher(config)

    assert searcher.strategies
    strategy = searcher.strategies[0]
    assert strategy["runtime_mode"] == "analysis-only"
    assert strategy["runtime_executable"] is False
    assert strategy["plan_summary"]["vpp_stage_segment_counts"] == [3, 3, 3]
