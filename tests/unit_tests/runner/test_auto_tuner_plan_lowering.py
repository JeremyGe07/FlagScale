import importlib

from omegaconf import OmegaConf


def _load_lowering_types():
    lowering_module = importlib.import_module("flagscale.runner.auto_tuner.plan.lowering")
    validator_module = importlib.import_module("flagscale.runner.auto_tuner.plan.validator")
    searcher_module = importlib.import_module("flagscale.runner.auto_tuner.search.searcher")
    return (
        lowering_module.build_execution_contract,
        lowering_module.lower_strategy_to_plan,
        validator_module.validate_model_plan,
        searcher_module.Searcher,
    )


def _make_config(tmp_path, *, num_layers=8, global_batch_size=16, cards=4):
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
                    "space": {
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
                    },
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
    build_execution_contract, _, _, _ = _load_lowering_types()
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
    _, lower_strategy_to_plan, validate_model_plan, _ = _load_lowering_types()
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
    _, lower_strategy_to_plan, validate_model_plan, _ = _load_lowering_types()
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


def test_lower_strategy_to_plan_honors_decoder_first_and_last_stage_sizes(tmp_path):
    _, lower_strategy_to_plan, validate_model_plan, _ = _load_lowering_types()
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


def test_searcher_injects_plan_and_runtime_metadata(tmp_path):
    _, _, _, Searcher = _load_lowering_types()
    config = _make_config(tmp_path, num_layers=8, cards=2)

    searcher = Searcher(config)

    assert searcher.strategies
    strategy = searcher.strategies[0]
    assert "plan" in strategy
    assert strategy["runtime_mode"] == "stage-executable"
    assert strategy["runtime_executable"] is True
    assert strategy["plan"].contract.global_batch_size == 16
