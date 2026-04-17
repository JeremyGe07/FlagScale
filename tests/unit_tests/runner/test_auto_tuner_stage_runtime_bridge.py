import pytest

from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.generate import Generator
from flagscale.runner.auto_tuner.plan.lowering import lower_strategy_to_plan
from flagscale.runner.auto_tuner.plan.summary import plan_kind, segment_count, summarize_execution_contract, summarize_plan
from flagscale.runner.auto_tuner.plan.validator import validate_model_plan


def _chip_profile():
    return {
        "identity": {"name": "nvidia_l20", "vendor": "nvidia", "chip_class": "gpu"},
        "memory": {"total_memory_mb": 46000},
        "compute": {"bf16_tflops": 119.5},
        "interconnect": {"intra_node": {"fabric": "pcie"}},
        "topology": {"max_nodes": 1, "devices_per_node": 4, "homogeneous_only": True},
        "strategy_hints": {
            "default_search_priority": "performance",
            "max_tensor_model_parallel_size": 4,
            "max_pipeline_model_parallel_size": 4,
            "disabled_dims": {},
        },
        "cost_model": {
            "reserved_memory_bias_mb": 0,
            "peak_activation_bias_mb": 0,
            "overlap": {
                "dp_comm_overlap_ratio": 0.0,
                "tp_comm_overlap_ratio": 0.0,
                "pp_comm_overlap_ratio": 0.0,
            },
        },
    }


def _config(tmp_path, *, with_profile=False):
    auto_tuner = {
        "control": {"train_iters": 3},
        "cards": 4,
        "nnodes": 1,
        "nproc_per_node": 4,
    }
    if with_profile:
        auto_tuner["chip_profile"] = {"profile": _chip_profile()}
    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {"nnodes": 1, "nproc_per_node": 4},
                "auto_tuner": auto_tuner,
            },
            "train": {
                "system": {
                    "logging": {},
                    "checkpoint": {"save_interval": 100},
                },
                "model": {
                    "num_layers": 4,
                    "global_batch_size": 8,
                    "hidden_size": 8,
                    "num_attention_heads": 4,
                    "seq_length": 16,
                    "eval_iters": 10,
                    "optimizer": {"lr_scheduler": {"lr": 1e-5, "min_lr": 0}},
                },
            },
        }
    )


def _stage_strategy(
    *,
    dp,
    tp,
    sequence_parallel=True,
    use_recompute=False,
    use_distributed_optimizer=False,
):
    return {
        "data_parallel_size": dp,
        "tensor_model_parallel_size": tp,
        "pipeline_model_parallel_size": 2,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "sequence_parallel": sequence_parallel,
        "use_distributed_optimizer": use_distributed_optimizer,
        "num_layers_per_virtual_pipeline_stage": None,
        "use_recompute": use_recompute,
        "recompute_method": None,
        "recompute_granularity": None,
        "recompute_num_layers": None,
        "micro_batch_size": 2,
        "acc_step": 4,
    }


def _stage_hetero_strategy(**overrides):
    strategy = {
        "idx": 1,
        "data_parallel_size": 1,
        "use_distributed_optimizer": False,
        "tensor_model_parallel_size": 2,
        "sequence_parallel": True,
        "pipeline_model_parallel_size": 2,
        "num_layers_per_virtual_pipeline_stage": None,
        "recompute_method": None,
        "recompute_granularity": None,
        "recompute_num_layers": None,
        "micro_batch_size": 2,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "decoder_first_pipeline_num_layers": None,
        "decoder_last_pipeline_num_layers": None,
        "acc_step": 4,
        "stage_partition_ranges": [[0, 1], [2, 3]],
        "stage_device_groups": [[0, 1], [2, 3]],
        "stage_strategies": [
            _stage_strategy(dp=1, tp=2, sequence_parallel=True),
            _stage_strategy(dp=2, tp=1, sequence_parallel=True),
        ],
    }
    strategy.update(overrides)
    return strategy


def _dp_style_stage_hetero_strategy(config):
    strategy = _stage_hetero_strategy()
    plan = lower_strategy_to_plan(strategy, config)
    strategy.update(
        {
            "plan_kind": plan_kind(plan),
            "stage_count": len(plan.stages),
            "segment_count": segment_count(plan),
            "runtime_mode": "stage-executable",
            "runtime_executable": True,
            "execution_contract": summarize_execution_contract(plan),
            "plan_summary": summarize_plan(plan),
        }
    )
    return strategy


def test_lower_strategy_to_plan_builds_stage_heterogeneous_plan_from_explicit_stage_metadata(
    tmp_path,
):
    config = _config(tmp_path)

    plan = lower_strategy_to_plan(_stage_hetero_strategy(), config)
    result = validate_model_plan(plan)

    spans = [(stage.segments[0].start, stage.segments[0].end) for stage in plan.stages]

    assert spans == [(0, 1), (2, 3)]
    assert plan.stages[0].device_group == (0, 1)
    assert plan.stages[1].device_group == (2, 3)
    assert plan.stages[0].segments[0].strategy["tensor_model_parallel_size"] == 2
    assert plan.stages[1].segments[0].strategy["data_parallel_size"] == 2
    assert result.runtime_mode == "stage-executable"


def test_validate_model_plan_accepts_stage_heterogeneous_tp_dp_variation(tmp_path):
    config = _config(tmp_path)

    plan = lower_strategy_to_plan(_stage_hetero_strategy(), config)
    result = validate_model_plan(plan)

    assert result.runtime_mode == "stage-executable"


def test_validate_model_plan_rejects_tp_mismatch_without_sequence_parallel(tmp_path):
    config = _config(tmp_path)
    strategy = _stage_hetero_strategy(
        stage_strategies=[
            _stage_strategy(dp=1, tp=2, sequence_parallel=False),
            _stage_strategy(dp=2, tp=1, sequence_parallel=False),
        ]
    )

    with pytest.raises(ValueError, match="sequence_parallel"):
        validate_model_plan(lower_strategy_to_plan(strategy, config))


def test_generator_materializes_stage_hetero_runtime_config(tmp_path):
    config = _config(tmp_path, with_profile=True)

    task = Generator(config).gen(_stage_hetero_strategy())

    assert task.experiment.auto_tuner.plan.plan_kind == "stage-heterogeneous"
    assert task.train.system.hetero.enable_hetero is True
    assert task.train.system.hetero.hetero_pipeline_layer_split == [2, 2]
    assert task.train.system.hetero.hetero_process_meshes == [2, 1, 1, 1, 1, 1, 1, 1, 2, 1]
    assert task.train.system.hetero.hetero_device_types == ["nvidia_l20", "nvidia_l20"]
    assert task.train.system.hetero.hetero_current_device_type == "nvidia_l20"
    assert task.train.system.pipeline_model_parallel_size == 2
    assert task.train.system.tensor_model_parallel_size == 2
    assert task.train.system.sequence_parallel is True


def test_generator_materializes_dp_style_stage_chain_runtime_config(tmp_path):
    config = _config(tmp_path, with_profile=True)

    task = Generator(config).gen(_dp_style_stage_hetero_strategy(config))

    assert task.experiment.auto_tuner.plan.plan_kind == "stage-heterogeneous"
    assert task.experiment.auto_tuner.plan.runtime_mode == "stage-executable"
    assert task.train.system.hetero.enable_hetero is True
    assert task.train.system.hetero.hetero_pipeline_layer_split == [2, 2]
    assert task.train.system.hetero.hetero_process_meshes == [2, 1, 1, 1, 1, 1, 1, 1, 2, 1]
    assert task.train.system.hetero.hetero_device_types == ["nvidia_l20", "nvidia_l20"]
    assert task.train.system.hetero.hetero_current_device_type == "nvidia_l20"


def test_generator_rejects_stage_heterogeneous_recompute_mismatch(tmp_path):
    config = _config(tmp_path, with_profile=True)
    strategy = _stage_hetero_strategy(
        stage_strategies=[
            _stage_strategy(dp=1, tp=2, sequence_parallel=True, use_recompute=False),
            _stage_strategy(dp=2, tp=1, sequence_parallel=True, use_recompute=True),
        ]
    )

    with pytest.raises(ValueError, match="recompute"):
        Generator(config).gen(strategy)


def test_generator_rejects_missing_stage_device_types_without_profile(tmp_path):
    config = _config(tmp_path, with_profile=False)

    with pytest.raises(ValueError, match="hetero_device_types|device type"):
        Generator(config).gen(_stage_hetero_strategy())
