import pytest

from flagscale.runner.auto_tuner.generate import Generator
from flagscale.runner.auto_tuner.plan.lowering import lower_strategy_to_plan
from flagscale.runner.auto_tuner.plan.runtime import build_segment_hetero_runtime_overrides
from flagscale.runner.auto_tuner.plan.schema import (
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
    assert set(runtime) == {
        "hetero_stage_segment_splits",
        "hetero_stage_segment_meshes",
        "hetero_stage_segment_transitions",
    }
    assert len(runtime["hetero_stage_segment_transitions"]) == len(plan.transitions)


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


def test_generator_materializes_segment_runtime_for_segment_executable_plan(tmp_path):
    config = segment_runtime_config(tmp_path, with_num_layers=True)

    task = Generator(config).gen(
        {
            "idx": 1,
            "data_parallel_size": 1,
            "use_distributed_optimizer": False,
            "tensor_model_parallel_size": 1,
            "sequence_parallel": True,
            "pipeline_model_parallel_size": 2,
            "num_layers_per_virtual_pipeline_stage": None,
            "recompute_method": None,
            "recompute_granularity": None,
            "recompute_num_layers": None,
            "micro_batch_size": 2,
            "acc_step": 4,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "decoder_first_pipeline_num_layers": None,
            "decoder_last_pipeline_num_layers": None,
            "stage_partition_ranges": [[0, 1], [2, 3]],
            "stage_device_groups": [[0, 1], [2, 3]],
            "stage_strategies": [
                {
                    "segment_partition_ranges": [[0, 0], [1, 1]],
                    "segment_strategies": [
                        {
                            "data_parallel_size": 1,
                            "tensor_model_parallel_size": 2,
                            "pipeline_model_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_model_parallel_size": 1,
                            "sequence_parallel": True,
                            "use_distributed_optimizer": False,
                        },
                        {
                            "data_parallel_size": 2,
                            "tensor_model_parallel_size": 1,
                            "pipeline_model_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_model_parallel_size": 1,
                            "sequence_parallel": True,
                            "use_distributed_optimizer": False,
                        },
                    ],
                },
                {
                    "segment_partition_ranges": [[0, 0], [1, 1]],
                    "segment_strategies": [
                        {
                            "data_parallel_size": 2,
                            "tensor_model_parallel_size": 1,
                            "pipeline_model_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_model_parallel_size": 1,
                            "sequence_parallel": True,
                            "use_distributed_optimizer": False,
                        },
                        {
                            "data_parallel_size": 1,
                            "tensor_model_parallel_size": 2,
                            "pipeline_model_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_model_parallel_size": 1,
                            "sequence_parallel": True,
                            "use_distributed_optimizer": False,
                        },
                    ],
                },
            ],
        }
    )

    segment_runtime = task.train.system.hetero.segment_runtime

    assert task.experiment.auto_tuner.plan.plan_kind == "segment-heterogeneous"
    assert task.experiment.auto_tuner.plan.runtime_mode == "segment-executable"
    assert task.experiment.auto_tuner.plan.runtime_executable is False
    assert task.train.system.hetero.enable_hetero is True
    assert task.train.system.hetero.hetero_pipeline_layer_split == [2, 2]
    assert task.train.system.hetero.hetero_process_meshes == [2, 1, 1, 1, 1, 1, 1, 1, 2, 1]
    assert task.train.system.hetero.hetero_device_types == ["nvidia_l20", "nvidia_l20"]
    assert task.experiment.auto_tuner.plan.plan_summary["stages"][0]["segments"][0]["strategy"][
        "pp_local"
    ] == 1
    assert segment_runtime["hetero_stage_segment_splits"] == [[1, 1], [1, 1]]
    assert segment_runtime["hetero_stage_segment_meshes"][0][0] == [2, 1, 1, 1, 1]
    assert (
        segment_runtime["hetero_stage_segment_transitions"][0]["kind"]
        == "segment-redistribution"
    )


def test_generator_uses_lowered_stage_device_types_for_segment_shell(tmp_path):
    config = segment_runtime_config(tmp_path, with_num_layers=True)
    config.train.system.hetero = {"hetero_current_device_type": "mlu290"}
    stage_device_types = ["mlu290", "dcu_z100l"]
    strategy = segment_runtime_strategy()
    strategy["stage_device_types"] = stage_device_types
    for stage_strategy in strategy["stage_strategies"]:
        for segment_strategy in stage_strategy["segment_strategies"]:
            segment_strategy.pop("device_type", None)

    from flagscale.runner.auto_tuner.generate import Generator

    task = Generator(config).gen(strategy)

    assert task.train.system.hetero.enable_hetero is True
    assert task.train.system.hetero.hetero_device_types == stage_device_types
    assert task.train.system.hetero.segment_runtime["hetero_stage_segment_splits"] == [
        [1, 1],
        [1, 1],
    ]


def test_generator_rejects_segment_runtime_with_cross_stage_udo_mismatch(tmp_path):
    config = segment_runtime_config(tmp_path, with_num_layers=True)
    strategy = segment_runtime_strategy()
    for segment_strategy in strategy["stage_strategies"][0]["segment_strategies"]:
        segment_strategy["use_distributed_optimizer"] = True
    for segment_strategy in strategy["stage_strategies"][1]["segment_strategies"]:
        segment_strategy["use_distributed_optimizer"] = False

    with pytest.raises(ValueError, match="use_distributed_optimizer"):
        Generator(config).gen(strategy)


def test_generator_replaces_existing_segment_runtime_subtree(tmp_path):
    config = segment_runtime_config(tmp_path, with_num_layers=True)
    config.train.system.hetero = {
        "legacy_hetero_key": True,
        "segment_runtime": {
            "legacy_key": "stale-value",
        },
    }

    from flagscale.runner.auto_tuner.generate import Generator

    task = Generator(config).gen_best_task(
        {
            "idx": 1,
            "data_parallel_size": 1,
            "use_distributed_optimizer": False,
            "tensor_model_parallel_size": 1,
            "sequence_parallel": True,
            "pipeline_model_parallel_size": 2,
            "num_layers_per_virtual_pipeline_stage": None,
            "recompute_method": None,
            "recompute_granularity": None,
            "recompute_num_layers": None,
            "micro_batch_size": 2,
            "acc_step": 4,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "decoder_first_pipeline_num_layers": None,
            "decoder_last_pipeline_num_layers": None,
            "stage_partition_ranges": [[0, 1], [2, 3]],
            "stage_device_groups": [[0, 1], [2, 3]],
            "stage_strategies": [
                {
                    "segment_partition_ranges": [[0, 0], [1, 1]],
                    "segment_strategies": [
                        {
                            "data_parallel_size": 1,
                            "tensor_model_parallel_size": 2,
                            "pipeline_model_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_model_parallel_size": 1,
                            "sequence_parallel": True,
                            "use_distributed_optimizer": False,
                        },
                        {
                            "data_parallel_size": 2,
                            "tensor_model_parallel_size": 1,
                            "pipeline_model_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_model_parallel_size": 1,
                            "sequence_parallel": True,
                            "use_distributed_optimizer": False,
                        },
                    ],
                },
                {
                    "segment_partition_ranges": [[0, 0], [1, 1]],
                    "segment_strategies": [
                        {
                            "data_parallel_size": 2,
                            "tensor_model_parallel_size": 1,
                            "pipeline_model_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_model_parallel_size": 1,
                            "sequence_parallel": True,
                            "use_distributed_optimizer": False,
                        },
                        {
                            "data_parallel_size": 1,
                            "tensor_model_parallel_size": 2,
                            "pipeline_model_parallel_size": 1,
                            "context_parallel_size": 1,
                            "expert_model_parallel_size": 1,
                            "sequence_parallel": True,
                            "use_distributed_optimizer": False,
                        },
                    ],
                },
            ],
        },
        config,
    )

    segment_runtime = task.train.system.hetero.segment_runtime

    assert task.train.system.hetero.legacy_hetero_key is True
    assert "legacy_key" not in segment_runtime
    assert sorted(segment_runtime.keys()) == sorted(
        [
            "hetero_stage_segment_splits",
            "hetero_stage_segment_meshes",
            "hetero_stage_segment_transitions",
        ]
    )
