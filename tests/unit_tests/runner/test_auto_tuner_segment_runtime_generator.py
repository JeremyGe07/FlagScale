from pathlib import Path
import sys

import pytest

from flagscale.runner.auto_tuner.generate import Generator

TEST_UTILS_DIR = Path(__file__).resolve().parent
if str(TEST_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_UTILS_DIR))

from segment_runtime_test_utils import (
    segment_runtime_config,
    segment_runtime_strategy,
)


def test_generator_marks_raw_segment_executable_plan_runtime_executable(tmp_path):
    config = segment_runtime_config(tmp_path, with_num_layers=True)

    metadata = Generator(config)._build_plan_runtime_metadata(segment_runtime_strategy(), config)

    assert metadata["plan_kind"] == "segment-heterogeneous"
    assert metadata["runtime_mode"] == "segment-executable"
    assert metadata["runtime_executable"] is True


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
    assert task.experiment.auto_tuner.plan.runtime_executable is True
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
