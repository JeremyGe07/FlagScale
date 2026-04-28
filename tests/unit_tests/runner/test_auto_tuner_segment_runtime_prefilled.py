from pathlib import Path
import sys

import pytest

from flagscale.runner.auto_tuner.generate import Generator

TEST_UTILS_DIR = Path(__file__).resolve().parent
if str(TEST_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_UTILS_DIR))

from segment_runtime_test_utils import segment_runtime_config


def _segment_runtime_strategy():
    return {
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
        "plan_kind": "segment-heterogeneous",
        "stage_count": 2,
        "segment_count": 4,
        "runtime_mode": "segment-executable",
        "runtime_executable": True,
        "execution_contract": {
            "world_size": 2,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "global_batch_size": 8,
        },
        "plan_summary": {
            "stage_count": 2,
            "vpp_stage_segment_counts": [2, 2],
            "contract": {"global_batch_size": 8},
        },
    }


def test_generator_rejects_unsupported_segment_heterogeneous_plan_execution(tmp_path):
    config = segment_runtime_config(tmp_path, with_eval_iters=True)

    with pytest.raises(ValueError, match="segment-heterogeneous"):
        Generator(config).gen(
            {
                "idx": 1,
                "data_parallel_size": 1,
                "use_distributed_optimizer": False,
                "tensor_model_parallel_size": 1,
                "sequence_parallel": False,
                "pipeline_model_parallel_size": 1,
                "num_layers_per_virtual_pipeline_stage": None,
                "recompute_method": None,
                "recompute_granularity": None,
                "recompute_num_layers": None,
                "micro_batch_size": 2,
                "context_parallel_size": 1,
                "expert_model_parallel_size": 1,
                "decoder_first_pipeline_num_layers": None,
                "decoder_last_pipeline_num_layers": None,
                "plan_kind": "segment-heterogeneous",
                "stage_count": 2,
                "segment_count": 4,
                "runtime_mode": "analysis-only",
                "runtime_executable": False,
                "execution_contract": {
                    "world_size": 2,
                    "micro_batch_size": 2,
                    "gradient_accumulation_steps": 4,
                    "global_batch_size": 8,
                },
                "plan_summary": {
                    "stage_count": 2,
                    "vpp_stage_segment_counts": [2, 2],
                    "contract": {"global_batch_size": 8},
                },
            }
        )


def test_generator_reports_segment_heterogeneous_metadata_as_executable(tmp_path):
    config = segment_runtime_config(tmp_path, with_eval_iters=True)

    metadata = Generator(config)._build_plan_runtime_metadata(
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
            "plan_kind": "segment-heterogeneous",
            "stage_count": 2,
            "segment_count": 4,
            "runtime_mode": "segment-executable",
            "runtime_executable": True,
            "execution_contract": {
                "world_size": 2,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "global_batch_size": 8,
            },
            "plan_summary": {
                "stage_count": 2,
                "vpp_stage_segment_counts": [2, 2],
                "contract": {"global_batch_size": 8},
            },
        },
        config,
    )

    assert metadata["plan_kind"] == "segment-heterogeneous"
    assert metadata["runtime_mode"] == "segment-executable"
    assert metadata["runtime_executable"] is True


def test_generator_rejects_prefilled_segment_metadata_without_raw_stage_metadata(tmp_path):
    config = segment_runtime_config(tmp_path, with_eval_iters=True)

    with pytest.raises(ValueError, match="segment runtime bridge requires"):
        Generator(config).gen(_segment_runtime_strategy())


def test_generator_rejects_prefilled_segment_metadata_that_disagrees_with_raw_segment_strategy(
    tmp_path,
):
    config = segment_runtime_config(tmp_path, with_num_layers=True)
    strategy = _segment_runtime_strategy()
    strategy.update(
        {
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
            "segment_count": 999,
        }
    )

    with pytest.raises(ValueError, match="prefilled plan metadata.*segment_count"):
        Generator(config).gen(strategy)


def test_generator_rejects_incomplete_prefilled_plan_metadata(tmp_path):
    config = segment_runtime_config(tmp_path, with_eval_iters=True)

    with pytest.raises(ValueError, match="plan metadata is incomplete"):
        Generator(config).gen(
            {
                "idx": 1,
                "data_parallel_size": 1,
                "use_distributed_optimizer": False,
                "tensor_model_parallel_size": 1,
                "sequence_parallel": False,
                "pipeline_model_parallel_size": 1,
                "num_layers_per_virtual_pipeline_stage": None,
                "recompute_method": None,
                "recompute_granularity": None,
                "recompute_num_layers": None,
                "micro_batch_size": 2,
                "context_parallel_size": 1,
                "expert_model_parallel_size": 1,
                "decoder_first_pipeline_num_layers": None,
                "decoder_last_pipeline_num_layers": None,
                "plan_kind": "homogeneous",
                "plan_summary": {"stage_count": 1},
            }
        )


def test_generator_rejects_non_bool_runtime_executable_in_prefilled_metadata(tmp_path):
    config = segment_runtime_config(tmp_path, with_eval_iters=True)

    with pytest.raises(ValueError, match="runtime_executable must be bool"):
        Generator(config).gen(
            {
                "idx": 1,
                "data_parallel_size": 1,
                "use_distributed_optimizer": False,
                "tensor_model_parallel_size": 1,
                "sequence_parallel": False,
                "pipeline_model_parallel_size": 1,
                "num_layers_per_virtual_pipeline_stage": None,
                "recompute_method": None,
                "recompute_granularity": None,
                "recompute_num_layers": None,
                "micro_batch_size": 2,
                "context_parallel_size": 1,
                "expert_model_parallel_size": 1,
                "decoder_first_pipeline_num_layers": None,
                "decoder_last_pipeline_num_layers": None,
                "plan_kind": "homogeneous",
                "stage_count": 1,
                "segment_count": 1,
                "runtime_mode": "stage-executable",
                "runtime_executable": "False",
                "execution_contract": {
                    "world_size": 1,
                    "micro_batch_size": 2,
                    "gradient_accumulation_steps": 4,
                    "global_batch_size": 8,
                },
                "plan_summary": {
                    "stage_count": 1,
                    "vpp_stage_segment_counts": [1],
                    "contract": {"global_batch_size": 8},
                },
            }
        )
