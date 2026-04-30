from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.plan.runtime_bridge import apply_hetero_runtime_overrides


def _config(tmp_path):
    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {"nnodes": 1, "nproc_per_node": 2},
                "auto_tuner": {
                    "cards": 2,
                    "chip_profile": {
                        "profile": {"identity": {"name": "nvidia_l20"}},
                    },
                },
            },
            "train": {
                "model": {"num_layers": 2, "global_batch_size": 4},
                "system": {},
            },
        }
    )


def _strategy():
    return {
        "idx": 0,
        "data_parallel_size": 1,
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "micro_batch_size": 2,
        "acc_step": 2,
        "use_distributed_optimizer": False,
        "sequence_parallel": True,
        "num_layers_per_virtual_pipeline_stage": None,
        "recompute_method": None,
        "recompute_granularity": None,
        "recompute_num_layers": None,
        "decoder_first_pipeline_num_layers": None,
        "decoder_last_pipeline_num_layers": None,
        "stage_partition_ranges": [[0, 1]],
        "stage_device_groups": [[0, 1]],
        "stage_strategies": [
            {
                "segment_partition_ranges": [[0, 0], [1, 1]],
                "segment_strategies": [
                    _segment(tp=1, dp=2),
                    _segment(tp=2, dp=1),
                ],
            }
        ],
    }


def _segment(*, tp, dp):
    return {
        "data_parallel_size": dp,
        "tensor_model_parallel_size": tp,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "sequence_parallel": True,
        "use_distributed_optimizer": False,
    }


def test_segment_runtime_stage_shell_uses_max_tensor_parallel_segment(tmp_path):
    config = _config(tmp_path)

    apply_hetero_runtime_overrides(_strategy(), config, "segment-executable")

    assert config.train.system.hetero.hetero_process_meshes == [2, 1, 1, 1, 1]
    assert config.train.system.tensor_model_parallel_size == 2
    assert config.train.system.sequence_parallel is True
