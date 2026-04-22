from omegaconf import OmegaConf


def chip_profile():
    return {
        "identity": {"name": "nvidia_l20", "vendor": "nvidia", "chip_class": "gpu"},
        "memory": {"total_memory_mb": 46000, "bandwidth_gbps": 864},
        "compute": {"bf16_tflops": 119.5, "attention_tflops": 119.5},
        "interconnect": {
            "intra_node": {
                "fabric": "pcie",
                "p2p_bandwidth_gbps": 64,
                "p2p_latency_us": 3,
                "all_reduce_bandwidth_gbps": 45,
                "all_reduce_latency_us": 8,
            },
            "host_device": {"bandwidth_gbps": 24, "latency_us": 10},
        },
        "kernel_support": {
            "transformer_engine": True,
            "flash_attention": True,
            "fused_rmsnorm": True,
        },
        "topology": {"max_nodes": 1, "devices_per_node": 4, "homogeneous_only": True},
        "strategy_hints": {
            "default_search_priority": "performance",
            "max_tensor_model_parallel_size": 4,
            "max_pipeline_model_parallel_size": 4,
            "disabled_dims": {},
        },
    }


def segment_runtime_config(
    tmp_path,
    *,
    with_num_layers=False,
    with_runner=True,
    with_eval_iters=True,
    with_checkpoint=True,
):
    experiment = {
        "exp_dir": str(tmp_path),
        "auto_tuner": {
            "cards": 4,
            "nnodes": 1,
            "nproc_per_node": 4,
            "chip_profile": {"profile": chip_profile()},
        },
    }
    if with_runner:
        experiment["runner"] = {"nnodes": 1, "nproc_per_node": 4}
    model = {
        "global_batch_size": 8,
        "hidden_size": 8,
        "num_attention_heads": 4,
        "seq_length": 16,
        "optimizer": {
            "lr_scheduler": {
                "lr": 1e-5,
                "min_lr": 0,
            }
        },
    }
    if with_num_layers:
        model["num_layers"] = 4
    if with_eval_iters:
        model["eval_iters"] = 10
    train_system = {"logging": {}}
    if with_checkpoint:
        train_system["checkpoint"] = {"save_interval": 100}
    return OmegaConf.create(
        {
            "experiment": experiment,
            "train": {
                "model": model,
                "system": train_system,
            },
        }
    )


def segment_strategy(*, tp, dp, pipeline_model_parallel_size=1):
    return {
        "context_parallel_size": 1,
        "data_parallel_size": dp,
        "device_type": "nvidia_l20",
        "expert_model_parallel_size": 1,
        "pipeline_model_parallel_size": pipeline_model_parallel_size,
        "sequence_parallel": True,
        "tensor_model_parallel_size": tp,
        "use_distributed_optimizer": False,
    }


def segment_runtime_strategy():
    return {
        "idx": 0,
        "data_parallel_size": 1,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 2,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "micro_batch_size": 2,
        "acc_step": 4,
        "use_distributed_optimizer": False,
        "sequence_parallel": True,
        "num_layers_per_virtual_pipeline_stage": None,
        "recompute_method": None,
        "recompute_granularity": None,
        "recompute_num_layers": None,
        "decoder_first_pipeline_num_layers": None,
        "decoder_last_pipeline_num_layers": None,
        "stage_partition_ranges": [[0, 1], [2, 3]],
        "stage_device_groups": [[0, 1], [2, 3]],
        "stage_strategies": [
            {
                "segment_partition_ranges": [[0, 0], [1, 1]],
                "segment_strategies": [
                    segment_strategy(tp=2, dp=1),
                    segment_strategy(tp=1, dp=2),
                ],
            },
            {
                "segment_partition_ranges": [[0, 0], [1, 1]],
                "segment_strategies": [
                    segment_strategy(tp=1, dp=2),
                    segment_strategy(tp=2, dp=1),
                ],
            },
        ],
    }


def segment_runtime_strategy_without_stage_device_groups():
    strategy = segment_runtime_strategy()
    strategy.pop("stage_device_groups")
    return strategy
