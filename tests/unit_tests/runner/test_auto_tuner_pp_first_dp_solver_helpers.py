from omegaconf import OmegaConf


def chip_profile(total_memory_mb=46000):
    return {
        "identity": {"name": "nvidia_l20", "vendor": "nvidia", "chip_class": "gpu"},
        "memory": {"total_memory_mb": total_memory_mb, "bandwidth_gbps": 864},
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


def config(tmp_path, *, cards=4, global_batch_size=8, micro_batch_size=(1, 2), total_memory_mb=46000):
    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {"nnodes": 1, "nproc_per_node": cards},
                "auto_tuner": {
                    "algo": {"name": "grid", "use_profiled_time_cost": True},
                    "cards": cards,
                    "chip_profile": {"profile": chip_profile(total_memory_mb=total_memory_mb)},
                    "planner": {"max_stage_candidates_per_stage": 2},
                    "space": {
                        "data_parallel_size": [1, 2, 4],
                        "use_distributed_optimizer": [False, True],
                        "tensor_model_parallel_size": [1, 2, 4],
                        "sequence_parallel": [False, True],
                        "pipeline_model_parallel_size": [1, 2, 4],
                        "num_layers_per_virtual_pipeline_stage": [0],
                        "context_parallel_size": [1],
                        "expert_model_parallel_size": [1],
                        "micro_batch_size": list(micro_batch_size),
                        "use_recompute": [False, True],
                        "recompute_method": ["uniform", "block"],
                        "recompute_granularity": ["full", "selective"],
                        "recompute_num_layers": [1, 2, 4],
                    },
                },
            },
            "train": {
                "system": {"logging": {}, "checkpoint": {"save_interval": 100}},
                "model": {
                    "num_layers": 10,
                    "global_batch_size": global_batch_size,
                    "hidden_size": 64,
                    "num_attention_heads": 8,
                    "seq_length": 32,
                    "padded_vocab_size": 256,
                    "eval_iters": 0,
                    "optimizer": {"lr_scheduler": {"lr": 1e-5, "min_lr": 0}},
                },
            },
        }
    )


def stage_candidate(
    *,
    dp,
    tp,
    pp=2,
    sp=False,
    mb=2,
    acc_step=4,
):
    return {
        "data_parallel_size": dp,
        "use_distributed_optimizer": False,
        "tensor_model_parallel_size": tp,
        "sequence_parallel": sp,
        "pipeline_model_parallel_size": pp,
        "num_layers_per_virtual_pipeline_stage": None,
        "use_recompute": False,
        "recompute_method": None,
        "recompute_granularity": None,
        "recompute_num_layers": None,
        "micro_batch_size": mb,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "acc_step": acc_step,
        "decoder_first_pipeline_num_layers": None,
        "decoder_last_pipeline_num_layers": None,
    }
