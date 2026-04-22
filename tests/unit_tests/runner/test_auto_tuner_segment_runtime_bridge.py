import pytest

from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.plan.lowering import lower_strategy_to_plan
from flagscale.runner.auto_tuner.plan.schema import (
    ModelPlan,
    SegmentPlan,
    StagePlan,
    TransitionPlan,
)
from flagscale.runner.auto_tuner.plan.summary import (
    SEGMENT_HETEROGENEOUS_PLAN,
    plan_kind,
    summarize_plan,
)
from flagscale.runner.auto_tuner.plan.runtime_bridge import apply_hetero_runtime_overrides
from flagscale.runner.auto_tuner.plan.validator import validate_model_plan


def _chip_profile():
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


def _config(tmp_path):
    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path),
                "runner": {"nnodes": 1, "nproc_per_node": 4},
                "auto_tuner": {
                    "cards": 4,
                    "nnodes": 1,
                    "nproc_per_node": 4,
                    "chip_profile": {"profile": _chip_profile()},
                },
            },
            "train": {
                "model": {
                    "num_layers": 4,
                    "global_batch_size": 8,
                    "hidden_size": 8,
                    "num_attention_heads": 4,
                    "seq_length": 16,
                },
                "system": {},
            },
        }
    )


def _segment_strategy(*, tp, dp):
    return {
        "context_parallel_size": 1,
        "data_parallel_size": dp,
        "device_type": "nvidia_l20",
        "expert_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": True,
        "tensor_model_parallel_size": tp,
        "use_distributed_optimizer": False,
    }


def _segment_runtime_strategy():
    stage0_segments = (
        (0, 0, _segment_strategy(tp=2, dp=1)),
        (1, 1, _segment_strategy(tp=1, dp=2)),
    )
    stage1_segments = (
        (2, 2, _segment_strategy(tp=1, dp=2)),
        (3, 3, _segment_strategy(tp=2, dp=1)),
    )
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
                "segment_strategies": [dict(segment) for _, _, segment in stage0_segments],
            },
            {
                "segment_partition_ranges": [[0, 0], [1, 1]],
                "segment_strategies": [dict(segment) for _, _, segment in stage1_segments],
            },
        ],
    }


def _segment_runtime_strategy_without_stage_device_groups():
    strategy = _segment_runtime_strategy()
    strategy.pop("stage_device_groups")
    return strategy


@pytest.mark.parametrize(
    ("stage_strategy_field", "expected_message"),
    [
        ("segment_partition_ranges", "provided together"),
        ("segment_strategies", "provided together"),
    ],
)
def test_lower_strategy_to_plan_rejects_half_configured_segment_metadata(
    tmp_path,
    stage_strategy_field,
    expected_message,
):
    config = _config(tmp_path)
    strategy = _segment_runtime_strategy()
    for stage_strategy in strategy["stage_strategies"]:
        stage_strategy.pop(stage_strategy_field)

    with pytest.raises(ValueError, match=expected_message):
        lower_strategy_to_plan(strategy, config)


@pytest.mark.parametrize(
    ("segment_partition_ranges", "segment_strategies", "expected_message"),
    [
        ([[-1, 0], [1, 1]], None, "non-negative"),
        ([[0, 0], [1, 2]], None, "within stage range"),
        ([[0, 0], [0, 1]], None, "contiguous"),
        ([[0, 0]], [dict(_segment_strategy(tp=2, dp=1))], "exactly cover"),
        ([[True, 0], [1, 1]], None, "non-bool int"),
        ([[1.0, 0], [1, 1]], None, "non-bool int"),
        ([["1", 0], [1, 1]], None, "non-bool int"),
    ],
)
def test_lower_strategy_to_plan_rejects_malformed_nested_segment_ranges(
    tmp_path,
    segment_partition_ranges,
    segment_strategies,
    expected_message,
):
    config = _config(tmp_path)
    strategy = _segment_runtime_strategy()
    strategy["stage_strategies"][0]["segment_partition_ranges"] = segment_partition_ranges
    if segment_strategies is not None:
        strategy["stage_strategies"][0]["segment_strategies"] = segment_strategies

    with pytest.raises(ValueError, match=expected_message):
        lower_strategy_to_plan(strategy, config)


def test_lower_strategy_to_plan_builds_segment_heterogeneous_plan_from_explicit_stage_metadata(
    tmp_path,
):
    config = _config(tmp_path)
    plan = lower_strategy_to_plan(_segment_runtime_strategy(), config)
    validation = validate_model_plan(plan)
    summary = summarize_plan(plan)

    assert plan_kind(plan) == SEGMENT_HETEROGENEOUS_PLAN
    assert validation.runtime_mode == "segment-executable"
    assert [(len(stage.segments)) for stage in plan.stages] == [2, 2]
    assert [(segment.start, segment.end) for segment in plan.stages[0].segments] == [
        (0, 0),
        (1, 1),
    ]
    assert plan.stages[0].segments[0].strategy["pp_local"] == 1
    assert plan.stages[0].segments[1].strategy["pp_local"] == 1
    assert summary["transitions"][0]["metadata"]["source_mesh"]["tp"] == 2
    assert summary["transitions"][1]["metadata"]["target_mesh"]["tp"] == 2
def test_lower_strategy_to_plan_defaults_stage_device_groups_from_nested_segment_metadata(
    tmp_path,
):
    config = _config(tmp_path)
    plan = lower_strategy_to_plan(_segment_runtime_strategy_without_stage_device_groups(), config)

    assert plan.stages[0].device_group == (0, 1)
    assert plan.stages[1].device_group == (2, 3)


def test_lower_strategy_to_plan_rejects_inconsistent_nested_segment_world_size(tmp_path):
    config = _config(tmp_path)
    strategy = _segment_runtime_strategy_without_stage_device_groups()
    strategy["stage_strategies"][1]["segment_strategies"][0]["data_parallel_size"] = 4

    with pytest.raises(ValueError, match="world size"):
        lower_strategy_to_plan(strategy, config)


def test_apply_hetero_runtime_overrides_materializes_stage_shell_and_segment_runtime(tmp_path):
    config = _config(tmp_path)
    strategy = _segment_runtime_strategy()

    apply_hetero_runtime_overrides(strategy, config, "segment-executable")

    assert config.train.system.hetero.enable_hetero is True
    assert config.train.system.hetero.hetero_pipeline_layer_split == [2, 2]
    assert config.train.system.hetero.hetero_process_meshes == [2, 1, 1, 1, 1, 1, 1, 1, 2, 1]
    assert config.train.system.hetero.hetero_device_types == ["nvidia_l20", "nvidia_l20"]
    assert config.train.system.hetero.segment_runtime["hetero_stage_segment_splits"] == [
        [1, 1],
        [1, 1],
    ]


@pytest.mark.parametrize(
    ("segment_strategy_update", "expected_message"),
    [
        ({"pp_local": 2}, "segment-local pipeline size must be a non-bool int equal to 1"),
        ({"pp_local": True}, "segment-local pipeline size must be a non-bool int equal to 1"),
        ({"pp_local": 1.0}, "segment-local pipeline size must be a non-bool int equal to 1"),
        (
            {"pipeline_model_parallel_size": 2},
            "segment-local pipeline size must be a non-bool int equal to 1",
        ),
        (
            {"pipeline_model_parallel_size": True},
            "segment-local pipeline size must be a non-bool int equal to 1",
        ),
        (
            {"pipeline_model_parallel_size": 1.0},
            "segment-local pipeline size must be a non-bool int equal to 1",
        ),
    ],
)
def test_lower_strategy_to_plan_rejects_non_unit_segment_local_pipeline_semantics(
    tmp_path,
    segment_strategy_update,
    expected_message,
):
    config = _config(tmp_path)
    strategy = _segment_runtime_strategy()
    stage_strategy = strategy["stage_strategies"][0]["segment_strategies"][0]
    for key, value in segment_strategy_update.items():
        stage_strategy[key] = value

    with pytest.raises(ValueError, match=expected_message):
        lower_strategy_to_plan(strategy, config)
