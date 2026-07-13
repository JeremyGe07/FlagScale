from copy import deepcopy
from types import SimpleNamespace

from flagscale.runner.auto_tuner.chip_profile import get_chip_profile_or_none
from flagscale.runner.auto_tuner.plan.lowering import lower_strategy_to_plan
from flagscale.runner.auto_tuner.plan.schema import ModelPlan
from flagscale.runner.auto_tuner.plan.summary import extract_homogeneous_strategy, to_json_safe
from flagscale.runner.auto_tuner.utils import normalize_moe_layer_freq
from flagscale.train.theoretical_memory_usage import (
    NUM_BYTES_IN_MEGABYTE,
    compute_activation_memory,
    report_theoretical_memory,
)

DEFAULT_MEMORY_BREAKDOWN = {
    "parameters_mb": 0.0,
    "model_states_mb": 0.0,
    "activations_mb": 0.0,
    "recompute_saved_mb": 0.0,
    "peak_mb": 0.0,
    "reserved_mb": 0.0,
    "peak_activation_bias_mb": 0.0,
    "profiled_peak_mb": 0.0,
    "profiled_memory_total_mb": 0.0,
}
FULL_MODEL_STATE_MULTIPLIER = 18.0
SHARDED_MODEL_STATE_BASE = 6.0
SHARDED_MODEL_STATE_GRAD_FACTOR = 12.0


def estimate_memory_cost(strategy, config):
    plan = (
        strategy
        if isinstance(strategy, ModelPlan)
        else lower_strategy_to_plan(strategy, config, validate=False)
    )
    return _estimate_plan_memory_cost(plan, config)


def _estimate_strategy_memory_cost(strategy, config):
    args = _build_memory_args(config, strategy)
    num_microbatches = _get_num_microbatches(strategy, config)
    base_total_mb = float(report_theoretical_memory(args, num_microbatches=num_microbatches))
    actual_activation_mb = _activation_memory_mb(args, num_microbatches)
    disabled_recompute_activation_mb = _activation_without_recompute_mb(args, num_microbatches)
    model_states_mb = max(base_total_mb - actual_activation_mb, 0.0)
    recompute_saved_mb = max(disabled_recompute_activation_mb - actual_activation_mb, 0.0)
    reserved_mb, peak_activation_bias_mb = _read_biases(config)
    peak_mb = base_total_mb
    profiled_peak_mb = peak_mb + peak_activation_bias_mb

    breakdown = deepcopy(DEFAULT_MEMORY_BREAKDOWN)
    breakdown["model_states_mb"] = model_states_mb
    breakdown["parameters_mb"] = _estimate_parameter_memory_mb(model_states_mb, strategy)
    breakdown["activations_mb"] = max(peak_mb - model_states_mb, 0.0)
    breakdown["recompute_saved_mb"] = recompute_saved_mb
    breakdown["peak_mb"] = peak_mb
    breakdown["reserved_mb"] = reserved_mb
    breakdown["peak_activation_bias_mb"] = peak_activation_bias_mb
    breakdown["profiled_peak_mb"] = profiled_peak_mb
    breakdown["profiled_memory_total_mb"] = profiled_peak_mb + reserved_mb
    return {
        "memory_total_mb": peak_mb + reserved_mb,
        "memory_breakdown": breakdown,
    }


def _estimate_plan_memory_cost(plan, config):
    strategy = extract_homogeneous_strategy(plan)
    if strategy is not None:
        result = _estimate_strategy_memory_cost(strategy, config)
        result["memory_breakdown"]["plan"] = _build_plan_memory_breakdown(plan, config)
        return result
    plan_breakdown = _build_plan_memory_breakdown(plan, config)
    memory_total_mb = plan_breakdown["peak_combined_memory_total_mb"]
    peak_stage_breakdown = _peak_combined_stage_breakdown(plan_breakdown)
    breakdown = deepcopy(DEFAULT_MEMORY_BREAKDOWN)
    breakdown["parameters_mb"] = peak_stage_breakdown["parameters_mb"]
    breakdown["model_states_mb"] = peak_stage_breakdown["model_states_mb"]
    breakdown["recompute_saved_mb"] = peak_stage_breakdown["recompute_saved_mb"]
    breakdown["peak_mb"] = plan_breakdown["peak_combined_peak_mb"]
    breakdown["activations_mb"] = max(
        breakdown["peak_mb"] - breakdown["model_states_mb"],
        0.0,
    )
    breakdown["reserved_mb"] = plan_breakdown["peak_combined_reserved_mb"]
    breakdown["plan"] = plan_breakdown
    return {"memory_total_mb": memory_total_mb, "memory_breakdown": breakdown}


def _build_plan_memory_breakdown(plan, config):
    stages = [_build_stage_memory(stage, config) for stage in plan.stages]
    transitions = _build_transition_memory(plan, config)
    peak_stage = max(stages, key=lambda stage: stage["memory_total_mb"], default=None)
    peak_transition = max(
        transitions,
        key=lambda transition: transition["memory_mb"],
        default=None,
    )
    peak_combined_stage, peak_combined_transition_mb = _peak_combined_stage(stages, transitions)
    peak_combined = (
        0.0
        if peak_combined_stage is None
        else peak_combined_stage["memory_total_mb"] + peak_combined_transition_mb
    )
    peak_combined_reserved_mb = (
        0.0
        if peak_combined_stage is None
        else peak_combined_stage["memory_breakdown"]["reserved_mb"]
    )
    return {
        "stages": stages,
        "transitions": transitions,
        "peak_stage_id": None if peak_stage is None else peak_stage["stage_id"],
        "peak_stage_memory_total_mb": (
            0.0 if peak_stage is None else peak_stage["memory_total_mb"]
        ),
        "peak_transition_memory_mb": (
            0.0 if peak_transition is None else peak_transition["memory_mb"]
        ),
        "peak_combined_memory_total_mb": peak_combined,
        "peak_combined_peak_mb": max(peak_combined - peak_combined_reserved_mb, 0.0),
        "peak_combined_reserved_mb": peak_combined_reserved_mb,
        "peak_combined_stage_id": (
            None if peak_combined_stage is None else peak_combined_stage["stage_id"]
        ),
    }


def _build_stage_memory(stage, config):
    segments = [_build_segment_memory(segment, config) for segment in stage.segments]
    stage_breakdown = _aggregate_stage_memory_breakdown(segments)
    memory_total_mb = stage_breakdown["peak_mb"] + stage_breakdown["reserved_mb"]
    return {
        "stage_id": stage.stage_id,
        "device_group": list(stage.device_group),
        "memory_total_mb": memory_total_mb,
        "memory_breakdown": stage_breakdown,
        "segments": segments,
    }


def _build_segment_memory(segment, config):
    layer_count = segment.end - segment.start + 1
    strategy = _segment_strategy(segment.strategy, layer_count)
    cost = _estimate_strategy_memory_cost(
        strategy,
        _config_with_layer_span(config, segment.start, layer_count),
    )
    return {
        "start": segment.start,
        "end": segment.end,
        "layer_count": layer_count,
        "memory_total_mb": cost["memory_total_mb"],
        "memory_breakdown": cost["memory_breakdown"],
    }


def _aggregate_stage_memory_breakdown(segments):
    breakdown = deepcopy(DEFAULT_MEMORY_BREAKDOWN)
    if not segments:
        return breakdown
    breakdown["parameters_mb"] = sum(
        segment["memory_breakdown"]["parameters_mb"] for segment in segments
    )
    breakdown["model_states_mb"] = sum(
        segment["memory_breakdown"]["model_states_mb"] for segment in segments
    )
    breakdown["activations_mb"] = max(
        segment["memory_breakdown"]["activations_mb"] for segment in segments
    )
    breakdown["recompute_saved_mb"] = sum(
        segment["memory_breakdown"]["recompute_saved_mb"] for segment in segments
    )
    breakdown["reserved_mb"] = max(
        segment["memory_breakdown"]["reserved_mb"] for segment in segments
    )
    breakdown["peak_mb"] = breakdown["model_states_mb"] + breakdown["activations_mb"]
    return breakdown


def _build_transition_memory(plan, config):
    return [
        _estimate_transition_memory(spec, config) for spec in _plan_transition_specs(plan)
    ]


def _estimate_transition_memory(spec, config):
    source_mb = _activation_buffer_mb(config, spec["source_strategy"])
    target_mb = _activation_buffer_mb(config, spec["target_strategy"])
    factor = _transition_factor(spec)
    return {
        "source_stage_id": spec["source_stage_id"],
        "target_stage_id": spec["target_stage_id"],
        "source_segment_index": spec["source_segment_index"],
        "target_segment_index": spec["target_segment_index"],
        "kind": spec["kind"],
        "metadata": to_json_safe(dict(spec["metadata"])),
        "memory_mb": max(source_mb, target_mb) * factor,
    }


def _activation_buffer_mb(config, strategy):
    hidden_size = float(config.train.model.hidden_size)
    seq_length = float(config.train.model.seq_length)
    micro_batch_size = float(strategy["micro_batch_size"])
    return (micro_batch_size * seq_length * hidden_size * 2.0) / NUM_BYTES_IN_MEGABYTE


def _transition_factor(spec):
    metadata = spec["metadata"]
    explicit_factor = metadata.get("reshard_factor")
    if explicit_factor is not None:
        return float(explicit_factor)
    factor = 1.0
    for key in (
        "tensor_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
    ):
        if spec["source_strategy"].get(key) != spec["target_strategy"].get(key):
            factor += 0.5
    return factor


def _plan_transition_specs(plan):
    specs = _default_transition_specs(plan)
    for transition in plan.transitions:
        _merge_transition_spec(specs, _transition_spec_from_plan(plan, transition))
    return specs


def _transition_spec_from_plan(plan, transition):
    source_stage = plan.stages[transition.source_stage_id]
    target_stage = plan.stages[transition.target_stage_id]
    source_index = _resolve_transition_index(source_stage, transition.source_segment_index, -1)
    target_index = _resolve_transition_index(target_stage, transition.target_segment_index, 0)
    source_segment = source_stage.segments[source_index]
    target_segment = target_stage.segments[target_index]
    return {
        "source_stage_id": transition.source_stage_id,
        "target_stage_id": transition.target_stage_id,
        "source_segment_index": source_index,
        "target_segment_index": target_index,
        "kind": transition.kind,
        "metadata": dict(transition.metadata),
        "source_strategy": dict(source_segment.strategy),
        "target_strategy": dict(target_segment.strategy),
    }


def _default_transition_specs(plan):
    specs = []
    for stage in plan.stages:
        for source_index, (source_segment, target_segment) in enumerate(
            zip(stage.segments, stage.segments[1:])
        ):
            specs.append(
                {
                    "source_stage_id": stage.stage_id,
                    "target_stage_id": stage.stage_id,
                    "source_segment_index": source_index,
                    "target_segment_index": source_index + 1,
                    "kind": "intra-stage",
                    "metadata": {},
                    "source_strategy": dict(source_segment.strategy),
                    "target_strategy": dict(target_segment.strategy),
                }
            )
    for source_stage, target_stage in zip(plan.stages, plan.stages[1:]):
        specs.append(
            {
                "source_stage_id": source_stage.stage_id,
                "target_stage_id": target_stage.stage_id,
                "source_segment_index": len(source_stage.segments) - 1,
                "target_segment_index": 0,
                "kind": "pipeline",
                "metadata": {},
                "source_strategy": dict(source_stage.segments[-1].strategy),
                "target_strategy": dict(target_stage.segments[0].strategy),
            }
        )
    return specs


def _merge_transition_spec(specs, explicit_spec):
    matched = False
    for index, spec in enumerate(specs):
        if not _transition_matches(spec, explicit_spec):
            continue
        specs[index] = {
            **spec,
            "kind": explicit_spec["kind"],
            "metadata": {
                **dict(spec["metadata"]),
                **dict(explicit_spec["metadata"]),
            },
        }
        matched = True
    if not matched:
        specs.append(explicit_spec)


def _transition_matches(spec, explicit_spec):
    if (
        spec["source_stage_id"] != explicit_spec["source_stage_id"]
        or spec["target_stage_id"] != explicit_spec["target_stage_id"]
    ):
        return False
    source_segment_index = explicit_spec["source_segment_index"]
    target_segment_index = explicit_spec["target_segment_index"]
    if source_segment_index is None and target_segment_index is None:
        return True
    return (
        spec["source_segment_index"] == source_segment_index
        and spec["target_segment_index"] == target_segment_index
    )


def _resolve_transition_index(stage, index, default_index):
    if index is None:
        return len(stage.segments) + default_index if default_index < 0 else default_index
    return index


def _attached_transition_peak_mb(stage_id, transitions):
    return max(
        (
            transition["memory_mb"]
            for transition in transitions
            if transition["source_stage_id"] == stage_id
            or transition["target_stage_id"] == stage_id
        ),
        default=0.0,
    )


def _peak_combined_stage(stages, transitions):
    peak_stage = None
    peak_transition_mb = 0.0
    peak_total_mb = 0.0
    for stage in stages:
        transition_mb = _attached_transition_peak_mb(stage["stage_id"], transitions)
        total_mb = stage["memory_total_mb"] + transition_mb
        if peak_stage is None or total_mb > peak_total_mb:
            peak_stage = stage
            peak_transition_mb = transition_mb
            peak_total_mb = total_mb
    return peak_stage, peak_transition_mb


def _peak_combined_stage_breakdown(plan_breakdown):
    peak_stage_id = plan_breakdown["peak_combined_stage_id"]
    if peak_stage_id is None:
        return deepcopy(DEFAULT_MEMORY_BREAKDOWN)
    for stage in plan_breakdown["stages"]:
        if stage["stage_id"] == peak_stage_id:
            return deepcopy(stage["memory_breakdown"])
    return deepcopy(DEFAULT_MEMORY_BREAKDOWN)


def _config_with_layer_span(config, start, num_layers):
    segment_config = deepcopy(config)
    segment_config.train.model.num_layers = num_layers
    moe_layer_freq = config.train.model.get("moe_layer_freq", None)
    if moe_layer_freq is not None:
        sliced_freq = _slice_moe_layer_freq(
            moe_layer_freq,
            total_layers=int(config.train.model.num_layers),
            start=start,
            num_layers=num_layers,
        )
        segment_config.train.model.moe_layer_freq = repr(sliced_freq)
    return segment_config


def _slice_moe_layer_freq(value, *, total_layers, start, num_layers):
    normalized = normalize_moe_layer_freq(value, num_layers=total_layers)
    if isinstance(normalized, int):
        normalized = [
            1 if index % normalized == 0 else 0 for index in range(total_layers)
        ]
    if not isinstance(normalized, list):
        return normalized
    end = start + num_layers
    if start < 0 or end > len(normalized):
        raise ValueError("moe_layer_freq segment span exceeds model layer count")
    return normalized[start:end]


def _segment_strategy(strategy, num_layers):
    segment_strategy = dict(strategy)
    segment_strategy["pipeline_model_parallel_size"] = 1
    segment_strategy["num_layers_per_virtual_pipeline_stage"] = None
    segment_strategy["decoder_first_pipeline_num_layers"] = None
    segment_strategy["decoder_last_pipeline_num_layers"] = None
    if segment_strategy.get("recompute_num_layers") is not None:
        segment_strategy["recompute_num_layers"] = min(
            segment_strategy["recompute_num_layers"],
            num_layers,
        )
    return segment_strategy


def _get_num_microbatches(strategy, config):
    acc_step = strategy.get("acc_step")
    if isinstance(acc_step, int) and acc_step > 0:
        return acc_step
    global_batch_size = config.train.model.global_batch_size
    data_parallel_size = strategy["data_parallel_size"]
    micro_batch_size = strategy["micro_batch_size"]
    return global_batch_size // data_parallel_size // micro_batch_size


def _build_memory_args(config, strategy):
    args = SimpleNamespace()
    flagscale_args = config.train.model
    _populate_model_args(args, config, flagscale_args)
    _populate_parallelism_args(args, config, strategy, flagscale_args)
    return args


def _normalize_moe_layer_freq(value, num_layers):
    return normalize_moe_layer_freq(value, num_layers=num_layers)


def _populate_model_args(args, config, flagscale_args):
    args.hidden_size = flagscale_args.hidden_size
    args.num_attention_heads = flagscale_args.num_attention_heads
    args.num_layers = flagscale_args.num_layers
    args.multi_latent_attention = flagscale_args.get("multi_latent_attention", False)
    args.qk_head_dim = flagscale_args.get("qk_head_dim", None)
    args.v_head_dim = flagscale_args.get("v_head_dim", None)
    args.kv_lora_rank = flagscale_args.get("kv_lora_rank", None)
    args.q_lora_rank = flagscale_args.get("q_lora_rank", None)
    args.qk_pos_emb_head_dim = flagscale_args.get("qk_pos_emb_head_dim", None)
    args.qk_layernorm = flagscale_args.get("qk_layernorm", False)
    args.qk_layernorm_hidden_dim = flagscale_args.get("qk_layernorm_hidden_dim", False)
    args.kv_channels = flagscale_args.get("kv_channels", args.hidden_size // args.num_attention_heads)
    args.group_query_attention = flagscale_args.get("group_query_attention", False)
    args.num_query_groups = flagscale_args.get("num_query_groups", args.num_attention_heads)
    args.num_experts = flagscale_args.get("num_experts", None)
    args.moe_ffn_hidden_size = flagscale_args.get("moe_ffn_hidden_size", None)
    args.moe_shared_expert_intermediate_size = flagscale_args.get(
        "moe_shared_expert_intermediate_size", None
    )
    args.moe_layer_freq = _normalize_moe_layer_freq(
        flagscale_args.get("moe_layer_freq", 1),
        args.num_layers,
    )
    args.moe_router_topk = flagscale_args.get("moe_router_topk", None)
    args.mtp_num_layers = flagscale_args.get("mtp_num_layers", None)
    args.swiglu = flagscale_args.get("swiglu", False)
    args.multiple_of = flagscale_args.get("multiple_of", None)
    args.hidden_dim_multiplier = flagscale_args.get("hidden_dim_multiplier", None)
    args.ffn_hidden_size = flagscale_args.get("ffn_hidden_size", _default_ffn_hidden_size(args))
    args.make_vocab_size_divisible_by = flagscale_args.get("make_vocab_size_divisible_by", 128)
    args.padded_vocab_size = _resolve_padded_vocab_size(config, flagscale_args, args)
    args.untie_embeddings_and_output_weights = flagscale_args.get(
        "untie_embeddings_and_output_weights", None
    )


def _populate_parallelism_args(args, config, strategy, flagscale_args):
    args.tensor_model_parallel_size = strategy["tensor_model_parallel_size"]
    args.pipeline_model_parallel_size = strategy["pipeline_model_parallel_size"]
    args.data_parallel_size = strategy["data_parallel_size"]
    args.expert_model_parallel_size = strategy["expert_model_parallel_size"]
    args.use_distributed_optimizer = strategy["use_distributed_optimizer"]
    args.seq_length = flagscale_args.seq_length
    args.micro_batch_size = strategy["micro_batch_size"]
    args.virtual_pipeline_model_parallel_size = _virtual_pipeline_size(strategy, flagscale_args)
    args.sequence_parallel = strategy["sequence_parallel"]
    args.recompute_granularity = strategy["recompute_granularity"]
    args.recompute_method = strategy["recompute_method"]
    args.recompute_num_layers = strategy["recompute_num_layers"]
    args.context_parallel_size = strategy["context_parallel_size"]
    args.expert_tensor_parallel_size = config.train.system.get(
        "expert_tensor_parallel_size",
        args.tensor_model_parallel_size,
    )
    args.world_size = (
        args.tensor_model_parallel_size
        * args.context_parallel_size
        * args.data_parallel_size
        * args.pipeline_model_parallel_size
    )


def _default_ffn_hidden_size(args):
    if not args.swiglu:
        return 4 * args.hidden_size
    hidden_dim = int(4 * args.hidden_size * 2 / 3)
    if args.hidden_dim_multiplier is not None:
        hidden_dim = int(hidden_dim * args.hidden_dim_multiplier)
    multiple_of = args.multiple_of or 64
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


def _resolve_padded_vocab_size(config, flagscale_args, args):
    if "padded_vocab_size" in flagscale_args:
        return flagscale_args["padded_vocab_size"]
    vocab_size = flagscale_args.get("vocab_size", None)
    if vocab_size is None and "data" in config.train and "tokenizer" in config.train.data:
        vocab_size = config.train.data.tokenizer.get("vocab_size", None)
    if vocab_size is None:
        raise ValueError(
            "Memory cost model requires train.model.padded_vocab_size, train.model.vocab_size, "
            "or train.data.tokenizer.vocab_size."
        )
    divisible_by = args.make_vocab_size_divisible_by
    return divisible_by * ((vocab_size + divisible_by - 1) // divisible_by)


def _virtual_pipeline_size(strategy, flagscale_args):
    num_layers_per_virtual_pipeline_stage = strategy["num_layers_per_virtual_pipeline_stage"]
    if num_layers_per_virtual_pipeline_stage is None:
        return None
    return (
        flagscale_args.num_layers
        // strategy["pipeline_model_parallel_size"]
        // num_layers_per_virtual_pipeline_stage
    )


def _activation_memory_mb(args, num_microbatches):
    activation_bytes = compute_activation_memory(args, num_microbatches=num_microbatches)
    return activation_bytes / NUM_BYTES_IN_MEGABYTE


def _activation_without_recompute_mb(args, num_microbatches):
    args_without_recompute = deepcopy(args)
    args_without_recompute.recompute_method = None
    args_without_recompute.recompute_granularity = None
    args_without_recompute.recompute_num_layers = None
    return _activation_memory_mb(args_without_recompute, num_microbatches)


def _read_biases(config):
    profile = get_chip_profile_or_none(config)
    if profile is None:
        return 0.0, 0.0
    cost_model = profile["cost_model"]
    return (
        float(cost_model.get("reserved_memory_bias_mb", 0.0)),
        float(cost_model.get("peak_activation_bias_mb", 0.0)),
    )


def _estimate_parameter_memory_mb(model_states_mb, strategy):
    return model_states_mb / _model_state_multiplier(strategy)


def _model_state_multiplier(strategy):
    if not strategy.get("use_distributed_optimizer", False):
        return FULL_MODEL_STATE_MULTIPLIER
    data_parallel_size = strategy["data_parallel_size"]
    return SHARDED_MODEL_STATE_BASE + (SHARDED_MODEL_STATE_GRAD_FACTOR / data_parallel_size)
