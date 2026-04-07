from copy import deepcopy
from types import SimpleNamespace

from flagscale.runner.auto_tuner.chip_profile import get_chip_profile_or_none
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
}
FULL_MODEL_STATE_MULTIPLIER = 18.0
SHARDED_MODEL_STATE_BASE = 6.0
SHARDED_MODEL_STATE_GRAD_FACTOR = 12.0


def estimate_memory_cost(strategy, config):
    args = _build_memory_args(config, strategy)
    num_microbatches = _get_num_microbatches(strategy, config)
    base_total_mb = float(report_theoretical_memory(args, num_microbatches=num_microbatches))
    actual_activation_mb = _activation_memory_mb(args, num_microbatches)
    disabled_recompute_activation_mb = _activation_without_recompute_mb(args, num_microbatches)
    model_states_mb = max(base_total_mb - actual_activation_mb, 0.0)
    recompute_saved_mb = max(disabled_recompute_activation_mb - actual_activation_mb, 0.0)
    reserved_mb, peak_activation_bias_mb = _read_biases(config)
    peak_mb = base_total_mb + peak_activation_bias_mb

    breakdown = deepcopy(DEFAULT_MEMORY_BREAKDOWN)
    breakdown["model_states_mb"] = model_states_mb
    breakdown["parameters_mb"] = _estimate_parameter_memory_mb(model_states_mb, strategy)
    breakdown["activations_mb"] = max(peak_mb - model_states_mb, 0.0)
    breakdown["recompute_saved_mb"] = recompute_saved_mb
    breakdown["peak_mb"] = peak_mb
    breakdown["reserved_mb"] = reserved_mb
    return {
        "memory_total_mb": peak_mb + reserved_mb,
        "memory_breakdown": breakdown,
    }


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
