from copy import deepcopy
from math import log2

from flagscale.runner.auto_tuner.cost.profile_store import build_cost_profile

BF16_BYTES = 2.0
BITS_PER_BYTE = 8.0
MILLISECONDS_PER_SECOND = 1000.0
GIGA = 1_000_000_000.0
TERA = 1_000_000_000_000.0
COMPUTE_FLOPS_FACTOR = 24.0
TP_COMM_CALLS_PER_LAYER = 2
PP_TRANSFERS_PER_MICROBATCH = 2
RECOMPUTE_FULL_FACTOR = 0.3
RECOMPUTE_SELECTIVE_FACTOR = 0.15
BLOCK_RECOMPUTE_FACTOR = 1.1
UNIFORM_RECOMPUTE_FACTOR = 1.0
DEFAULT_FABRIC_PENALTY = 1.0
FABRIC_PENALTIES = {
    "pcie": 1.35,
    "ethernet": 1.5,
    "roce": 1.15,
    "infiniband": 1.05,
    "nvlink": 0.85,
}
DEFAULT_TIME_BREAKDOWN = {
    "compute_ms": 0.0,
    "tp_comm_ms": 0.0,
    "dp_comm_ms": 0.0,
    "pp_comm_ms": 0.0,
    "recompute_ms": 0.0,
}


def estimate_time_cost(strategy, config):
    profile = build_cost_profile(config, strategy)
    breakdown = deepcopy(DEFAULT_TIME_BREAKDOWN)
    compute_ms = _estimate_compute_ms(profile)
    breakdown["compute_ms"] = compute_ms
    breakdown["tp_comm_ms"] = _apply_overlap(
        _estimate_tp_comm_ms(profile),
        profile,
        "tp_comm_overlap_ratio",
    )
    breakdown["dp_comm_ms"] = _apply_overlap(
        _estimate_dp_comm_ms(profile),
        profile,
        "dp_comm_overlap_ratio",
    )
    breakdown["pp_comm_ms"] = _apply_overlap(
        _estimate_pp_comm_ms(profile),
        profile,
        "pp_comm_overlap_ratio",
    )
    breakdown["recompute_ms"] = _estimate_recompute_ms(profile, compute_ms)
    return {
        "time_total_ms": sum(breakdown.values()),
        "time_breakdown": breakdown,
    }


def _estimate_compute_ms(profile):
    strategy = profile["runtime"]["strategy"]
    model = profile["model"]
    compute = profile["hardware"]["compute"]
    tokens = _tokens_per_iteration(profile)
    stage_layers = _stage_layers(model["num_layers"], strategy)
    hidden_size = float(model["hidden_size"])
    parallel_factor = (
        strategy["tensor_model_parallel_size"]
        * strategy["context_parallel_size"]
        * strategy["expert_model_parallel_size"]
    )
    total_flops = tokens * stage_layers * hidden_size * hidden_size * COMPUTE_FLOPS_FACTOR
    effective_tflops = min(float(compute["bf16_tflops"]), float(compute["attention_tflops"]))
    return _flops_to_ms(total_flops / parallel_factor, effective_tflops)


def _estimate_tp_comm_ms(profile):
    strategy = profile["runtime"]["strategy"]
    tp_size = strategy["tensor_model_parallel_size"]
    if tp_size <= 1:
        return 0.0
    intra_node = profile["hardware"]["interconnect"]["intra_node"]
    stage_layers = _stage_layers(profile["model"]["num_layers"], strategy)
    volume_bytes = _activation_bytes(profile)
    if strategy["sequence_parallel"]:
        volume_bytes *= 0.75
    repetitions = strategy["acc_step"] * stage_layers * TP_COMM_CALLS_PER_LAYER
    return _collective_ms(
        volume_bytes,
        intra_node["all_reduce_bandwidth_gbps"],
        intra_node["all_reduce_latency_us"],
        tp_size,
        repetitions,
    )


def _estimate_dp_comm_ms(profile):
    strategy = profile["runtime"]["strategy"]
    dp_size = strategy["data_parallel_size"]
    if dp_size <= 1:
        return 0.0
    intra_node = profile["hardware"]["interconnect"]["intra_node"]
    volume_bytes = _parameter_bytes(profile)
    if strategy["use_distributed_optimizer"]:
        volume_bytes *= 0.5
    return _collective_ms(
        volume_bytes,
        intra_node["all_reduce_bandwidth_gbps"],
        intra_node["all_reduce_latency_us"],
        dp_size,
        1,
    )


def _estimate_pp_comm_ms(profile):
    strategy = profile["runtime"]["strategy"]
    pp_size = strategy["pipeline_model_parallel_size"]
    if pp_size <= 1:
        return 0.0
    intra_node = profile["hardware"]["interconnect"]["intra_node"]
    transfers = strategy["acc_step"] * (pp_size - 1) * PP_TRANSFERS_PER_MICROBATCH
    base_ms = _transfer_ms(
        _activation_bytes(profile),
        intra_node["p2p_bandwidth_gbps"],
        intra_node["p2p_latency_us"],
        transfers,
    )
    fabric = str(intra_node["fabric"]).lower()
    return base_ms * FABRIC_PENALTIES.get(fabric, DEFAULT_FABRIC_PENALTY)


def _estimate_recompute_ms(profile, compute_ms):
    strategy = profile["runtime"]["strategy"]
    if not strategy["use_recompute"]:
        return 0.0
    stage_layers = _stage_layers(profile["model"]["num_layers"], strategy)
    recompute_layers = strategy["recompute_num_layers"] or stage_layers
    layer_ratio = min(recompute_layers, stage_layers) / stage_layers
    granularity_factor = _recompute_granularity_factor(strategy["recompute_granularity"])
    method_factor = _recompute_method_factor(strategy["recompute_method"])
    return compute_ms * layer_ratio * granularity_factor * method_factor


def _tokens_per_iteration(profile):
    strategy = profile["runtime"]["strategy"]
    model = profile["model"]
    return strategy["micro_batch_size"] * model["seq_length"] * strategy["acc_step"]


def _stage_layers(num_layers, strategy):
    pp_size = strategy["pipeline_model_parallel_size"]
    if pp_size <= 1:
        return num_layers
    first_layers = strategy["decoder_first_pipeline_num_layers"]
    last_layers = strategy["decoder_last_pipeline_num_layers"]
    if first_layers is None and last_layers is None:
        return num_layers / pp_size
    if pp_size == 2:
        if first_layers is None:
            return num_layers - last_layers
        return max(first_layers, num_layers - first_layers)
    remaining_layers = num_layers - (first_layers or 0) - (last_layers or 0)
    middle_stages = pp_size - 2
    middle_layers = remaining_layers / middle_stages if middle_stages > 0 else 0.0
    return max(first_layers or 0.0, last_layers or 0.0, middle_layers)


def _activation_bytes(profile):
    strategy = profile["runtime"]["strategy"]
    model = profile["model"]
    return (
        strategy["micro_batch_size"]
        * model["seq_length"]
        * model["hidden_size"]
        * BF16_BYTES
    )


def _parameter_bytes(profile):
    strategy = profile["runtime"]["strategy"]
    model = profile["model"]
    hidden_size = float(model["hidden_size"])
    num_layers = profile["model"]["num_layers"]
    vocab_size = float(model.get("padded_vocab_size", hidden_size * 16))
    per_layer_params = 12.0 * hidden_size * hidden_size
    embedding_params = vocab_size * hidden_size
    total_params = (per_layer_params * num_layers) + embedding_params
    partition_factor = (
        strategy["tensor_model_parallel_size"]
        * strategy["pipeline_model_parallel_size"]
        * strategy["expert_model_parallel_size"]
    )
    return (total_params * BF16_BYTES) / partition_factor


def _apply_overlap(raw_ms, profile, field):
    overlap = profile["hardware"]["cost_model"]["overlap"][field]
    return raw_ms * (1.0 - overlap)


def _collective_ms(volume_bytes, bandwidth_gbps, latency_us, group_size, repetitions):
    if group_size <= 1:
        return 0.0
    collective_factor = 2.0 * (group_size - 1) / group_size
    return _transfer_ms(
        volume_bytes * collective_factor,
        bandwidth_gbps,
        latency_us,
        repetitions * max(log2(group_size), 1.0),
    )


def _transfer_ms(volume_bytes, bandwidth_gbps, latency_us, repetitions):
    if repetitions <= 0:
        return 0.0
    bandwidth_bytes_per_second = (bandwidth_gbps * GIGA) / BITS_PER_BYTE
    transfer_ms = (volume_bytes * repetitions / bandwidth_bytes_per_second) * MILLISECONDS_PER_SECOND
    latency_ms = repetitions * (latency_us / MILLISECONDS_PER_SECOND)
    return transfer_ms + latency_ms


def _flops_to_ms(total_flops, tflops):
    return (total_flops / (tflops * TERA)) * MILLISECONDS_PER_SECOND


def _recompute_granularity_factor(granularity):
    if granularity == "selective":
        return RECOMPUTE_SELECTIVE_FACTOR
    return RECOMPUTE_FULL_FACTOR


def _recompute_method_factor(method):
    if method == "block":
        return BLOCK_RECOMPUTE_FACTOR
    return UNIFORM_RECOMPUTE_FACTOR
