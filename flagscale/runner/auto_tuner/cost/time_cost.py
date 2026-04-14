from copy import deepcopy
from math import log2

from flagscale.runner.auto_tuner.cost.profile_store import build_cost_profile
from flagscale.runner.auto_tuner.plan.lowering import lower_strategy_to_plan
from flagscale.runner.auto_tuner.plan.schema import ModelPlan
from flagscale.runner.auto_tuner.plan.summary import extract_homogeneous_strategy, to_json_safe

BF16_BYTES = 2.0
BITS_PER_BYTE = 8.0
MILLISECONDS_PER_SECOND = 1000.0
GIGA = 1_000_000_000.0
TERA = 1_000_000_000_000.0
COMPUTE_FLOPS_FACTOR = 24.0
MOE_EXPERT_FLOPS_FACTOR = 4.0
TP_COMM_CALLS_PER_LAYER = 2
PP_TRANSFERS_PER_MICROBATCH = 2
MOE_ALLTOALL_EXCHANGES = 2.0
MOE_ALLGATHER_EXCHANGES = 4.0
DEFAULT_MOE_ROUTER_TOPK = 1.0
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
    "expert_comm_ms": 0.0,
    "recompute_ms": 0.0,
}


def estimate_time_cost(strategy, config):
    plan = (
        strategy
        if isinstance(strategy, ModelPlan)
        else lower_strategy_to_plan(strategy, config, validate=False)
    )
    return _estimate_plan_time_cost(plan, config)


def _estimate_strategy_time_cost(strategy, config):
    profile = build_cost_profile(config, strategy)
    breakdown = deepcopy(DEFAULT_TIME_BREAKDOWN)
    compute_ms = _estimate_compute_ms(profile)
    breakdown["compute_ms"] = compute_ms
    breakdown["tp_comm_ms"] = _apply_overlap(
        _estimate_tp_comm_ms(profile), profile, "tp_comm_overlap_ratio"
    )
    breakdown["dp_comm_ms"] = _apply_overlap(
        _estimate_dp_comm_ms(profile), profile, "dp_comm_overlap_ratio"
    )
    breakdown["pp_comm_ms"] = _apply_overlap(
        _estimate_pp_comm_ms(profile), profile, "pp_comm_overlap_ratio"
    )
    breakdown["expert_comm_ms"] = _estimate_expert_comm_ms(profile)
    breakdown["recompute_ms"] = _estimate_recompute_ms(profile, compute_ms)
    return {"time_total_ms": sum(breakdown.values()), "time_breakdown": breakdown}


def _estimate_plan_time_cost(plan, config):
    strategy = extract_homogeneous_strategy(plan)
    if strategy is not None:
        result = _estimate_strategy_time_cost(strategy, config)
        result["time_breakdown"]["plan"] = _build_plan_time_breakdown(plan, config)
        return result
    plan_breakdown = _build_plan_time_breakdown(plan, config)
    inter_stage_transitions = [
        transition
        for transition in plan_breakdown["transitions"]
        if transition["source_stage_id"] != transition["target_stage_id"]
    ]
    breakdown = deepcopy(DEFAULT_TIME_BREAKDOWN)
    peak_stage = max(
        plan_breakdown["stages"],
        key=lambda stage: stage["time_total_ms"],
        default=None,
    )
    if peak_stage is not None:
        for key in DEFAULT_TIME_BREAKDOWN:
            breakdown[key] = peak_stage["time_breakdown"][key]
    breakdown["pp_comm_ms"] += sum(
        item["time_ms"] for item in inter_stage_transitions
    )
    time_total_ms = sum(breakdown.values())
    breakdown["plan"] = plan_breakdown
    return {"time_total_ms": time_total_ms, "time_breakdown": breakdown}


def _build_plan_time_breakdown(plan, config):
    transitions = _build_transition_breakdown(plan, config)
    stages = [_build_stage_time(stage, config, transitions) for stage in plan.stages]
    return {"stages": stages, "transitions": transitions}


def _build_stage_time(stage, config, transitions):
    segments = [_build_segment_time(segment, config) for segment in stage.segments]
    breakdown = deepcopy(DEFAULT_TIME_BREAKDOWN)
    for segment in segments:
        _accumulate_breakdown(breakdown, segment["time_breakdown"])
    local_transition_ms = sum(
        transition["time_ms"]
        for transition in transitions
        if transition["source_stage_id"] == stage.stage_id
        and transition["target_stage_id"] == stage.stage_id
    )
    breakdown["pp_comm_ms"] += local_transition_ms
    return {
        "stage_id": stage.stage_id,
        "device_group": list(stage.device_group),
        "time_total_ms": sum(segment["time_total_ms"] for segment in segments)
        + local_transition_ms,
        "time_breakdown": breakdown,
        "segments": segments,
    }


def _build_segment_time(segment, config):
    layer_count = segment.end - segment.start + 1
    cost = _estimate_strategy_time_cost(
        _segment_strategy(segment.strategy, layer_count),
        _config_with_num_layers(config, layer_count),
    )
    return {
        "start": segment.start,
        "end": segment.end,
        "layer_count": layer_count,
        "time_total_ms": cost["time_total_ms"],
        "time_breakdown": cost["time_breakdown"],
    }


def _build_transition_breakdown(plan, config):
    return [_estimate_transition_time(spec, config) for spec in _plan_transition_specs(plan)]


def _accumulate_breakdown(target, source):
    for key in DEFAULT_TIME_BREAKDOWN:
        target[key] += source[key]


def _estimate_transition_time(spec, config):
    source_strategy = _transition_strategy(spec["source_strategy"], spec["source_layer_count"])
    target_strategy = _transition_strategy(spec["target_strategy"], spec["target_layer_count"])
    source_profile = build_cost_profile(
        _transition_config(config, source_strategy, spec["source_layer_count"]),
        source_strategy,
    )
    target_profile = build_cost_profile(
        _transition_config(config, target_strategy, spec["target_layer_count"]),
        target_strategy,
    )
    source_bytes = _activation_bytes(source_profile)
    target_bytes = _activation_bytes(target_profile)
    bandwidth_gbps, latency_us, fabric = _communication_link(source_profile, "pp", "p2p")
    transition_ms = _transfer_ms(
        max(source_bytes, target_bytes) * _transition_factor(spec),
        bandwidth_gbps,
        latency_us,
        _transition_repetitions(spec),
    )
    transition_ms *= FABRIC_PENALTIES.get(str(fabric).lower(), DEFAULT_FABRIC_PENALTY)
    return {
        "source_stage_id": spec["source_stage_id"],
        "target_stage_id": spec["target_stage_id"],
        "source_segment_index": spec["source_segment_index"],
        "target_segment_index": spec["target_segment_index"],
        "kind": spec["kind"],
        "metadata": to_json_safe(dict(spec["metadata"])),
        "time_ms": transition_ms,
    }


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


def _transition_repetitions(spec):
    repetitions = max(
        int(spec["source_strategy"]["acc_step"]),
        int(spec["target_strategy"]["acc_step"]),
    )
    if spec["source_stage_id"] != spec["target_stage_id"]:
        return repetitions * PP_TRANSFERS_PER_MICROBATCH
    return repetitions


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
        "source_layer_count": source_segment.end - source_segment.start + 1,
        "target_layer_count": target_segment.end - target_segment.start + 1,
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
                    "source_layer_count": source_segment.end - source_segment.start + 1,
                    "target_layer_count": target_segment.end - target_segment.start + 1,
                }
            )
    for source_stage, target_stage in zip(plan.stages, plan.stages[1:]):
        source_segment = source_stage.segments[-1]
        target_segment = target_stage.segments[0]
        specs.append(
            {
                "source_stage_id": source_stage.stage_id,
                "target_stage_id": target_stage.stage_id,
                "source_segment_index": len(source_stage.segments) - 1,
                "target_segment_index": 0,
                "kind": "pipeline",
                "metadata": {},
                "source_strategy": dict(source_segment.strategy),
                "target_strategy": dict(target_segment.strategy),
                "source_layer_count": source_segment.end - source_segment.start + 1,
                "target_layer_count": target_segment.end - target_segment.start + 1,
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


def _transition_config(config, strategy, layer_count):
    transition_config = _config_with_num_layers(config, layer_count)
    transition_config.train.model.num_layers = max(
        layer_count,
        int(strategy["pipeline_model_parallel_size"]),
    )
    return transition_config


def _transition_strategy(strategy, num_layers):
    transition_strategy = _segment_strategy(strategy, num_layers)
    transition_strategy["pipeline_model_parallel_size"] = strategy["pipeline_model_parallel_size"]
    return transition_strategy


def _config_with_num_layers(config, num_layers):
    segment_config = deepcopy(config)
    segment_config.train.model.num_layers = num_layers
    return segment_config


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


def _estimate_compute_ms(profile):
    return _estimate_dense_compute_ms(profile) + _estimate_expert_compute_ms(profile)


def _estimate_dense_compute_ms(profile):
    strategy = profile["runtime"]["strategy"]
    model = profile["model"]
    total_flops = (
        _tokens_per_iteration(profile)
        * _stage_layers(model["num_layers"], strategy)
        * float(model["hidden_size"])
        * float(model["hidden_size"])
        * COMPUTE_FLOPS_FACTOR
    )
    return _flops_to_ms(
        total_flops / _dense_parallel_factor(strategy), _effective_tflops(profile)
    )


def _estimate_expert_compute_ms(profile):
    model = profile["model"]
    if not _is_moe_model(model):
        return 0.0
    strategy = profile["runtime"]["strategy"]
    hidden_size = float(model["hidden_size"])
    moe_hidden_size = float(model.get("moe_ffn_hidden_size", hidden_size * 4.0))
    total_flops = (
        _tokens_per_iteration(profile)
        * _stage_moe_layers(profile)
        * hidden_size
        * moe_hidden_size
        * float(model.get("moe_router_topk", DEFAULT_MOE_ROUTER_TOPK))
        * MOE_EXPERT_FLOPS_FACTOR
    )
    parallel_factor = _dense_parallel_factor(strategy) * _effective_expert_parallel_size(
        strategy, model
    )
    return _flops_to_ms(total_flops / parallel_factor, _effective_tflops(profile))


def _estimate_tp_comm_ms(profile):
    strategy = profile["runtime"]["strategy"]
    tp_size = strategy["tensor_model_parallel_size"]
    if tp_size <= 1:
        return 0.0
    volume_bytes = _activation_bytes(profile)
    if strategy["sequence_parallel"]:
        volume_bytes *= 0.75
    bandwidth_gbps, latency_us, _ = _communication_link(profile, "tp", "collective")
    repetitions = (
        strategy["acc_step"]
        * _stage_layers(profile["model"]["num_layers"], strategy)
        * TP_COMM_CALLS_PER_LAYER
    )
    return _collective_ms(volume_bytes, bandwidth_gbps, latency_us, tp_size, repetitions)


def _estimate_dp_comm_ms(profile):
    strategy = profile["runtime"]["strategy"]
    dp_size = strategy["data_parallel_size"]
    if dp_size <= 1:
        return 0.0
    volume_bytes = _parameter_bytes(profile)
    if strategy["use_distributed_optimizer"]:
        volume_bytes *= 0.5
    bandwidth_gbps, latency_us, _ = _communication_link(profile, "dp", "collective")
    return _collective_ms(volume_bytes, bandwidth_gbps, latency_us, dp_size, 1)


def _estimate_pp_comm_ms(profile):
    strategy = profile["runtime"]["strategy"]
    pp_size = strategy["pipeline_model_parallel_size"]
    if pp_size <= 1:
        return 0.0
    bandwidth_gbps, latency_us, fabric = _communication_link(profile, "pp", "p2p")
    transfers = strategy["acc_step"] * (pp_size - 1) * PP_TRANSFERS_PER_MICROBATCH
    base_ms = _transfer_ms(
        _activation_bytes(profile), bandwidth_gbps, latency_us, transfers
    )
    return base_ms * FABRIC_PENALTIES.get(str(fabric).lower(), DEFAULT_FABRIC_PENALTY)


def _estimate_expert_comm_ms(profile):
    strategy = profile["runtime"]["strategy"]
    model = profile["model"]
    if strategy["expert_model_parallel_size"] <= 1 or not _is_moe_model(model):
        return 0.0
    bandwidth_gbps, latency_us, _ = _communication_link(profile, "ep", "collective")
    repetitions = strategy["acc_step"] * _stage_moe_layers(profile) * _dispatcher_exchanges(model)
    volume_bytes = _activation_bytes(profile) * float(
        model.get("moe_router_topk", DEFAULT_MOE_ROUTER_TOPK)
    )
    return _collective_ms(
        volume_bytes,
        bandwidth_gbps,
        latency_us,
        strategy["expert_model_parallel_size"],
        repetitions,
    )


def _estimate_recompute_ms(profile, compute_ms):
    strategy = profile["runtime"]["strategy"]
    if not strategy["use_recompute"]:
        return 0.0
    stage_layers = _stage_layers(profile["model"]["num_layers"], strategy)
    recompute_layers = strategy["recompute_num_layers"] or stage_layers
    layer_ratio = min(recompute_layers, stage_layers) / stage_layers
    return (
        compute_ms
        * layer_ratio
        * _recompute_granularity_factor(strategy["recompute_granularity"])
        * _recompute_method_factor(strategy["recompute_method"])
    )


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
    vocab_size = hidden_size * 16.0
    dense_params = (12.0 * hidden_size * hidden_size * model["num_layers"]) + (
        vocab_size * hidden_size
    )
    dense_partition = (
        strategy["tensor_model_parallel_size"]
        * strategy["pipeline_model_parallel_size"]
    )
    dense_bytes = (dense_params * BF16_BYTES) / dense_partition
    if not _is_moe_model(model):
        return dense_bytes
    expert_bytes = _moe_expert_params(model) * BF16_BYTES
    expert_partition = dense_partition * _effective_expert_parallel_size(strategy, model)
    return dense_bytes + (expert_bytes / expert_partition)


def _apply_overlap(raw_ms, profile, field):
    overlap = profile["hardware"]["cost_model"]["overlap"][field]
    return raw_ms * (1.0 - overlap)


def _collective_ms(volume_bytes, bandwidth_gbps, latency_us, group_size, repetitions):
    if group_size <= 1:
        return 0.0
    collective_factor = 2.0 * (group_size - 1) / group_size
    scaled_volume = volume_bytes * collective_factor
    scaled_repetitions = repetitions * max(log2(group_size), 1.0)
    return _transfer_ms(scaled_volume, bandwidth_gbps, latency_us, scaled_repetitions)


def _transfer_ms(volume_bytes, bandwidth_gbps, latency_us, repetitions):
    if repetitions <= 0:
        return 0.0
    bandwidth_bytes_per_second = (bandwidth_gbps * GIGA) / BITS_PER_BYTE
    transfer_ms = (
        volume_bytes * repetitions / bandwidth_bytes_per_second
    ) * MILLISECONDS_PER_SECOND
    latency_ms = repetitions * (latency_us / MILLISECONDS_PER_SECOND)
    return transfer_ms + latency_ms


def _flops_to_ms(total_flops, tflops):
    return (total_flops / (tflops * TERA)) * MILLISECONDS_PER_SECOND


def _effective_tflops(profile):
    compute = profile["hardware"]["compute"]
    return min(float(compute["bf16_tflops"]), float(compute["attention_tflops"]))


def _dense_parallel_factor(strategy):
    return strategy["tensor_model_parallel_size"] * strategy["context_parallel_size"]


def _is_moe_model(model):
    return int(model.get("num_experts") or 0) > 1


def _stage_moe_layers(profile):
    model = profile["model"]
    strategy = profile["runtime"]["strategy"]
    return _stage_layers(model["num_layers"], strategy) * _moe_layer_ratio(model)


def _moe_layer_ratio(model):
    moe_layer_freq = model.get("moe_layer_freq")
    if isinstance(moe_layer_freq, list) and moe_layer_freq:
        return sum(1 for layer in moe_layer_freq if layer) / len(moe_layer_freq)
    return 1.0 if _is_moe_model(model) else 0.0


def _moe_expert_params(model):
    hidden_size = float(model["hidden_size"])
    moe_hidden_size = float(model.get("moe_ffn_hidden_size", hidden_size * 4.0))
    num_experts = float(model["num_experts"])
    moe_layers = model["num_layers"] * _moe_layer_ratio(model)
    return moe_layers * num_experts * hidden_size * moe_hidden_size * 2.0


def _effective_expert_parallel_size(strategy, model):
    ep_size = strategy["expert_model_parallel_size"]
    return float(max(1, min(ep_size, int(model.get("num_experts") or 1))))


def _dispatcher_exchanges(model):
    if str(model.get("moe_token_dispatcher_type", "allgather")).lower() == "alltoall":
        return MOE_ALLTOALL_EXCHANGES
    return MOE_ALLGATHER_EXCHANGES


def _communication_link(profile, kind, traffic):
    interconnect = profile["hardware"]["interconnect"]
    if _group_spans_nodes(profile, kind):
        host_device = interconnect["host_device"]
        return (
            float(host_device["bandwidth_gbps"]),
            float(host_device["latency_us"]),
            "host_device",
        )
    intra_node = interconnect["intra_node"]
    if traffic == "p2p":
        return (
            float(intra_node["p2p_bandwidth_gbps"]),
            float(intra_node["p2p_latency_us"]),
            intra_node["fabric"],
        )
    return (
        float(intra_node["all_reduce_bandwidth_gbps"]),
        float(intra_node["all_reduce_latency_us"]),
        intra_node["fabric"],
    )


def _group_spans_nodes(profile, kind):
    runtime = profile["runtime"]
    if runtime["nnodes"] <= 1:
        return False
    if kind in ("tp", "ep", "pp"):
        return _group_size(profile, kind) > runtime["nproc_per_node"]
    if kind == "dp":
        return runtime["strategy"]["data_parallel_size"] > 1
    raise ValueError(f"Unsupported communication kind: {kind}")


def _group_size(profile, kind):
    strategy = profile["runtime"]["strategy"]
    if kind == "tp":
        return strategy["tensor_model_parallel_size"]
    if kind == "ep":
        return strategy["expert_model_parallel_size"]
    if kind == "pp":
        return strategy["pipeline_model_parallel_size"]
    raise ValueError(f"Unsupported communication kind: {kind}")


def _recompute_granularity_factor(granularity):
    if granularity == "selective":
        return RECOMPUTE_SELECTIVE_FACTOR
    return RECOMPUTE_FULL_FACTOR


def _recompute_method_factor(method):
    if method == "block":
        return BLOCK_RECOMPUTE_FACTOR
    return UNIFORM_RECOMPUTE_FACTOR
