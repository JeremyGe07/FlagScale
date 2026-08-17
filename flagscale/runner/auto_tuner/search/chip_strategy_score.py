_INTERCONNECT_FABRIC_WEIGHT = {
    "nvlink": 1.0,
    "xgmi": 1.0,
    "mtlink": 1.0,
    "hccs": 0.8,
    "pcie": 0.4,
}
_DEFAULT_FABRIC_WEIGHT = 0.5
_PERFORMANCE_PRIORITY_WEIGHT = 0.4
_MEMORY_PRIORITY_WEIGHT = 0.2
_COMPUTE_REFERENCE_TFLOPS = 100.0


def build_chip_score(strategy, profile):
    priority = profile["strategy_hints"]["default_search_priority"]
    reasons = []
    score = 0.0
    score += _score_tp_pp_match(strategy, profile, reasons)
    score += _score_priority(strategy, priority, reasons)
    score += _score_compute_preference(strategy, profile["compute"], priority, reasons)
    return {"score": score, "priority": priority, "reasons": reasons}


def _score_tp_pp_match(strategy, profile, reasons):
    tp_size = strategy["tensor_model_parallel_size"]
    pp_size = strategy["pipeline_model_parallel_size"]
    devices_per_node = profile["topology"]["devices_per_node"]
    fabric = profile["interconnect"]["intra_node"]["fabric"].lower()
    fabric_weight = _INTERCONNECT_FABRIC_WEIGHT.get(fabric, _DEFAULT_FABRIC_WEIGHT)
    tp_fit = min(tp_size, devices_per_node) / devices_per_node
    pp_penalty = max(pp_size - 1, 0) * (1.0 - fabric_weight)
    reasons.append(
        f"tp={tp_size} fits {min(tp_size, devices_per_node)}/{devices_per_node} intra-node devices on {fabric}"
    )
    if pp_penalty > 0:
        reasons.append(f"pp={pp_size} pays extra staging cost on {fabric}")
    return tp_fit * (1.0 + fabric_weight) - pp_penalty


def _score_priority(strategy, priority, reasons):
    tp_size = strategy["tensor_model_parallel_size"]
    pp_size = strategy["pipeline_model_parallel_size"]
    if priority == "performance":
        reasons.append("default priority favors higher tensor parallelism and shallower pipeline")
        return tp_size * _PERFORMANCE_PRIORITY_WEIGHT - pp_size * _MEMORY_PRIORITY_WEIGHT
    reasons.append("default priority favors deeper pipeline over wider tensor parallelism")
    return pp_size * _MEMORY_PRIORITY_WEIGHT - tp_size * _MEMORY_PRIORITY_WEIGHT


def _score_compute_preference(strategy, compute, priority, reasons):
    tp_size = strategy["tensor_model_parallel_size"]
    pp_size = strategy["pipeline_model_parallel_size"]
    compute_strength = min(
        compute["attention_tflops"] / _COMPUTE_REFERENCE_TFLOPS,
        compute["bf16_tflops"] / _COMPUTE_REFERENCE_TFLOPS,
    )
    if priority == "performance":
        reasons.append("compute profile prefers strategies that keep more work inside each stage")
        return compute_strength * (tp_size / max(pp_size, 1))
    reasons.append("compute profile softly penalizes over-wide tensor parallelism")
    return compute_strength / max(tp_size, 1)
