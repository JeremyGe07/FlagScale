BYTES_PER_MEGABYTE = 1024.0 * 1024.0
TP_TRANSITION_COST_PER_UNIT_MS = 1.0
DP_TRANSITION_COST_PER_UNIT_MS = 1.0
ACTIVATION_TRANSFER_COST_PER_MB_MS = 1.0


def estimate_transition_cost(previous_stage, current_stage, config):
    previous = _validate_stage(previous_stage, "previous_stage")
    current = _validate_stage(current_stage, "current_stage")
    hidden_size = _require_positive_number(
        config.train.model.hidden_size,
        "train.model.hidden_size",
    )
    seq_length = _require_positive_number(
        config.train.model.seq_length,
        "train.model.seq_length",
    )

    tp_transition_ms = _scaled_transition_cost(
        previous["tensor_model_parallel_size"],
        current["tensor_model_parallel_size"],
        TP_TRANSITION_COST_PER_UNIT_MS,
    )
    dp_transition_ms = _scaled_transition_cost(
        previous["data_parallel_size"],
        current["data_parallel_size"],
        DP_TRANSITION_COST_PER_UNIT_MS,
    )
    activation_transfer_ms = _activation_transfer_cost(
        previous,
        current,
        hidden_size,
        seq_length,
    )
    transition_total_ms = tp_transition_ms + dp_transition_ms + activation_transfer_ms
    return {
        "tp_transition_ms": tp_transition_ms,
        "dp_transition_ms": dp_transition_ms,
        "activation_transfer_ms": activation_transfer_ms,
        "transition_total_ms": transition_total_ms,
    }


def _scaled_transition_cost(previous_value, current_value, scale_ms):
    if previous_value == current_value:
        return 0.0
    return abs(float(current_value) - float(previous_value)) * scale_ms


def _activation_transfer_cost(previous, current, hidden_size, seq_length):
    if previous["pipeline_model_parallel_size"] == current["pipeline_model_parallel_size"]:
        return 0.0
    micro_batch_size = min(previous["micro_batch_size"], current["micro_batch_size"])
    boundary_mb = hidden_size * seq_length * micro_batch_size * 2.0 / BYTES_PER_MEGABYTE
    pipeline_delta = abs(
        float(current["pipeline_model_parallel_size"])
        - float(previous["pipeline_model_parallel_size"])
    )
    return boundary_mb * pipeline_delta * ACTIVATION_TRANSFER_COST_PER_MB_MS


def _validate_stage(stage, stage_name):
    if not isinstance(stage, dict):
        raise ValueError(f"{stage_name} must be a mapping")
    required_fields = (
        "data_parallel_size",
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "micro_batch_size",
    )
    validated = {}
    for field in required_fields:
        validated[field] = _require_positive_number(stage.get(field), f"{stage_name}.{field}")
    return validated


def _require_positive_number(value, field_name):
    if value is None:
        raise ValueError(f"{field_name} is required")
    numeric = float(value)
    if numeric <= 0:
        raise ValueError(f"{field_name} must be positive")
    return numeric


__all__ = ["estimate_transition_cost"]
