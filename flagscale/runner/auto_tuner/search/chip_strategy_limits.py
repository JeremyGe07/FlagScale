VALID_STRATEGY_DIMS = {
    "data_parallel_size",
    "use_distributed_optimizer",
    "tensor_model_parallel_size",
    "sequence_parallel",
    "pipeline_model_parallel_size",
    "num_layers_per_virtual_pipeline_stage",
    "use_recompute",
    "recompute_method",
    "recompute_granularity",
    "recompute_num_layers",
    "micro_batch_size",
    "context_parallel_size",
    "expert_model_parallel_size",
}


def validate_runtime_topology(config, profile):
    if profile is None:
        return

    topology = profile["topology"]
    runtime_nodes = config.experiment.runner.get("nnodes", config.experiment.auto_tuner.nnodes)
    runtime_devices = config.experiment.runner.get(
        "nproc_per_node", config.experiment.auto_tuner.nproc_per_node
    )
    if runtime_nodes > topology["max_nodes"]:
        raise ValueError("Runtime topology exceeds chip profile max_nodes.")
    if runtime_devices > topology["devices_per_node"]:
        raise ValueError("Runtime topology exceeds chip profile devices_per_node.")


def apply_dim_limits(space, profile):
    if profile is None:
        return {key: list(values) for key, values in space.items()}

    limited_space = {}
    strategy_hints = profile["strategy_hints"]
    _validate_disabled_dims(strategy_hints["disabled_dims"])
    for key, values in space.items():
        limited_values = list(values)
        if key == "tensor_model_parallel_size":
            limited_values = [
                value
                for value in limited_values
                if value <= strategy_hints["max_tensor_model_parallel_size"]
            ]
        if key == "pipeline_model_parallel_size":
            limited_values = [
                value
                for value in limited_values
                if value <= strategy_hints["max_pipeline_model_parallel_size"]
            ]
        disabled_values = strategy_hints["disabled_dims"].get(key, [])
        if disabled_values:
            limited_values = [value for value in limited_values if value not in disabled_values]
        if not limited_values:
            raise ValueError(f"Chip profile hard limits emptied search-space dim '{key}'.")
        limited_space[key] = limited_values
    return limited_space


def is_strategy_disabled_by_chip_profile(strategy, profile):
    if profile is None:
        return False

    strategy_hints = profile["strategy_hints"]
    if strategy["tensor_model_parallel_size"] > strategy_hints["max_tensor_model_parallel_size"]:
        return True
    if strategy["pipeline_model_parallel_size"] > strategy_hints["max_pipeline_model_parallel_size"]:
        return True

    disabled_dims = strategy_hints["disabled_dims"]
    return any(strategy.get(key) in values for key, values in disabled_dims.items())


def _validate_disabled_dims(disabled_dims):
    invalid_dims = sorted(set(disabled_dims) - VALID_STRATEGY_DIMS)
    if invalid_dims:
        dims = ", ".join(invalid_dims)
        raise ValueError(f"Chip profile disabled_dims contains invalid dim names: {dims}")
