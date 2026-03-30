from flagscale.runner.auto_tuner.chip_profile import get_attached_chip_profile

REASON_DISABLED_DIMS = "chip_profile.disabled_dims"
REASON_MAX_TENSOR = "chip_profile.max_tensor_model_parallel_size"
REASON_MAX_PIPELINE = "chip_profile.max_pipeline_model_parallel_size"
REASON_TOPOLOGY = "chip_profile.topology"


def prune_by_chip_profile(config, strategy, history=None):
    del history
    profile = get_attached_chip_profile(config)
    if profile is None:
        return False, None

    if _runtime_topology_exceeds_profile(config, profile["topology"]):
        return True, REASON_TOPOLOGY
    if _has_disabled_dim(strategy, profile["strategy_hints"]["disabled_dims"]):
        return True, REASON_DISABLED_DIMS
    if _exceeds_tensor_parallel_limit(strategy, profile["strategy_hints"]):
        return True, REASON_MAX_TENSOR
    if _exceeds_pipeline_parallel_limit(strategy, profile["strategy_hints"]):
        return True, REASON_MAX_PIPELINE
    return False, None


def _runtime_topology_exceeds_profile(config, topology):
    runtime_nodes = config.experiment.runner.get("nnodes", config.experiment.auto_tuner.nnodes)
    runtime_devices = config.experiment.runner.get(
        "nproc_per_node", config.experiment.auto_tuner.nproc_per_node
    )
    return runtime_nodes > topology["max_nodes"] or runtime_devices > topology["devices_per_node"]


def _has_disabled_dim(strategy, disabled_dims):
    for key, values in disabled_dims.items():
        if strategy.get(key) in values:
            return True
    return False


def _exceeds_tensor_parallel_limit(strategy, strategy_hints):
    value = strategy.get("tensor_model_parallel_size")
    return value is not None and value > strategy_hints["max_tensor_model_parallel_size"]


def _exceeds_pipeline_parallel_limit(strategy, strategy_hints):
    value = strategy.get("pipeline_model_parallel_size")
    return value is not None and value > strategy_hints["max_pipeline_model_parallel_size"]
