from copy import deepcopy

from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.chip_profile import get_chip_profile_or_none

REQUIRED_MODEL_FIELDS = (
    "num_layers",
    "hidden_size",
    "num_attention_heads",
    "global_batch_size",
    "seq_length",
)
REQUIRED_STRATEGY_FIELDS = (
    "data_parallel_size",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "micro_batch_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "sequence_parallel",
    "use_distributed_optimizer",
    "use_recompute",
)


def build_cost_profile(config, strategy):
    profile = get_chip_profile_or_none(config)
    if profile is None:
        raise ValueError(
            "build_cost_profile requires an attached chip profile. "
            "Call attach_chip_profile(config) first."
        )

    config_dict = _to_plain_mapping(config, "config")
    strategy_dict = _to_plain_mapping(strategy, "strategy")
    return {
        "hardware": _build_hardware_profile(profile),
        "runtime": _build_runtime_profile(config_dict, strategy_dict),
        "model": _build_model_profile(config_dict),
    }


def _build_hardware_profile(profile):
    profile_dict = _to_plain_mapping(profile, "chip profile")
    return {
        "identity": deepcopy(
            _require_mapping(profile_dict, "identity", "chip profile")
        ),
        "memory": deepcopy(
            _require_mapping(profile_dict, "memory", "chip profile")
        ),
        "compute": deepcopy(
            _require_mapping(profile_dict, "compute", "chip profile")
        ),
        "interconnect": deepcopy(
            _require_mapping(profile_dict, "interconnect", "chip profile")
        ),
        "cost_model": deepcopy(
            _require_mapping(profile_dict, "cost_model", "chip profile")
        ),
    }


def _build_runtime_profile(config, strategy):
    experiment = _require_mapping(config, "experiment", "config")
    runner = _require_mapping(experiment, "runner", "config.experiment")
    auto_tuner = _require_mapping(experiment, "auto_tuner", "config.experiment")
    runner_nnodes = _read_positive_int(
        runner,
        "nnodes",
        "config.experiment.runner",
        None,
    )
    runner_nproc_per_node = _read_positive_int(
        runner,
        "nproc_per_node",
        "config.experiment.runner",
        None,
    )
    nnodes = _read_positive_int(
        auto_tuner,
        "nnodes",
        "config.experiment.auto_tuner",
        runner_nnodes,
    )
    nproc_per_node = _read_positive_int(
        auto_tuner,
        "nproc_per_node",
        "config.experiment.auto_tuner",
        runner_nproc_per_node,
    )
    world_size = _read_positive_int(
        auto_tuner,
        "cards",
        "config.experiment.auto_tuner",
        nnodes * nproc_per_node,
    )
    return {
        "nnodes": nnodes,
        "nproc_per_node": nproc_per_node,
        "world_size": world_size,
        "strategy": {
            field: _require_field(strategy, field, "strategy")
            for field in REQUIRED_STRATEGY_FIELDS
        },
    }


def _build_model_profile(config):
    train = _require_mapping(config, "train", "config")
    model = _require_mapping(train, "model", "config.train")
    return {
        field: _require_field(model, field, "config.train.model")
        for field in REQUIRED_MODEL_FIELDS
    }


def _to_plain_mapping(value, name):
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _require_mapping(mapping, key, path):
    value = _require_field(mapping, key, path)
    if not isinstance(value, dict):
        raise ValueError(f"{path}.{key} must be a mapping.")
    return value


def _require_field(mapping, key, path):
    if key not in mapping:
        raise ValueError(f"Missing required field: {path}.{key}")
    return mapping[key]


def _read_positive_int(mapping, key, path, default):
    value = mapping.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path}.{key} must be a positive integer.")
    return value
