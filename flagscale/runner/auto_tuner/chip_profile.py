from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf

REQUIRED_SECTIONS = (
    "identity",
    "memory",
    "compute",
    "interconnect",
    "kernel_support",
    "topology",
    "strategy_hints",
)
REQUIRED_FIELDS = {
    "identity": ("name", "vendor", "chip_class"),
    "memory": ("total_memory_mb", "bandwidth_gbps"),
    "compute": ("bf16_tflops", "attention_tflops"),
    "kernel_support": ("transformer_engine", "flash_attention", "fused_rmsnorm"),
    "topology": ("max_nodes", "devices_per_node", "homogeneous_only"),
    "strategy_hints": (
        "default_search_priority",
        "max_tensor_model_parallel_size",
        "max_pipeline_model_parallel_size",
        "disabled_dims",
    ),
}
INTERCONNECT_FIELDS = {
    "intra_node": (
        "fabric",
        "p2p_bandwidth_gbps",
        "p2p_latency_us",
        "all_reduce_bandwidth_gbps",
        "all_reduce_latency_us",
    ),
    "host_device": ("bandwidth_gbps", "latency_us"),
}
ALLOWED_PRIORITIES = {"memory", "performance"}
REPO_ROOT = Path(__file__).resolve().parents[3]


def load_chip_profile(profile_path):
    resolved_path = _resolve_profile_path(profile_path)
    profile = OmegaConf.to_container(OmegaConf.load(resolved_path), resolve=True)
    if not isinstance(profile, dict):
        raise ValueError("Chip profile must be a mapping at the top level.")
    return _normalize_chip_profile(profile)


def attach_chip_profile(config):
    auto_tuner_cfg = config.experiment.get("auto_tuner", None)
    if auto_tuner_cfg is None or "chip_profile" not in auto_tuner_cfg:
        return None

    chip_profile_cfg = auto_tuner_cfg.chip_profile
    if "profile" in chip_profile_cfg:
        return chip_profile_cfg.profile
    if "path" not in chip_profile_cfg:
        raise ValueError("experiment.auto_tuner.chip_profile.path is required.")

    chip_profile_cfg.profile = load_chip_profile(chip_profile_cfg.path)
    return chip_profile_cfg.profile


def get_attached_chip_profile(config):
    auto_tuner_cfg = config.experiment.get("auto_tuner", None)
    if auto_tuner_cfg is None or "chip_profile" not in auto_tuner_cfg:
        return None

    chip_profile_cfg = auto_tuner_cfg.chip_profile
    if "profile" not in chip_profile_cfg:
        return None
    return chip_profile_cfg.profile


def _resolve_profile_path(profile_path):
    path = Path(str(profile_path)).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, REPO_ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"Chip profile file does not exist: {profile_path}")


def _normalize_chip_profile(profile):
    normalized = deepcopy(profile)
    _validate_sections(normalized)
    _validate_interconnect(normalized["interconnect"])
    _validate_boolean_fields("kernel_support", normalized["kernel_support"])
    _validate_topology(normalized["topology"])
    _validate_strategy_hints(normalized["strategy_hints"])
    return normalized


def _validate_sections(profile):
    missing_sections = [section for section in REQUIRED_SECTIONS if section not in profile]
    if missing_sections:
        missing = ", ".join(missing_sections)
        raise ValueError(f"Missing required chip profile sections: {missing}")

    for section_name, fields in REQUIRED_FIELDS.items():
        section = _require_mapping(section_name, profile[section_name])
        _require_keys(section_name, section, fields)

    _require_positive_fields("memory", profile["memory"], REQUIRED_FIELDS["memory"])
    _require_positive_fields("compute", profile["compute"], REQUIRED_FIELDS["compute"])


def _validate_interconnect(interconnect):
    section = _require_mapping("interconnect", interconnect)
    for name, fields in INTERCONNECT_FIELDS.items():
        sub_section = _require_mapping(f"interconnect.{name}", section.get(name))
        _require_keys(f"interconnect.{name}", sub_section, fields)
        numeric_fields = [field for field in fields if field != "fabric"]
        _require_positive_fields(f"interconnect.{name}", sub_section, numeric_fields)
        if name == "intra_node" and not isinstance(sub_section["fabric"], str):
            raise ValueError("Chip profile field 'interconnect.intra_node.fabric' must be a string.")


def _validate_boolean_fields(section_name, section):
    for key in REQUIRED_FIELDS[section_name]:
        value = section[key]
        if not isinstance(value, bool):
            raise ValueError(f"Chip profile field '{section_name}.{key}' must be a boolean.")


def _validate_topology(topology):
    _require_positive_fields("topology", topology, ("max_nodes", "devices_per_node"))
    if not isinstance(topology["homogeneous_only"], bool):
        raise ValueError("Chip profile field 'topology.homogeneous_only' must be a boolean.")


def _validate_strategy_hints(strategy_hints):
    priority = strategy_hints["default_search_priority"]
    if priority not in ALLOWED_PRIORITIES:
        raise ValueError("Chip profile field 'strategy_hints.default_search_priority' is invalid.")

    _require_positive_fields(
        "strategy_hints",
        strategy_hints,
        ("max_tensor_model_parallel_size", "max_pipeline_model_parallel_size"),
    )
    disabled_dims = strategy_hints["disabled_dims"]
    if not isinstance(disabled_dims, dict):
        raise ValueError("Chip profile field 'strategy_hints.disabled_dims' must be a mapping.")
    for dim, values in disabled_dims.items():
        if not isinstance(dim, str) or not isinstance(values, list):
            raise ValueError("Each disabled chip-profile dim must map to a list of values.")


def _require_mapping(section_name, section):
    if not isinstance(section, dict):
        raise ValueError(f"Chip profile section '{section_name}' must be a mapping.")
    return section


def _require_keys(section_name, section, fields):
    missing_fields = [field for field in fields if field not in section]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"Chip profile section '{section_name}' is missing keys: {missing}")


def _require_positive_fields(section_name, section, fields):
    for field in fields:
        value = section[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Chip profile field '{section_name}.{field}' must be positive.")
