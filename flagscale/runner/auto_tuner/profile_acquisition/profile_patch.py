from copy import deepcopy

from omegaconf import OmegaConf


def merge_profile_patch(profile, patch):
    merged = deepcopy(profile)
    for dotted_key, value in patch.items():
        _set_nested_value(merged, dotted_key, value)
    return merged


def write_profile_patch(profile_in_path, profile_out_path, patch):
    profile = OmegaConf.to_container(OmegaConf.load(profile_in_path), resolve=True)
    merged = merge_profile_patch(profile, patch)
    OmegaConf.save(config=OmegaConf.create(merged), f=profile_out_path)
    return merged


def _set_nested_value(mapping, dotted_key, value):
    parts = dotted_key.split(".")
    current = mapping
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value
