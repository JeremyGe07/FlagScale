import json
from types import MappingProxyType

from omegaconf import OmegaConf


def to_json_ready(value):
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    if OmegaConf.is_config(value):
        return to_json_ready(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, MappingProxyType):
        return to_json_ready(dict(value))
    if isinstance(value, dict):
        return {str(key): to_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [to_json_ready(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    return value


def dump_json_string(value):
    return json.dumps(to_json_ready(value), ensure_ascii=False)
