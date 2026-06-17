def resolve_all_reduce_metrics(interconnect, group_size):
    return resolve_collective_metrics(interconnect, "all_reduce", group_size)


def resolve_collective_metrics(interconnect, collective, group_size):
    _validate_group_size(group_size)
    _validate_collective_name(collective)
    intra_node = interconnect["intra_node"]
    if "collective_profiles" not in intra_node:
        return _legacy_all_reduce_metrics(intra_node)
    profiles = intra_node["collective_profiles"]
    if not isinstance(profiles, dict):
        raise ValueError("interconnect.intra_node.collective_profiles must be a mapping.")

    if collective not in profiles:
        return _legacy_all_reduce_metrics(intra_node)
    collective_profiles = profiles[collective]
    if not isinstance(collective_profiles, dict):
        raise ValueError(
            f"interconnect.intra_node.collective_profiles.{collective} must be a mapping."
        )

    group_key = f"group_size_{group_size}"
    if group_key not in collective_profiles:
        return _legacy_all_reduce_metrics(intra_node)
    selected = collective_profiles[group_key]
    if not isinstance(selected, dict):
        raise ValueError(
            f"interconnect.intra_node.collective_profiles.{collective}.{group_key} "
            "must be a mapping."
        )
    return float(selected["bandwidth_gbps"]), float(selected["latency_us"])


def _validate_group_size(group_size):
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
        raise ValueError("Collective group size must be a positive integer.")


def _validate_collective_name(collective):
    if not isinstance(collective, str) or not collective:
        raise ValueError("Collective name must be a non-empty string.")


def _legacy_all_reduce_metrics(intra_node):
    return (
        float(intra_node["all_reduce_bandwidth_gbps"]),
        float(intra_node["all_reduce_latency_us"]),
    )
