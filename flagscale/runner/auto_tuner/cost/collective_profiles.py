def resolve_all_reduce_metrics(interconnect, group_size):
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
        raise ValueError("All-reduce group size must be a positive integer.")

    intra_node = interconnect["intra_node"]
    if "collective_profiles" not in intra_node:
        return _legacy_all_reduce_metrics(intra_node)
    profiles = intra_node["collective_profiles"]
    if not isinstance(profiles, dict):
        raise ValueError("interconnect.intra_node.collective_profiles must be a mapping.")

    if "all_reduce" not in profiles:
        return _legacy_all_reduce_metrics(intra_node)
    all_reduce_profiles = profiles["all_reduce"]
    if not isinstance(all_reduce_profiles, dict):
        raise ValueError(
            "interconnect.intra_node.collective_profiles.all_reduce must be a mapping."
        )

    group_key = f"group_size_{group_size}"
    if group_key not in all_reduce_profiles:
        return _legacy_all_reduce_metrics(intra_node)
    selected = all_reduce_profiles[group_key]
    if not isinstance(selected, dict):
        raise ValueError(
            f"interconnect.intra_node.collective_profiles.all_reduce.{group_key} "
            "must be a mapping."
        )
    return float(selected["bandwidth_gbps"]), float(selected["latency_us"])


def _legacy_all_reduce_metrics(intra_node):
    return (
        float(intra_node["all_reduce_bandwidth_gbps"]),
        float(intra_node["all_reduce_latency_us"]),
    )
