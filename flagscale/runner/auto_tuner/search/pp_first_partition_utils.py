def validate_partition_inputs(num_layers: int, pp_degree: int, world_size: int):
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if pp_degree <= 0:
        raise ValueError("pp_degree must be positive")
    if world_size % pp_degree != 0:
        raise ValueError("world_size must be divisible by pp_degree")


def balanced_stage_counts(
    num_layers: int,
    stage_biases: tuple[float, ...],
) -> tuple[int, ...]:
    remaining_layers = num_layers
    remaining_bias = float(sum(stage_biases))
    counts = []
    total_stages = len(stage_biases)
    for stage_index, bias in enumerate(stage_biases):
        remaining_stages = total_stages - stage_index
        if remaining_stages == 1:
            counts.append(remaining_layers)
            break
        target_total = (remaining_layers + remaining_bias) / remaining_stages
        target_layers = round(target_total - bias)
        min_layers = 1
        max_layers = remaining_layers - (remaining_stages - 1)
        layer_count = min(max(target_layers, min_layers), max_layers)
        counts.append(layer_count)
        remaining_layers -= layer_count
        remaining_bias -= bias
    return tuple(counts)


def build_stage_ranges(stage_counts: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    start = 0
    ranges = []
    for count in stage_counts:
        end = start + count - 1
        ranges.append((start, end))
        start = end + 1
    return tuple(ranges)


def build_contiguous_groups(world_size: int, pp_degree: int) -> tuple[tuple[int, ...], ...]:
    group_size = world_size // pp_degree
    return tuple(
        tuple(range(offset, offset + group_size))
        for offset in range(0, world_size, group_size)
    )


__all__ = [
    "balanced_stage_counts",
    "build_contiguous_groups",
    "build_stage_ranges",
    "validate_partition_inputs",
]
