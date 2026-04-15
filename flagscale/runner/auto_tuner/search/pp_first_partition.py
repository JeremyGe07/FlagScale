from dataclasses import dataclass

from flagscale.runner.auto_tuner.search.searcher import get_first_last_num_layers_for_pp

DEFAULT_LAYER_PARAM_FACTOR = 12
PROVENANCE_HEURISTIC = "heuristic"
POLICY_LAYER_COUNT_BALANCED = "layer_count_balanced"
POLICY_PARAM_BALANCED = "param_balanced"


@dataclass(frozen=True)
class PartitionCandidate:
    pp_degree: int
    stage_ranges: tuple[tuple[int, int], ...]
    device_groups: tuple[tuple[int, ...], ...]
    partition_policy: str
    topology_signature: str
    provenance: str
    legality_flags: tuple[str, ...] = ()


def build_layer_count_balanced_partition(
    *,
    num_layers: int,
    pp_degree: int,
    world_size: int,
) -> PartitionCandidate:
    if pp_degree <= 0:
        raise ValueError("pp_degree must be positive")
    if world_size % pp_degree != 0:
        raise ValueError("world_size must be divisible by pp_degree")
    stage_counts = _stage_layer_counts(num_layers, pp_degree)
    stage_ranges = _build_stage_ranges(stage_counts)
    device_groups = _build_contiguous_groups(world_size, pp_degree)
    return PartitionCandidate(
        pp_degree=pp_degree,
        stage_ranges=stage_ranges,
        device_groups=device_groups,
        partition_policy=POLICY_LAYER_COUNT_BALANCED,
        topology_signature=f"contiguous-equal:{pp_degree}x{world_size // pp_degree}",
        provenance=PROVENANCE_HEURISTIC,
        legality_flags=(),
    )


def build_param_balanced_partition(
    *,
    num_layers: int,
    pp_degree: int,
    world_size: int,
    hidden_size: int,
    padded_vocab_size: int,
) -> PartitionCandidate:
    if hidden_size <= 0 or padded_vocab_size <= 0:
        raise ValueError("hidden_size and padded_vocab_size must be positive")
    _validate_partition_inputs(num_layers, pp_degree, world_size)
    stage_counts = _param_balanced_stage_counts(
        num_layers=num_layers,
        pp_degree=pp_degree,
        hidden_size=hidden_size,
        padded_vocab_size=padded_vocab_size,
    )
    return PartitionCandidate(
        pp_degree=pp_degree,
        stage_ranges=_build_stage_ranges(stage_counts),
        device_groups=_build_contiguous_groups(world_size, pp_degree),
        partition_policy=POLICY_PARAM_BALANCED,
        topology_signature=f"contiguous-equal:{pp_degree}x{world_size // pp_degree}",
        provenance=PROVENANCE_HEURISTIC,
        legality_flags=(),
    )


def generate_partition_candidates(
    *,
    num_layers: int,
    pp_degree: int,
    world_size: int,
    partition_policy,
    max_partitions: int,
    hidden_size: int | None = None,
    padded_vocab_size: int | None = None,
) -> list[PartitionCandidate]:
    if max_partitions <= 0:
        return []
    return _build_partition_candidates(
        num_layers=num_layers,
        pp_degree=pp_degree,
        world_size=world_size,
        partition_policy=partition_policy,
        max_partitions=max_partitions,
        hidden_size=hidden_size,
        padded_vocab_size=padded_vocab_size,
    )


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _stage_layer_counts(num_layers: int, pp_degree: int) -> tuple[int, ...]:
    if num_layers % pp_degree == 0:
        return tuple([num_layers // pp_degree] * pp_degree)
    if pp_degree == 1:
        return (num_layers,)
    if pp_degree == 2:
        first, last = get_first_last_num_layers_for_pp(num_layers, pp_degree)
        return (first, last)
    first, last = get_first_last_num_layers_for_pp(num_layers, pp_degree)
    middle_stages = pp_degree - 2
    middle_total = num_layers - first - last
    if middle_total <= 0 or middle_total % middle_stages != 0:
        raise ValueError("layer_count_balanced cannot derive middle stage allocation")
    middle = middle_total // middle_stages
    return (first,) + tuple([middle] * middle_stages) + (last,)


def _validate_partition_inputs(num_layers: int, pp_degree: int, world_size: int):
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if pp_degree <= 0:
        raise ValueError("pp_degree must be positive")
    if world_size % pp_degree != 0:
        raise ValueError("world_size must be divisible by pp_degree")


def _build_partition_candidates(
    *,
    num_layers: int,
    pp_degree: int,
    world_size: int,
    partition_policy,
    max_partitions: int,
    hidden_size: int | None,
    padded_vocab_size: int | None,
) -> list[PartitionCandidate]:
    candidates = []
    for policy in _normalize_partition_policies(partition_policy):
        candidates.append(
            _build_partition_candidate(
                policy=policy,
                num_layers=num_layers,
                pp_degree=pp_degree,
                world_size=world_size,
                hidden_size=hidden_size,
                padded_vocab_size=padded_vocab_size,
            )
        )
        if len(candidates) >= max_partitions:
            break
    return candidates


def _normalize_partition_policies(partition_policy) -> tuple[str, ...]:
    if isinstance(partition_policy, str):
        return (partition_policy,)
    return tuple(partition_policy)


def _build_partition_candidate(
    *,
    policy: str,
    num_layers: int,
    pp_degree: int,
    world_size: int,
    hidden_size: int | None,
    padded_vocab_size: int | None,
) -> PartitionCandidate:
    if policy == POLICY_LAYER_COUNT_BALANCED:
        return build_layer_count_balanced_partition(
            num_layers=num_layers,
            pp_degree=pp_degree,
            world_size=world_size,
        )
    if policy == POLICY_PARAM_BALANCED:
        return build_param_balanced_partition(
            num_layers=num_layers,
            pp_degree=pp_degree,
            world_size=world_size,
            hidden_size=hidden_size or 0,
            padded_vocab_size=padded_vocab_size or 0,
        )
    raise ValueError(f"Unsupported partition policy: {policy}")


def _param_balanced_stage_counts(
    *,
    num_layers: int,
    pp_degree: int,
    hidden_size: int,
    padded_vocab_size: int,
) -> tuple[int, ...]:
    layer_units = DEFAULT_LAYER_PARAM_FACTOR * hidden_size * hidden_size
    edge_units = padded_vocab_size * hidden_size
    stage_biases = [0.0] * pp_degree
    stage_biases[0] = edge_units / layer_units
    stage_biases[-1] = edge_units / layer_units
    return _balanced_stage_counts(num_layers=num_layers, stage_biases=tuple(stage_biases))


def _balanced_stage_counts(num_layers: int, stage_biases: tuple[float, ...]) -> tuple[int, ...]:
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


def _build_stage_ranges(stage_counts: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    start = 0
    ranges = []
    for count in stage_counts:
        end = start + count - 1
        ranges.append((start, end))
        start = end + 1
    return tuple(ranges)


def _build_contiguous_groups(world_size: int, pp_degree: int) -> tuple[tuple[int, ...], ...]:
    group_size = world_size // pp_degree
    return tuple(
        tuple(range(offset, offset + group_size))
        for offset in range(0, world_size, group_size)
    )


__all__ = [
    "DEFAULT_LAYER_PARAM_FACTOR",
    "POLICY_LAYER_COUNT_BALANCED",
    "POLICY_PARAM_BALANCED",
    "PROVENANCE_HEURISTIC",
    "PartitionCandidate",
    "build_layer_count_balanced_partition",
    "build_param_balanced_partition",
    "generate_partition_candidates",
    "is_power_of_two",
]
