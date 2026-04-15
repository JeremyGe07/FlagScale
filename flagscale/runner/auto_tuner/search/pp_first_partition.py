from dataclasses import dataclass

from flagscale.runner.auto_tuner.search.searcher import get_first_last_num_layers_for_pp

PROVENANCE_HEURISTIC = "heuristic"
POLICY_LAYER_COUNT_BALANCED = "layer_count_balanced"


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


def generate_partition_candidates(
    *,
    num_layers: int,
    pp_degree: int,
    world_size: int,
    partition_policy: str,
    max_partitions: int,
) -> list[PartitionCandidate]:
    if max_partitions <= 0:
        return []
    if partition_policy != POLICY_LAYER_COUNT_BALANCED:
        raise ValueError(f"Unsupported partition policy: {partition_policy}")
    return [
        build_layer_count_balanced_partition(
            num_layers=num_layers,
            pp_degree=pp_degree,
            world_size=world_size,
        )
    ][:max_partitions]


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
    "POLICY_LAYER_COUNT_BALANCED",
    "PROVENANCE_HEURISTIC",
    "PartitionCandidate",
    "build_layer_count_balanced_partition",
    "generate_partition_candidates",
    "is_power_of_two",
]
