from dataclasses import dataclass

from flagscale.runner.auto_tuner.search.searcher import get_first_last_num_layers_for_pp
from flagscale.runner.auto_tuner.search.pp_first_partition_utils import (
    balanced_stage_counts,
    build_contiguous_groups,
    build_stage_ranges,
    validate_partition_inputs,
)

DEFAULT_LAYER_PARAM_FACTOR = 12
DEFAULT_LAYER_TIME_FACTOR = 48
DEFAULT_SEQUENCE_TIME_FACTOR = 1
DEFAULT_INPUT_EDGE_TIME_FACTOR = 0.5
DEFAULT_OUTPUT_EDGE_TIME_FACTOR = 1.25
PROVENANCE_HEURISTIC = "heuristic"
POLICY_LAYER_COUNT_BALANCED = "layer_count_balanced"
POLICY_PARAM_BALANCED = "param_balanced"
POLICY_PROFILE_TIME_BALANCED = "profile_time_balanced"


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
    stage_ranges = build_stage_ranges(stage_counts)
    device_groups = build_contiguous_groups(world_size, pp_degree)
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
    validate_partition_inputs(num_layers, pp_degree, world_size)
    stage_counts = _param_balanced_stage_counts(
        num_layers=num_layers,
        pp_degree=pp_degree,
        hidden_size=hidden_size,
        padded_vocab_size=padded_vocab_size,
    )
    return PartitionCandidate(
        pp_degree=pp_degree,
        stage_ranges=build_stage_ranges(stage_counts),
        device_groups=build_contiguous_groups(world_size, pp_degree),
        partition_policy=POLICY_PARAM_BALANCED,
        topology_signature=f"contiguous-equal:{pp_degree}x{world_size // pp_degree}",
        provenance=PROVENANCE_HEURISTIC,
        legality_flags=(),
    )


def build_profile_time_balanced_partition(
    *,
    num_layers: int,
    pp_degree: int,
    world_size: int,
    hidden_size: int,
    padded_vocab_size: int,
    seq_length: int,
) -> PartitionCandidate:
    if hidden_size <= 0 or padded_vocab_size <= 0 or seq_length <= 0:
        raise ValueError("hidden_size, padded_vocab_size and seq_length must be positive")
    validate_partition_inputs(num_layers, pp_degree, world_size)
    stage_counts = _profile_time_balanced_stage_counts(
        num_layers=num_layers,
        pp_degree=pp_degree,
        hidden_size=hidden_size,
        padded_vocab_size=padded_vocab_size,
        seq_length=seq_length,
    )
    return PartitionCandidate(
        pp_degree=pp_degree,
        stage_ranges=build_stage_ranges(stage_counts),
        device_groups=build_contiguous_groups(world_size, pp_degree),
        partition_policy=POLICY_PROFILE_TIME_BALANCED,
        topology_signature=f"contiguous-equal:{pp_degree}x{world_size // pp_degree}",
        provenance=PROVENANCE_HEURISTIC,
        legality_flags=("heuristic:profile_time_balanced",),
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
    seq_length: int | None = None,
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
        seq_length=seq_length,
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


def _build_partition_candidates(
    *,
    num_layers: int,
    pp_degree: int,
    world_size: int,
    partition_policy,
    max_partitions: int,
    hidden_size: int | None,
    padded_vocab_size: int | None,
    seq_length: int | None,
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
                seq_length=seq_length,
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
    seq_length: int | None,
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
    if policy == POLICY_PROFILE_TIME_BALANCED:
        return build_profile_time_balanced_partition(
            num_layers=num_layers,
            pp_degree=pp_degree,
            world_size=world_size,
            hidden_size=hidden_size or 0,
            padded_vocab_size=padded_vocab_size or 0,
            seq_length=seq_length or 0,
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
    return balanced_stage_counts(num_layers=num_layers, stage_biases=tuple(stage_biases))


def _profile_time_balanced_stage_counts(
    *,
    num_layers: int,
    pp_degree: int,
    hidden_size: int,
    padded_vocab_size: int,
    seq_length: int,
) -> tuple[int, ...]:
    layer_units = (
        DEFAULT_LAYER_TIME_FACTOR * hidden_size * hidden_size
        + DEFAULT_SEQUENCE_TIME_FACTOR * seq_length * hidden_size
    )
    edge_units = padded_vocab_size * hidden_size
    stage_biases = [0.0] * pp_degree
    stage_biases[0] = DEFAULT_INPUT_EDGE_TIME_FACTOR * edge_units / layer_units
    stage_biases[-1] = DEFAULT_OUTPUT_EDGE_TIME_FACTOR * edge_units / layer_units
    return balanced_stage_counts(num_layers=num_layers, stage_biases=tuple(stage_biases))


__all__ = [
    "DEFAULT_LAYER_PARAM_FACTOR",
    "DEFAULT_LAYER_TIME_FACTOR",
    "POLICY_LAYER_COUNT_BALANCED",
    "POLICY_PARAM_BALANCED",
    "POLICY_PROFILE_TIME_BALANCED",
    "PROVENANCE_HEURISTIC",
    "PartitionCandidate",
    "build_layer_count_balanced_partition",
    "build_param_balanced_partition",
    "build_profile_time_balanced_partition",
    "generate_partition_candidates",
    "is_power_of_two",
]
