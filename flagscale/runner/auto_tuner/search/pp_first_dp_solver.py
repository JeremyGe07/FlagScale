from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

LOCAL_STAGE_COST_FIELD = "stage_time_cost"
TRANSITION_TOTAL_FIELD = "transition_total_ms"


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen_values = {key: _freeze_value(value) for key, value in dict(values).items()}
    return MappingProxyType(frozen_values)


@dataclass(frozen=True)
class StageLevelDPChain:
    stage_strategies: tuple[Mapping[str, Any], ...]
    dp_aggregate_cost: float
    transition_breakdown: tuple[Mapping[str, Any], ...]
    stage_cost_breakdown: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_strategies", _freeze_tuple(self.stage_strategies))
        object.__setattr__(self, "transition_breakdown", _freeze_tuple(self.transition_breakdown))
        object.__setattr__(self, "stage_cost_breakdown", _freeze_tuple(self.stage_cost_breakdown))


def solve_stage_level_dp(
    stage_candidates: Sequence[Sequence[Mapping[str, Any]]],
    transition_costs: (
        Mapping[Any, Any]
        | Callable[[Mapping[str, Any], Mapping[str, Any]], Any]
    ),
    *,
    planner=None,
    max_results: int | None = None,
) -> tuple[StageLevelDPChain, ...]:
    resolved_max_results = _resolve_max_results(planner, max_results)
    if resolved_max_results <= 0:
        return ()
    validated_stages = _validate_stage_candidates(stage_candidates)
    total_stage_count = len(validated_stages)
    frontier = _initial_frontier(validated_stages[0])
    for stage_index, candidates in enumerate(validated_stages[1:], start=1):
        frontier = _advance_frontier(
            frontier,
            candidates,
            stage_index,
            transition_costs,
            resolved_max_results,
            total_stage_count,
        )
    final_chains = [chain for chains in frontier for chain in chains]
    return tuple(_top_k(final_chains, resolved_max_results))


def _validate_stage_candidates(stage_candidates):
    if not stage_candidates:
        raise ValueError("stage_candidates must not be empty")
    validated_stages = []
    for stage_index, candidates in enumerate(stage_candidates):
        if not candidates:
            raise ValueError(f"stage_candidates[{stage_index}] must not be empty")
        validated_stages.append(tuple(candidates))
    return tuple(validated_stages)


def _resolve_max_results(planner, max_results):
    if max_results is not None:
        return int(max_results)
    if planner is None:
        raise ValueError("planner.max_dp_results_per_partition is required")
    if isinstance(planner, Mapping):
        value = planner.get("max_dp_results_per_partition")
    else:
        value = getattr(planner, "max_dp_results_per_partition", None)
    if value is None:
        raise ValueError("planner.max_dp_results_per_partition is required")
    return int(value)


def _initial_frontier(candidates):
    return tuple(
        (_initial_partial(candidate, 0, candidate_index),)
        for candidate_index, candidate in enumerate(candidates)
    )


def _initial_partial(candidate, stage_index, candidate_index):
    stage_cost = _local_stage_cost(candidate)
    return StageLevelDPChain(
        stage_strategies=(_freeze_mapping(candidate),),
        dp_aggregate_cost=stage_cost,
        transition_breakdown=(),
        stage_cost_breakdown=(
            {
                "stage_index": stage_index,
                "candidate_index": candidate_index,
                LOCAL_STAGE_COST_FIELD: stage_cost,
            },
        ),
    )


def _advance_frontier(
    previous_frontier,
    candidates,
    stage_index,
    transition_costs,
    max_results,
    total_stage_count,
):
    next_frontier = []
    for current_index, candidate in enumerate(candidates):
        extended = []
        for previous_index, previous_partials in enumerate(previous_frontier):
            for partial in previous_partials:
                extended.append(
                    _extend_partial(
                        partial,
                        candidate,
                        stage_index,
                        previous_index,
                        current_index,
                        transition_costs,
                        total_stage_count,
                    )
                )
        next_frontier.append(tuple(_top_k(extended, max_results)))
    return tuple(next_frontier)


def _extend_partial(
    partial,
    candidate,
    stage_index,
    previous_candidate_index,
    current_candidate_index,
    transition_costs,
    total_stage_count,
):
    previous_candidate = partial.stage_strategies[-1]
    transition_cost_breakdown = _transition_cost(
        previous_candidate,
        candidate,
        previous_candidate_index,
        current_candidate_index,
        transition_costs,
        stage_index - 1,
        total_stage_count,
    )
    transition_cost = _transition_total_ms(transition_cost_breakdown)
    stage_cost = _local_stage_cost(candidate)
    return StageLevelDPChain(
        stage_strategies=partial.stage_strategies + (_freeze_mapping(candidate),),
        dp_aggregate_cost=partial.dp_aggregate_cost + stage_cost + transition_cost,
        transition_breakdown=partial.transition_breakdown
        + (
            {
                "source_stage_index": stage_index - 1,
                "target_stage_index": stage_index,
                "source_candidate_index": previous_candidate_index,
                "target_candidate_index": current_candidate_index,
                "transition_cost_breakdown": transition_cost_breakdown,
            },
        ),
        stage_cost_breakdown=partial.stage_cost_breakdown
        + (
            {
                "stage_index": stage_index,
                "candidate_index": current_candidate_index,
                LOCAL_STAGE_COST_FIELD: stage_cost,
            },
        ),
    )


def _top_k(chains, max_results):
    return sorted(chains, key=_chain_sort_key)[:max_results]


def _chain_sort_key(chain):
    return (chain.dp_aggregate_cost, _chain_signature(chain))


def _chain_signature(chain):
    return tuple(_candidate_signature(candidate) for candidate in chain.stage_strategies)


def _candidate_signature(candidate):
    return tuple(sorted((key, _sort_value(value)) for key, value in candidate.items()))


def _local_stage_cost(candidate):
    if LOCAL_STAGE_COST_FIELD not in candidate:
        raise ValueError(f"stage candidates must define {LOCAL_STAGE_COST_FIELD}")
    return float(candidate[LOCAL_STAGE_COST_FIELD])


def _transition_cost(
    previous_candidate,
    current_candidate,
    previous_candidate_index,
    current_candidate_index,
    transition_costs,
    boundary_index,
    total_stage_count,
):
    if callable(transition_costs):
        return _normalize_transition_cost(
            transition_costs(previous_candidate, current_candidate)
        )
    boundary_key = (boundary_index, previous_candidate_index, current_candidate_index)
    if boundary_key in transition_costs:
        return _normalize_transition_cost(transition_costs[boundary_key])
    if total_stage_count > 2:
        raise ValueError(
            "transition_costs mapping for multi-stage DP must use boundary-aware keys "
            "like (stage_boundary_index, previous_candidate_index, current_candidate_index)"
        )
    key_options = (
        (previous_candidate_index, current_candidate_index),
        (
            _candidate_signature(previous_candidate),
            _candidate_signature(current_candidate),
        ),
    )
    for key in key_options:
        if key in transition_costs:
            return _normalize_transition_cost(transition_costs[key])
    raise ValueError(f"missing transition cost for {boundary_key!r}")


def _normalize_transition_cost(transition_cost):
    if isinstance(transition_cost, Mapping):
        breakdown = _freeze_mapping(transition_cost)
        _transition_total_ms(breakdown)
        return breakdown
    total = float(transition_cost)
    return _freeze_mapping({TRANSITION_TOTAL_FIELD: total})


def _transition_total_ms(transition_cost_breakdown):
    if TRANSITION_TOTAL_FIELD not in transition_cost_breakdown:
        raise ValueError(
            f"transition_cost_breakdown must define {TRANSITION_TOTAL_FIELD}"
        )
    return float(transition_cost_breakdown[TRANSITION_TOTAL_FIELD])


def _sort_value(value):
    if isinstance(value, Mapping):
        return tuple(sorted((key, _sort_value(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_sort_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_sort_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_sort_value(item) for item in value), key=repr))
    return (type(value).__name__, repr(value))


def _freeze_tuple(values):
    return tuple(_freeze_value(value) for value in values)


__all__ = ["StageLevelDPChain", "solve_stage_level_dp"]
