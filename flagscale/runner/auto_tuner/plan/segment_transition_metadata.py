from flagscale.runner.auto_tuner.plan.schema import ExecutionContract, ModelPlan, TransitionPlan


def enrich_segment_transition_metadata(plan: ModelPlan) -> ModelPlan:
    stage_map = {stage.stage_id: stage for stage in plan.stages}
    batch_unit = _segment_batch_unit(plan.contract)
    transitions = []
    for transition in plan.transitions:
        if transition.kind != "segment-redistribution":
            transitions.append(transition)
            continue
        source_mesh = _segment_transition_mesh(
            stage_map,
            transition.source_stage_id,
            transition.source_segment_index,
        )
        target_mesh = _segment_transition_mesh(
            stage_map,
            transition.target_stage_id,
            transition.target_segment_index,
        )
        metadata = {
            **dict(transition.metadata),
            "source_mesh": source_mesh,
            "target_mesh": target_mesh,
            "batch_unit": batch_unit,
            "redistribution_kind": _redistribution_kind(source_mesh, target_mesh),
            "requires_sequence_parallel": source_mesh["tp"] != target_mesh["tp"],
        }
        transitions.append(
            TransitionPlan(
                source_stage_id=transition.source_stage_id,
                target_stage_id=transition.target_stage_id,
                kind=transition.kind,
                source_segment_index=transition.source_segment_index,
                target_segment_index=transition.target_segment_index,
                metadata=metadata,
            )
        )
    return ModelPlan(
        stages=plan.stages,
        transitions=tuple(transitions),
        contract=plan.contract,
        total_layers=plan.total_layers,
    )


def _segment_batch_unit(contract: ExecutionContract) -> int:
    if contract.global_batch_size is None:
        raise ValueError("segment runtime contract requires global_batch_size")
    if contract.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if contract.global_batch_size % contract.gradient_accumulation_steps != 0:
        raise ValueError("global_batch_size must be divisible by gradient_accumulation_steps")
    return contract.global_batch_size // contract.gradient_accumulation_steps


def _segment_transition_mesh(stage_map, stage_id: int | None, segment_index: int | None):
    if stage_id is None or segment_index is None:
        raise ValueError("segment-redistribution transitions require stage and segment indexes")
    stage = stage_map.get(stage_id)
    if stage is None or segment_index < 0 or segment_index >= len(stage.segments):
        raise ValueError("segment-redistribution transition indexes are out of range")
    strategy = stage.segments[segment_index].strategy
    return {
        "tp": _required_strategy_int(strategy, ("tensor_model_parallel_size", "tp")),
        "cp": _required_strategy_int(strategy, ("context_parallel_size", "cp")),
        "ep": _required_strategy_int(strategy, ("expert_model_parallel_size", "ep")),
        "dp": _required_strategy_int(strategy, ("data_parallel_size", "dp")),
        "pp": _segment_local_pipeline_size(strategy),
    }


def _redistribution_kind(source_mesh, target_mesh) -> str:
    tp_changed = source_mesh["tp"] != target_mesh["tp"]
    dp_changed = source_mesh["dp"] != target_mesh["dp"]
    if tp_changed and dp_changed:
        return "tp-dp"
    if tp_changed:
        return "tp-only"
    if dp_changed:
        return "dp-only"
    raise ValueError("redistribution_kind requires tp or dp changes across the boundary")


def _required_strategy_int(strategy, keys) -> int:
    for key in keys:
        value = strategy.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    raise ValueError(f"missing required strategy field {keys[0]}")


def _segment_local_pipeline_size(strategy) -> int:
    value = strategy.get("pp_local")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    value = strategy.get("pipeline_model_parallel_size")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 1


__all__ = ["enrich_segment_transition_metadata"]
