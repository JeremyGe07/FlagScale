from flagscale.runner.auto_tuner.plan.schema import ModelPlan


def summarize_plan(plan: ModelPlan) -> dict[str, object]:
    return {
        "total_layers": plan.total_layers,
        "stage_count": len(plan.stages),
        "vpp_stage_segment_counts": [len(stage.segments) for stage in plan.stages],
        "contract": {
            "world_size": None if plan.contract is None else plan.contract.world_size,
            "micro_batch_size": None if plan.contract is None else plan.contract.micro_batch_size,
            "gradient_accumulation_steps": (
                None if plan.contract is None else plan.contract.gradient_accumulation_steps
            ),
            "global_batch_size": None if plan.contract is None else plan.contract.global_batch_size,
        },
        "stages": [
            {
                "stage_id": stage.stage_id,
                "device_group": list(stage.device_group),
                "segments": [
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "strategy": dict(segment.strategy),
                    }
                    for segment in stage.segments
                ],
            }
            for stage in plan.stages
        ],
        "transitions": [
            {
                "source_stage_id": transition.source_stage_id,
                "target_stage_id": transition.target_stage_id,
                "kind": transition.kind,
                "metadata": dict(transition.metadata),
            }
            for transition in plan.transitions
        ],
    }


def extract_homogeneous_strategy(plan: ModelPlan) -> dict[str, object] | None:
    if not plan.stages or plan.transitions:
        return None
    stage_segments = [segment for stage in plan.stages for segment in stage.segments]
    if not stage_segments:
        return None
    strategy = dict(stage_segments[0].strategy)
    for segment in stage_segments[1:]:
        if dict(segment.strategy) != strategy:
            return None
    return strategy


__all__ = ["extract_homogeneous_strategy", "summarize_plan"]
