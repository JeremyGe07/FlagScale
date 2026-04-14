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
                    {"start": segment.start, "end": segment.end} for segment in stage.segments
                ],
            }
            for stage in plan.stages
        ],
    }


def extract_homogeneous_strategy(plan: ModelPlan) -> dict[str, object] | None:
    if not plan.stages or any(len(stage.segments) != 1 for stage in plan.stages):
        return None
    strategy = dict(plan.stages[0].segments[0].strategy)
    for stage in plan.stages[1:]:
        if dict(stage.segments[0].strategy) != strategy:
            return None
    return strategy


__all__ = ["extract_homogeneous_strategy", "summarize_plan"]
