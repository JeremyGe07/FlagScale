from flagscale.runner.auto_tuner.plan.schema import ModelPlan
from flagscale.runner.auto_tuner.plan.summary import summarize_segment_runtime_contract


def build_segment_runtime_contract(plan: ModelPlan) -> dict[str, object]:
    return summarize_segment_runtime_contract(plan)


__all__ = ["build_segment_runtime_contract"]
