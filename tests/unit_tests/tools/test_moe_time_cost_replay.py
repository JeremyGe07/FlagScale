from types import SimpleNamespace

from tools.experiments.moe_time_cost_replay import (
    best_performance,
    hydra_overrides,
    normalize_history_value,
    topk_recall,
)


def test_normalize_history_value_handles_common_csv_values():
    assert normalize_history_value("") is None
    assert normalize_history_value(False) is False
    assert normalize_history_value("True") is True
    assert normalize_history_value("4.0") == 4
    assert normalize_history_value("block") == "block"


def test_best_performance_ignores_error_rows():
    rows = [
        {"performance": "", "error": "OOM"},
        {"performance": "20.0", "error": ""},
        {"performance": "10.0", "error": ""},
    ]

    assert best_performance(rows)["performance"] == "10.0"


def test_topk_recall_reports_best_kept_performance():
    rows = [
        {"performance": "10.0", "error": "", "rank": 9},
        {"performance": "12.0", "error": "", "rank": 4},
        {"performance": "30.0", "error": "", "rank": 1},
    ]

    assert topk_recall(rows, [4, 9]) == {
        4: {"successes_kept": 2, "best_kept": 12.0},
        9: {"successes_kept": 3, "best_kept": 10.0},
    }


def test_hydra_overrides_add_or_update_experimental_moe_options():
    args = SimpleNamespace(
        chip_profile_path=None,
        moe_time_cost_model="experimental_v1",
        moe_time_cost_option=["min_microbatch_efficiency=0.6"],
    )

    assert hydra_overrides(args) == [
        "++experiment.auto_tuner.algo.moe_time_cost_model=experimental_v1",
        (
            "++experiment.auto_tuner.algo.moe_time_cost_model_options."
            "min_microbatch_efficiency=0.6"
        ),
    ]
