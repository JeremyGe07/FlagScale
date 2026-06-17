from pathlib import Path

from tools.experiments.qwen25_family_repeat_bench import (
    build_train_args,
    parse_iteration_performance,
    strategy_specs,
)


def test_parse_iteration_performance_uses_post_warmup_mean(tmp_path: Path):
    log = tmp_path / "stdout.log"
    log.write_text(
        "\n".join(
            [
                "elapsed time per iteration (ms): 4586.7 |",
                "elapsed time per iteration (ms): 2703.9 |",
                "elapsed time per iteration (ms): 2690.3 |",
            ]
        )
    )

    assert parse_iteration_performance(tmp_path) == 2697.1


def test_strategy_labels_generate_expected_parallelism_flags(tmp_path: Path):
    specs = {spec.label: spec for spec in strategy_specs()}

    dp4_args = build_train_args(specs["dp4_tp1_pp1"], tmp_path)
    dp1_args = build_train_args(specs["dp1_tp2_pp2_uniform14"], tmp_path)

    assert "--tensor-model-parallel-size 1" in " ".join(dp4_args)
    assert "--pipeline-model-parallel-size 1" in " ".join(dp4_args)
    assert "--use-distributed-optimizer" in dp4_args
    assert "--sequence-parallel" not in dp4_args
    assert "--recompute-method" not in dp4_args

    assert "--tensor-model-parallel-size 2" in " ".join(dp1_args)
    assert "--pipeline-model-parallel-size 2" in " ".join(dp1_args)
    assert "--sequence-parallel" in dp1_args
    assert "--use-distributed-optimizer" not in dp1_args
    assert "--recompute-method uniform" in " ".join(dp1_args)
    assert "--recompute-num-layers 14" in " ".join(dp1_args)
