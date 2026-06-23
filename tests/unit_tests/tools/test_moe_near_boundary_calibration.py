import csv

from omegaconf import OmegaConf

from tools.experiments.moe_near_boundary_calibration import (
    build_near_boundary_template,
    main,
)


FIELDNAMES = [
    "idx",
    "data_parallel_size",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "expert_model_parallel_size",
    "context_parallel_size",
    "use_distributed_optimizer",
    "sequence_parallel",
    "micro_batch_size",
    "use_recompute",
    "recompute_method",
    "recompute_granularity",
    "recompute_num_layers",
    "memory_model",
    "max_mem",
    "performance",
    "error",
]


def _write_history(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _row(
    idx,
    *,
    dp=1,
    tp=1,
    pp=4,
    ep=1,
    mbs=4,
    memory_model=76000,
    max_mem="",
    performance="",
    error="",
):
    return {
        "idx": str(idx),
        "data_parallel_size": str(dp),
        "tensor_model_parallel_size": str(tp),
        "pipeline_model_parallel_size": str(pp),
        "expert_model_parallel_size": str(ep),
        "context_parallel_size": "1",
        "use_distributed_optimizer": "False",
        "sequence_parallel": "False" if tp == 1 else "True",
        "micro_batch_size": str(mbs),
        "use_recompute": "True",
        "recompute_method": "block",
        "recompute_granularity": "full",
        "recompute_num_layers": "4",
        "memory_model": str(memory_model),
        "max_mem": str(max_mem),
        "performance": str(performance),
        "error": error,
    }


def test_build_near_boundary_template_prefers_memory_boundary_and_diverse_families():
    rows = [
        _row(1, pp=4, memory_model=69000, max_mem=78500, performance=13140),
        _row(2, pp=4, memory_model=75500, error="CUDA out of memory"),
        _row(3, tp=2, pp=2, memory_model=71000, max_mem=78000, performance=20000),
        _row(4, tp=4, pp=1, memory_model=45000, max_mem=52000, performance=50000),
        _row(5, dp=2, pp=1, memory_model=81000, error="NameError"),
    ]

    template = build_near_boundary_template(
        rows,
        template_name="portable-near-boundary",
        sample_count=3,
        gpu_memory_mb=81920,
        per_family_limit=2,
    )

    assert template["name"] == "portable-near-boundary"
    assert [task["source_history_idx"] for task in template["tasks"]] == [1, 2, 3]
    assert {task["historical_status"] for task in template["tasks"]} == {"success", "oom"}
    assert template["tasks"][0]["branch"] == "dp1_tp1_pp4_ep1_spfalse_distfalse"
    assert template["tasks"][2]["sequence_parallel"] is True
    assert 5 not in {task["source_history_idx"] for task in template["tasks"]}


def test_main_writes_file_template_without_model_specific_fields(tmp_path):
    history = tmp_path / "history.csv"
    output = tmp_path / "template.yaml"
    _write_history(
        history,
        [
            _row(10, pp=4, memory_model=70000, max_mem=78000, performance=13140),
            _row(11, pp=4, memory_model=76000, error="NCCL error: out of memory"),
        ],
    )

    exit_code = main(
        [
            "--history-csv",
            str(history),
            "--output",
            str(output),
            "--template-name",
            "near-boundary",
            "--sample-count",
            "2",
            "--gpu-memory-mb",
            "81920",
        ]
    )

    payload = OmegaConf.to_container(OmegaConf.load(output), resolve=True)
    assert exit_code == 0
    assert payload["name"] == "near-boundary"
    assert len(payload["tasks"]) == 2
    assert "model_name" not in payload
