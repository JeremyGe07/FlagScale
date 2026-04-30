from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_yaml(relative_path: str):
    with (_repo_root() / relative_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_qwen3_14b_a800_autotune_uses_local_runner_and_assets():
    cfg = _load_yaml("examples/qwen3/conf/train_auto_tuner_14b_4xa800_revalidation.yaml")
    train_cfg = _load_yaml("examples/qwen3/conf/train/14b_4xa800_revalidation.yaml")

    assert cfg["defaults"][1] == {"train": "14b_4xa800_revalidation"}
    assert cfg["experiment"]["runner"]["nnodes"] == 1
    assert cfg["experiment"]["runner"]["nproc_per_node"] == 4
    assert cfg["experiment"]["envs"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert cfg["experiment"]["auto_tuner"]["control"]["max_time_per_task"] == 900
    assert (
        train_cfg["data"]["data_path"]
        == "/home/user-a800/GuoChanZhiSuan/data/pile_wikipedia_demo/pile_wikipedia_demo"
    )
    assert (
        train_cfg["data"]["tokenizer"]["tokenizer_path"]
        == "/home/user-a800/GuoChanZhiSuan/models/Qwen3-14B"
    )


def test_qwen3_14b_a800_autotune_keeps_broad_search_space():
    cfg = _load_yaml("examples/qwen3/conf/train_auto_tuner_14b_4xa800_revalidation.yaml")
    train_cfg = _load_yaml("examples/qwen3/conf/train/14b_4xa800_revalidation.yaml")
    space = cfg["experiment"]["auto_tuner"]["space"]

    assert space["data_parallel_size"] == "auto"
    assert space["use_distributed_optimizer"] == "auto"
    assert space["tensor_model_parallel_size"] == [1, 2, 4]
    assert space["sequence_parallel"] == "auto"
    assert space["pipeline_model_parallel_size"] == "auto"
    assert space["num_layers_per_virtual_pipeline_stage"] == "auto"
    assert space["context_parallel_size"] == "auto"
    assert space["micro_batch_size"] == "auto"
    assert space["use_recompute"] == "auto"
    assert space["recompute_method"] == "auto"
    assert space["recompute_granularity"] == "auto"
    assert space["recompute_num_layers"] == "auto"
    assert train_cfg["model"]["num_attention_heads"] % 4 == 0
    assert train_cfg["model"]["num_query_groups"] % 4 == 0
