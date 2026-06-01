from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_yaml(relative_path: str):
    with (_repo_root() / relative_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_train_auto_tuner_qwen3_14b_4xa800_revalidation_has_local_runner_shape():
    cfg = _load_yaml("examples/qwen3/conf/train_auto_tuner_14b_4xa800_revalidation.yaml")

    assert cfg["defaults"][1] == {"train": "14b_4xa800_revalidation"}
    assert cfg["experiment"]["runner"]["nnodes"] == 1
    assert cfg["experiment"]["runner"]["nproc_per_node"] == 4
    assert cfg["experiment"]["envs"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert cfg["experiment"]["auto_tuner"]["control"]["train_iters"] == 3
    assert cfg["experiment"]["auto_tuner"]["control"]["max_time_per_task"] == 7200
    assert (
        cfg["experiment"]["exp_dir"]
        == "${oc.env:FLAGSCALE_OUTPUT_ROOT,./outputs}/qwen3_14b_a800_revalidation/default"
    )
    assert 'source "$CONDA_SH"' in cfg["experiment"]["cmds"]["before_start"]
    assert 'conda activate "$FLAGSCALE_TRAIN_ENV"' in cfg["experiment"]["cmds"]["before_start"]
    assert cfg["experiment"]["auto_tuner"]["space"]["use_distributed_optimizer"] == [True]
    assert cfg["experiment"]["auto_tuner"]["space"]["sequence_parallel"] == [True]
    assert cfg["experiment"]["auto_tuner"]["space"]["context_parallel_size"] == [1]


def test_qwen3_14b_4xa800_revalidation_points_to_local_assets():
    cfg = _load_yaml("examples/qwen3/conf/train/14b_4xa800_revalidation.yaml")

    assert (
        cfg["data"]["data_path"]
        == "${oc.env:PILE_WIKIPEDIA_DEMO_PATH,../data/pile_wikipedia_demo/pile_wikipedia_demo}"
    )
    assert (
        cfg["data"]["tokenizer"]["tokenizer_path"]
        == "${oc.env:QWEN3_14B_PATH,../models/Qwen3-14B}"
    )
    assert cfg["data"]["tokenizer"]["legacy_tokenizer"] is True
    assert cfg["system"]["logging"]["wandb_project"] == "qwen3_14b_a800_revalidation"
    assert cfg["system"]["overlap_param_gather"] is False
    assert cfg["model"]["optimizer"]["lr_scheduler"]["lr_wsd_decay_iters"] == 1
