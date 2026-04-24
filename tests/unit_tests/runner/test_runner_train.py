import shlex
import subprocess
import sys

from omegaconf import OmegaConf

from flagscale.runner.runner_train import _generate_run_script_train, _get_args_megatron


def test_get_args_megatron_includes_train_root_level_flags():
    config = OmegaConf.create(
        {
            "experiment": {"task": {"backend": "megatron"}},
            "train": {
                "mock_data": True,
                "system": {},
                "model": {"train_iters": 1},
                "data": {"split": 1},
            },
        }
    )

    args = _get_args_megatron(config)

    assert "--mock-data" in args
    assert "--train-iters" in args
    assert "--split" in args


def test_generate_run_script_preserves_quoted_nested_literal_args(tmp_path):
    result_file = tmp_path / "argv.txt"
    nested_literal = "{'kind': 'segment-redistribution', 'requires_sequence_parallel': True}"
    command = shlex.join(
        [
            sys.executable,
            "-c",
            "import pathlib, sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])",
            str(result_file),
            nested_literal,
        ]
    )
    config = OmegaConf.create(
        {
            "experiment": {"runner": {}},
            "train": {
                "system": {
                    "checkpoint": {
                        "load": str(tmp_path / "load"),
                        "save": str(tmp_path / "save"),
                    },
                    "logging": {
                        "log_dir": str(tmp_path / "logs"),
                        "scripts_dir": str(tmp_path / "logs" / "scripts"),
                        "pids_dir": str(tmp_path / "logs" / "pids"),
                        "details_dir": str(tmp_path / "logs" / "details"),
                        "tensorboard_dir": str(tmp_path / "tensorboard"),
                        "wandb_save_dir": str(tmp_path / "wandb"),
                    },
                }
            },
        }
    )

    script = _generate_run_script_train(
        config,
        "localhost",
        0,
        command,
        background=False,
        root_dir=str(tmp_path),
    )
    subprocess.run(["bash", script], check=True)

    assert result_file.read_text() == nested_literal
