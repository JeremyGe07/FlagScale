from omegaconf import OmegaConf

from flagscale.runner.runner_train import _get_args_megatron


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
