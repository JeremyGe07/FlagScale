from flagscale.runner.utils import flatten_dict_to_args


def test_flatten_dict_to_args_skips_none_values():
    config = {
        "checkpoint": {
            "load": None,
            "ckpt_format": None,
        },
        "train_iters": 1,
        "use_flash_attn": True,
    }

    args = flatten_dict_to_args(config)

    assert "--load" not in args
    assert "--ckpt-format" not in args
    assert args == ["--train-iters", "1", "--use-flash-attn"]
