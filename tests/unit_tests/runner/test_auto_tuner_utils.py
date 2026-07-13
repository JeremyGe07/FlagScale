import sys
from types import ModuleType, SimpleNamespace

from flagscale.runner.auto_tuner import utils


class ConfigNode(dict):
    def __getattr__(self, key):
        return self[key]


def _install_fake_megatron_tokenizer(monkeypatch):
    tokenizer_module = ModuleType("megatron.training.tokenizer.tokenizer")
    tokenizer_module._vocab_size_with_padding = lambda vocab_size, args: vocab_size

    monkeypatch.setitem(sys.modules, "megatron", ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.training", ModuleType("megatron.training"))
    monkeypatch.setitem(
        sys.modules,
        "megatron.training.tokenizer",
        ModuleType("megatron.training.tokenizer"),
    )
    monkeypatch.setitem(sys.modules, "megatron.training.tokenizer.tokenizer", tokenizer_module)


def test_convert_config_to_megatron_args_does_not_write_stdout(monkeypatch, capsys):
    _install_fake_megatron_tokenizer(monkeypatch)
    monkeypatch.setattr(utils, "_ensure_megatron_path", lambda: None)

    config = SimpleNamespace(
        train=SimpleNamespace(
            model=ConfigNode(
                hidden_size=128,
                num_attention_heads=4,
                num_layers=4,
                ffn_hidden_size=512,
                seq_length=1024,
            ),
            system=ConfigNode(),
            data=SimpleNamespace(tokenizer=SimpleNamespace(vocab_size=151936)),
        )
    )
    strategy = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "data_parallel_size": 4,
        "expert_model_parallel_size": 1,
        "use_distributed_optimizer": False,
        "micro_batch_size": 1,
        "num_layers_per_virtual_pipeline_stage": None,
        "sequence_parallel": False,
        "recompute_granularity": None,
        "recompute_method": None,
        "recompute_num_layers": None,
        "context_parallel_size": 1,
    }

    utils.convert_config_to_megatron_args(config, strategy)

    assert capsys.readouterr().out == ""
