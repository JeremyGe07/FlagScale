from types import MappingProxyType

from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.record.recorder import Recorder


def test_recorder_save_serializes_stage_heterogeneous_strategy_metadata(tmp_path):
    (tmp_path / "auto_tuner").mkdir()
    config = OmegaConf.create({"experiment": {"exp_dir": str(tmp_path)}})
    recorder = Recorder(config)
    history = [
        {
            "idx": 1,
            "performance": 9870.8,
            "plan_kind": "stage-heterogeneous",
            "stage_partition_ranges": ((0, 13), (14, 27)),
            "stage_device_groups": ((0,), (1,)),
            "stage_strategies": (
                MappingProxyType({"tensor_model_parallel_size": 1, "use_recompute": False}),
                MappingProxyType({"tensor_model_parallel_size": 1, "use_recompute": False}),
            ),
        }
    ]

    recorder.save(history)
    saved = recorder.read()

    assert saved[0]["performance"] == 9870.8
    assert saved[0]["plan_kind"] == "stage-heterogeneous"
    assert saved[0]["stage_partition_ranges"] == [[0, 13], [14, 27]]
    assert saved[0]["stage_device_groups"] == [[0], [1]]
    assert saved[0]["stage_strategies"] == [
        {"tensor_model_parallel_size": 1, "use_recompute": False},
        {"tensor_model_parallel_size": 1, "use_recompute": False},
    ]
