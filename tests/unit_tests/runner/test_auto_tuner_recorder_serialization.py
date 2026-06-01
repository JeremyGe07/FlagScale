from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.record.recorder import Recorder


def test_recorder_save_serializes_omegaconf_list_values(tmp_path):
    (tmp_path / "auto_tuner").mkdir()
    config = OmegaConf.create({"experiment": {"exp_dir": str(tmp_path)}})
    recorder = Recorder(config)

    recorder.save(
        [
            {
                "idx": 1,
                "performance": 123.4,
                "gpu_utilization": OmegaConf.create([0.1, 1.0]),
            }
        ]
    )

    history = (tmp_path / "auto_tuner" / "history.csv").read_text(encoding="utf-8")
    assert "[0.1, 1.0]" in history
