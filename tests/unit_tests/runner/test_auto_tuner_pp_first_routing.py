import pytest

from flagscale.runner.auto_tuner.search.pp_first_searcher import PPFirstSearcher
from flagscale.runner.auto_tuner.tuner import AutoTuner
from tests.unit_tests.runner.test_auto_tuner_pp_first_planner import _config


def test_auto_tuner_uses_pp_first_searcher_when_planner_enabled(tmp_path):
    config = _config(tmp_path)
    config.experiment.auto_tuner.planner = {"name": "pp_first"}

    tuner = AutoTuner(config)

    assert isinstance(tuner.searcher, PPFirstSearcher)


def test_auto_tuner_rejects_unknown_planner_name(tmp_path):
    config = _config(tmp_path)
    config.experiment.auto_tuner.planner = {"name": "unknown"}

    with pytest.raises(ValueError, match="Unsupported auto_tuner planner"):
        AutoTuner(config)
