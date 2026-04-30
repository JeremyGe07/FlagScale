from dataclasses import asdict

from flagscale.train.hetero.segment_runtime_args import parse_segment_runtime_args
from tests.unit_tests.runner.test_train_segment_runtime import _segment_runtime_config


def test_segment_runtime_spec_supports_dataclasses_asdict():
    spec = parse_segment_runtime_args(_segment_runtime_config())

    serialized = asdict(spec)

    transition = serialized["stages"][0]["transitions"][0]
    assert transition["source_mesh"]["tp"] == 2
    assert transition["target_mesh"]["dp"] == 2
