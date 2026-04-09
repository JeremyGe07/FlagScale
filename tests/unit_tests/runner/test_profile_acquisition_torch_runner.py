from io import StringIO
from unittest.mock import patch

import pytest

from tools.profile_acquisition.measure_collectives_torch import (
    main as torch_collective_main,
    parse_args,
)


def test_parse_args_rejects_unknown_collective():
    with pytest.raises(SystemExit):
        parse_args(['--collective', 'invalid'])


def test_main_prints_key_value_metrics():
    output = StringIO()
    with patch(
        'tools.profile_acquisition.measure_collectives_torch.measure_collective',
        return_value={'bandwidth_gbps': 71.5, 'latency_us': 2.8},
    ):
        exit_code = torch_collective_main(
            ['--collective', 'p2p'],
            stdout=output,
        )

    assert exit_code == 0
    assert output.getvalue().strip().splitlines() == [
        'bandwidth_gbps=71.5',
        'latency_us=2.8',
    ]
