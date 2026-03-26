import inspect
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
MEGATRON_DIR = ROOT_DIR / "third_party" / "Megatron-LM"

for path in (ROOT_DIR, MEGATRON_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from flagscale.train.train import evaluate


def test_evaluate_accepts_eval_iters_without_breaking_extra_valid_position():
    params = list(inspect.signature(evaluate).parameters)

    assert "extra_valid_index" in params
    assert "eval_iters" in params
    assert params.index("extra_valid_index") < params.index("eval_iters")
