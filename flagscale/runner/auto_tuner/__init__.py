from importlib import import_module
from typing import Any

__all__ = ["AutoTuner", "ServeAutoTunner"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    tuner_module = import_module("flagscale.runner.auto_tuner.tuner")
    value = getattr(tuner_module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
