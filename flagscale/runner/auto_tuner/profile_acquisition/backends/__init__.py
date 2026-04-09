from flagscale.runner.auto_tuner.profile_acquisition.backends.nvidia import (
    NvidiaProfileBackend,
)

BACKEND_REGISTRY = {
    "nvidia": NvidiaProfileBackend,
}


def build_backend(name):
    try:
        backend_cls = BACKEND_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported profile acquisition backend: {name}") from exc
    return backend_cls()


__all__ = ["BACKEND_REGISTRY", "NvidiaProfileBackend", "build_backend"]
