#!/usr/bin/env python3
"""Safely materialize the pinned legacy Megatron backend for MUSA S5000.

This helper never resets or removes an existing backend. It initializes an
absent submodule, verifies the pinned commit from ``diff.yaml``, and applies
only patches that are not already present.
"""

import argparse
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = REPO_ROOT / "third_party" / "Megatron-LM"
PATCH_ROOT = REPO_ROOT / "hardware" / "MUSA_S5000" / "Megatron-LM"
MANIFEST_PATH = PATCH_ROOT / "diff.yaml"


def _git(*args, cwd=REPO_ROOT, check=True):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def _expected_backend_commit():
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    return manifest["backends_commit"]["Megatron-LM"]


def _ensure_initialized():
    if BACKEND_PATH.exists():
        probe = _git("rev-parse", "--show-toplevel", cwd=BACKEND_PATH, check=False)
        if (
            probe.returncode == 0
            and Path(probe.stdout.strip()).resolve() == BACKEND_PATH.resolve()
        ):
            return
    print(f"Initializing {BACKEND_PATH.relative_to(REPO_ROOT)}")
    _git("submodule", "update", "--init", "--", "third_party/Megatron-LM")


def _patch_state(patch_path):
    patch_arg = str(patch_path)
    pending = _git("apply", "--check", patch_arg, cwd=BACKEND_PATH, check=False)
    if pending.returncode == 0:
        return "pending"
    applied = _git(
        "apply",
        "--reverse",
        "--check",
        patch_arg,
        cwd=BACKEND_PATH,
        check=False,
    )
    if applied.returncode == 0:
        return "applied"
    return "conflict"


def prepare(check_only=False):
    _ensure_initialized()
    expected_commit = _expected_backend_commit()
    actual_commit = _git("rev-parse", "HEAD", cwd=BACKEND_PATH).stdout.strip()
    if actual_commit != expected_commit:
        raise RuntimeError(
            "Refusing to modify Megatron-LM at unexpected commit: "
            f"expected {expected_commit}, found {actual_commit}."
        )

    patch_paths = sorted(PATCH_ROOT.rglob("*.patch"))
    states = {patch_path: _patch_state(patch_path) for patch_path in patch_paths}
    conflicts = [path for path, state in states.items() if state == "conflict"]
    if conflicts:
        formatted = "\n".join(f"  - {path.relative_to(REPO_ROOT)}" for path in conflicts)
        raise RuntimeError(
            "Refusing a partial patch application because these patches are neither "
            f"pending nor already applied:\n{formatted}"
        )

    pending = [path for path, state in states.items() if state == "pending"]
    applied = [path for path, state in states.items() if state == "applied"]
    print(
        f"Megatron-LM commit {actual_commit}; "
        f"{len(applied)} patches already applied, {len(pending)} pending."
    )
    if check_only:
        return

    for patch_path in pending:
        print(f"Applying {patch_path.relative_to(REPO_ROOT)}")
        _git("apply", str(patch_path), cwd=BACKEND_PATH)
    print("Legacy MUSA S5000 backend is ready.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the backend commit and patch state without applying pending patches",
    )
    args = parser.parse_args()
    prepare(check_only=args.check)


if __name__ == "__main__":
    main()
