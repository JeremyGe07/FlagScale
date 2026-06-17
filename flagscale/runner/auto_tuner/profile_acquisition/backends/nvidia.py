import os
import subprocess
import sys
from pathlib import Path

from flagscale.runner.auto_tuner.profile_acquisition.backends.base import ProfileBackend
from flagscale.runner.auto_tuner.profile_acquisition.backends.nvidia_topology import (
    TOPOLOGY_COMMAND,
    select_representative_pairs,
)
from flagscale.runner.auto_tuner.profile_acquisition.models import AcquisitionMeasurement

DEVICE_MEMORY_COMMAND = [
    "nvidia-smi",
    "--query-gpu=memory.total",
    "--format=csv,noheader,nounits",
]
DEFAULT_TEST_MIN_BYTES = "8"
DEFAULT_TEST_MAX_BYTES = "128M"
DEFAULT_TEST_STEP_FACTOR = "2"
DEFAULT_TEST_GPUS = 2
P2P_TEST_GPUS = 2
NCCL_TESTS_BIN_DIR_ENV = "NCCL_TESTS_BIN_DIR"
P2P_BINARY = "p2p_bw"
ALL_REDUCE_BINARY = "all_reduce_perf"
NCCL_TABLE_MIN_COLUMNS = 4
RUNNER_NCCL_TESTS = "nccl_tests"
RUNNER_TORCH_NCCL = "torch_nccl"
TORCHRUN_MODULE = "torch.distributed.run"
TORCH_BENCHMARK_SCRIPT = "measure_collectives_torch.py"


class NvidiaProfileBackend(ProfileBackend):
    def collect_device_memory(self):
        output = _run_command(DEVICE_MEMORY_COMMAND)
        values = [int(line.strip()) for line in output.splitlines() if line.strip()]
        if not values:
            raise ValueError("NVIDIA device-memory query returned no values.")
        if len(set(values)) != 1:
            raise ValueError("NVIDIA adapter currently requires homogeneous device memory.")
        return AcquisitionMeasurement(
            kind="device_memory",
            source="nvidia",
            metrics={"total_memory_mb": values[0]},
            metadata={"device_count": len(values)},
        )

    def collect_collectives(
        self,
        p2p_command,
        all_reduce_command,
        runner=None,
        nccl_tests_bin_dir=None,
        ngpus=DEFAULT_TEST_GPUS,
        p2p_pair_classes=None,
        all_reduce_group_sizes=None,
    ):
        if p2p_pair_classes is not None or all_reduce_group_sizes is not None:
            return self._collect_extended_collectives(
                runner=runner,
                ngpus=ngpus,
                p2p_pair_classes=p2p_pair_classes,
                all_reduce_group_sizes=all_reduce_group_sizes,
            )
        p2p_command = _build_collective_command(
            command=p2p_command,
            binary_name=P2P_BINARY,
            collective="p2p",
            runner=runner,
            nccl_tests_bin_dir=nccl_tests_bin_dir,
            ngpus=ngpus,
        )
        all_reduce_command = _build_collective_command(
            command=all_reduce_command,
            binary_name=ALL_REDUCE_BINARY,
            collective="all_reduce",
            runner=runner,
            nccl_tests_bin_dir=nccl_tests_bin_dir,
            ngpus=ngpus,
        )
        return {
            "p2p": self._collect_collective("p2p", p2p_command),
            "all_reduce": self._collect_collective("all_reduce", all_reduce_command),
        }

    def _collect_extended_collectives(
        self,
        runner,
        ngpus,
        p2p_pair_classes,
        all_reduce_group_sizes,
    ):
        _validate_torch_runner(runner)
        p2p_classes = (
            self._collect_p2p_classes(p2p_pair_classes)
            if p2p_pair_classes is not None
            else {}
        )
        group_sizes = _resolve_group_sizes(all_reduce_group_sizes, ngpus)
        all_reduce_profiles = self._collect_group_size_profiles("all_reduce", group_sizes)
        all_to_all_profiles = self._collect_group_size_profiles("all_to_all", group_sizes)
        result = {
            "p2p": _legacy_p2p_measurement(p2p_classes)
            if p2p_classes
            else self._collect_collective("p2p", _build_torchrun_command("p2p", ngpus)),
            "all_reduce": all_reduce_profiles[max(all_reduce_profiles)],
            "all_reduce_profiles": all_reduce_profiles,
            "all_to_all_profiles": all_to_all_profiles,
        }
        if p2p_classes:
            result["p2p_classes"] = p2p_classes
        return result

    def _collect_p2p_classes(self, p2p_pair_classes):
        selected_pairs = select_representative_pairs(
            _run_command(TOPOLOGY_COMMAND),
            p2p_pair_classes,
        )
        measurements = {}
        for name, pair in selected_pairs.items():
            measurement = self._collect_collective(
                "p2p",
                _build_torchrun_command("p2p", P2P_TEST_GPUS),
                env=_with_visible_devices(pair.devices),
            )
            measurements[name] = _with_metadata(
                measurement,
                {"collective": "p2p", "gpu_pair": list(pair.devices), "pair_class": name},
            )
        return measurements

    def _collect_group_size_profiles(self, collective, group_sizes):
        profiles = {}
        for group_size in group_sizes:
            measurement = self._collect_collective(
                collective,
                _build_torchrun_command(collective, group_size),
            )
            profiles[group_size] = _with_metadata(
                measurement,
                {"collective": collective, "group_size": group_size},
            )
        return profiles

    def _collect_collective(self, name, command, env=None):
        output = _run_command(command, env=env)
        metrics = _parse_key_value_output(output)
        return AcquisitionMeasurement(
            kind="collective_bw_latency",
            source="nvidia",
            metrics=metrics,
            metadata={"collective": name},
        )


def _run_command(command, env=None):
    run_kwargs = {"capture_output": True, "text": True, "check": False}
    if env is not None:
        run_kwargs["env"] = env
    result = subprocess.run(command, **run_kwargs)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}\n"
            f"{result.stderr}"
        )
    return result.stdout


def _parse_key_value_output(output):
    metrics = {}
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "=" not in stripped:
            return _parse_nccl_table_output(output)
        key, value = stripped.split("=", 1)
        metrics[key] = float(value)
    return metrics


def _validate_torch_runner(runner):
    if runner != RUNNER_TORCH_NCCL:
        raise ValueError("Topology-aware collective acquisition requires runner torch_nccl.")


def _resolve_group_sizes(all_reduce_group_sizes, ngpus):
    if all_reduce_group_sizes is None:
        return (ngpus,)
    group_sizes = tuple(all_reduce_group_sizes)
    if not group_sizes:
        raise ValueError("all_reduce_group_sizes must include at least one group size.")
    invalid = [group_size for group_size in group_sizes if group_size <= 0]
    if invalid:
        raise ValueError(f"all_reduce_group_sizes must be positive integers: {invalid}")
    return group_sizes


def _legacy_p2p_measurement(p2p_classes):
    if "sys" in p2p_classes:
        return p2p_classes["sys"]
    return min(p2p_classes.values(), key=lambda measurement: measurement.metrics["bandwidth_gbps"])


def _with_metadata(measurement, metadata):
    return AcquisitionMeasurement(
        kind=measurement.kind,
        source=measurement.source,
        metrics=dict(measurement.metrics),
        metadata=metadata,
    )


def _with_visible_devices(devices):
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(str(device) for device in devices),
    }


def _build_collective_command(
    command,
    binary_name,
    collective,
    runner,
    nccl_tests_bin_dir,
    ngpus,
):
    if runner == RUNNER_TORCH_NCCL:
        return _build_torchrun_command(collective, ngpus)
    if runner == RUNNER_NCCL_TESTS:
        return _resolve_nccl_test_command(command, binary_name, nccl_tests_bin_dir, ngpus)
    raise ValueError(f"Unknown collective runner: {runner}")


def _build_torchrun_command(collective, ngpus):
    repo_root = Path(__file__).resolve().parents[5]
    script_path = repo_root / "tools" / "profile_acquisition" / TORCH_BENCHMARK_SCRIPT
    return [
        sys.executable,
        "-m",
        TORCHRUN_MODULE,
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={ngpus}",
        str(script_path),
        "--collective",
        collective,
    ]


def _resolve_nccl_test_command(command, binary_name, nccl_tests_bin_dir, ngpus):
    if command:
        return command
    bin_dir = nccl_tests_bin_dir or os.environ.get(NCCL_TESTS_BIN_DIR_ENV)
    if not bin_dir:
        raise ValueError(
            f"{binary_name} requires an explicit command; set p2p_command/all_reduce_command "
            f"or {NCCL_TESTS_BIN_DIR_ENV}."
        )
    binary_path = Path(bin_dir) / binary_name
    return [
        str(binary_path),
        "-b",
        DEFAULT_TEST_MIN_BYTES,
        "-e",
        DEFAULT_TEST_MAX_BYTES,
        "-f",
        DEFAULT_TEST_STEP_FACTOR,
        "-g",
        str(ngpus),
    ]


def _parse_nccl_table_output(output):
    rows = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) >= NCCL_TABLE_MIN_COLUMNS:
            rows.append(tokens)
    if not rows:
        raise ValueError("Unable to parse NCCL benchmark output.")
    return {
        "bandwidth_gbps": _extract_bandwidth_gbps(rows[-1]),
        "latency_us": _extract_latency_us(rows[0]),
    }


def _extract_latency_us(row):
    if len(row) >= 13:
        return float(row[5])
    return float(row[-4])


def _extract_bandwidth_gbps(row):
    if len(row) >= 13:
        return max(float(row[7]), float(row[11]))
    return float(row[-2])
