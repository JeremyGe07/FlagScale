import os
import subprocess
from pathlib import Path

from flagscale.runner.auto_tuner.profile_acquisition.backends.base import ProfileBackend
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
NCCL_TESTS_BIN_DIR_ENV = "NCCL_TESTS_BIN_DIR"
P2P_BINARY = "p2p_bw"
ALL_REDUCE_BINARY = "all_reduce_perf"
NCCL_TABLE_MIN_COLUMNS = 4


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
        nccl_tests_bin_dir=None,
        ngpus=DEFAULT_TEST_GPUS,
    ):
        p2p_command = _resolve_collective_command(
            command=p2p_command,
            binary_name=P2P_BINARY,
            nccl_tests_bin_dir=nccl_tests_bin_dir,
            ngpus=ngpus,
        )
        all_reduce_command = _resolve_collective_command(
            command=all_reduce_command,
            binary_name=ALL_REDUCE_BINARY,
            nccl_tests_bin_dir=nccl_tests_bin_dir,
            ngpus=ngpus,
        )
        return {
            "p2p": self._collect_collective("p2p", p2p_command),
            "all_reduce": self._collect_collective("all_reduce", all_reduce_command),
        }

    def _collect_collective(self, name, command):
        output = _run_command(command)
        metrics = _parse_key_value_output(output)
        return AcquisitionMeasurement(
            kind="collective_bw_latency",
            source="nvidia",
            metrics=metrics,
            metadata={"collective": name},
        )


def _run_command(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
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


def _resolve_collective_command(command, binary_name, nccl_tests_bin_dir, ngpus):
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
