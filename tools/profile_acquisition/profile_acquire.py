import argparse

from flagscale.runner.auto_tuner.profile_acquisition import (
    build_acquisition_dataset,
    fit_memory_bias,
    write_profile_patch,
)
from flagscale.runner.auto_tuner.profile_acquisition.backends import build_backend

RUNNER_NCCL_TESTS = 'nccl_tests'
RUNNER_TORCH_NCCL = 'torch_nccl'


def main(argv=None):
    args = _parse_args(argv)
    patch = _build_profile_patch(args)
    write_profile_patch(args.profile_in, args.profile_out, patch)
    return 0


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Acquire or calibrate chip-profile fields.")
    parser.add_argument("--profile-in", required=True)
    parser.add_argument("--profile-out", required=True)
    parser.add_argument("--backend")
    parser.add_argument("--history-csv")
    parser.add_argument("--tuner-log")
    parser.add_argument("--gpu-memory-mb", type=float)
    parser.add_argument("--fit-memory-bias", action="store_true")
    parser.add_argument("--measure-device-memory", action="store_true")
    parser.add_argument("--measure-collectives", action="store_true")
    parser.add_argument(
        "--collective-runner",
        choices=[RUNNER_NCCL_TESTS, RUNNER_TORCH_NCCL],
    )
    parser.add_argument("--p2p-command", nargs="+")
    parser.add_argument("--all-reduce-command", nargs="+")
    parser.add_argument("--nccl-tests-bin-dir")
    parser.add_argument("--ngpus", type=int, default=2)
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args):
    has_action = args.fit_memory_bias or args.measure_device_memory or args.measure_collectives
    if not has_action:
        raise ValueError("At least one acquisition action is required.")
    if args.fit_memory_bias:
        if not args.history_csv or not args.tuner_log:
            raise ValueError("--history-csv and --tuner-log are required for --fit-memory-bias.")
        if args.gpu_memory_mb is None:
            raise ValueError("--gpu-memory-mb is required for --fit-memory-bias.")
    if args.measure_device_memory or args.measure_collectives:
        if not args.backend:
            raise ValueError("--backend is required for measurement actions.")
    if args.measure_collectives:
        _validate_collective_args(args)


def _validate_collective_args(args):
    if not args.collective_runner:
        raise ValueError("--collective-runner is required for --measure-collectives.")
    if args.collective_runner == RUNNER_TORCH_NCCL:
        _reject_external_collective_args(args)
        return
    has_explicit_commands = args.p2p_command and args.all_reduce_command
    if not has_explicit_commands and not args.nccl_tests_bin_dir:
        raise ValueError(
            "--measure-collectives with runner nccl_tests requires "
            "--nccl-tests-bin-dir or explicit commands."
        )


def _reject_external_collective_args(args):
    external_args = args.p2p_command or args.all_reduce_command or args.nccl_tests_bin_dir
    if not external_args:
        return
    raise ValueError(
        "--collective-runner torch_nccl does not accept external collective commands."
    )


def _build_profile_patch(args):
    patch = {}
    if args.fit_memory_bias:
        patch.update(_fit_bias_patch(args))
    if args.measure_device_memory or args.measure_collectives:
        patch.update(_measurement_patch(args))
    return patch


def _fit_bias_patch(args):
    records = build_acquisition_dataset(args.history_csv, args.tuner_log)
    result = fit_memory_bias(records, gpu_memory_mb=args.gpu_memory_mb)
    return {
        "cost_model.reserved_memory_bias_mb": result.reserved_memory_bias_mb,
        "cost_model.peak_activation_bias_mb": result.peak_activation_bias_mb,
    }


def _measurement_patch(args):
    backend = build_backend(args.backend)
    patch = {}
    if args.measure_device_memory:
        patch.update(_device_measurement_to_patch(backend.collect_device_memory()))
    if args.measure_collectives:
        patch.update(
            _collective_measurements_to_patch(
                backend.collect_collectives(
                    args.p2p_command,
                    args.all_reduce_command,
                    runner=args.collective_runner,
                    nccl_tests_bin_dir=args.nccl_tests_bin_dir,
                    ngpus=args.ngpus,
                )
            )
        )
    return patch


def _device_measurement_to_patch(measurement):
    return {"memory.total_memory_mb": int(measurement.metrics["total_memory_mb"])}


def _collective_measurements_to_patch(measurements):
    p2p = measurements["p2p"].metrics
    all_reduce = measurements["all_reduce"].metrics
    return {
        "interconnect.intra_node.p2p_bandwidth_gbps": p2p["bandwidth_gbps"],
        "interconnect.intra_node.p2p_latency_us": p2p["latency_us"],
        "interconnect.intra_node.all_reduce_bandwidth_gbps": all_reduce["bandwidth_gbps"],
        "interconnect.intra_node.all_reduce_latency_us": all_reduce["latency_us"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
