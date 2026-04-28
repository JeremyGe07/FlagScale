import argparse

from pathlib import Path

from flagscale.runner.auto_tuner.profile_acquisition import (
    build_acquisition_dataset,
    fit_memory_bias,
    write_profile_patch,
)
from flagscale.runner.auto_tuner.profile_acquisition.backends import build_backend
from flagscale.runner.auto_tuner.profile_acquisition.calibration_cli import (
    format_calibration_summary,
    run_profile_calibration,
)

RUNNER_NCCL_TESTS = 'nccl_tests'
RUNNER_TORCH_NCCL = 'torch_nccl'
CALIBRATION_TEMPLATES = ("dense-8", "moe-8")
GROUP_SIZE_SEPARATOR = ','
MIN_GROUP_SIZE = 1


def main(argv=None):
    args = _parse_args(argv)
    patch, calibration_summary = _build_profile_patch(args)
    write_profile_patch(args.profile_in, args.profile_out, patch)
    if calibration_summary is not None:
        _print_calibration_summary(calibration_summary, args.profile_out)
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
    parser.add_argument("--run-calibration", action="store_true")
    parser.add_argument("--calibration-template", choices=CALIBRATION_TEMPLATES)
    parser.add_argument("--config-path")
    parser.add_argument("--config-name")
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
    parser.add_argument("--p2p-pair-classes")
    parser.add_argument("--all-reduce-group-sizes")
    args = parser.parse_args(argv)
    args.all_reduce_group_sizes = _parse_group_sizes(args.all_reduce_group_sizes)
    _validate_args(args)
    return args


def _validate_args(args):
    has_action = (
        args.fit_memory_bias
        or args.run_calibration
        or args.measure_device_memory
        or args.measure_collectives
    )
    if not has_action:
        raise ValueError("At least one acquisition action is required.")
    _validate_profile_paths(args)
    if args.fit_memory_bias:
        if not args.history_csv or not args.tuner_log:
            raise ValueError("--history-csv and --tuner-log are required for --fit-memory-bias.")
        if args.gpu_memory_mb is None:
            raise ValueError("--gpu-memory-mb is required for --fit-memory-bias.")
    if args.run_calibration:
        _validate_calibration_args(args)
    if args.measure_device_memory or args.measure_collectives:
        if not args.backend:
            raise ValueError("--backend is required for measurement actions.")
    if args.measure_collectives:
        _validate_collective_args(args)


def _validate_collective_args(args):
    if not args.collective_runner:
        raise ValueError("--collective-runner is required for --measure-collectives.")
    _validate_topology_aware_collective_args(args)
    if args.collective_runner == RUNNER_TORCH_NCCL:
        _reject_external_collective_args(args)
        return
    has_explicit_commands = args.p2p_command and args.all_reduce_command
    if not has_explicit_commands and not args.nccl_tests_bin_dir:
        raise ValueError(
            "--measure-collectives with runner nccl_tests requires "
            "--nccl-tests-bin-dir or explicit commands."
        )


def _validate_topology_aware_collective_args(args):
    if args.p2p_pair_classes is not None and not args.p2p_pair_classes.strip():
        raise ValueError("--p2p-pair-classes must be a comma-separated list or auto.")
    has_topology_options = (
        args.p2p_pair_classes is not None or args.all_reduce_group_sizes is not None
    )
    if has_topology_options and args.collective_runner != RUNNER_TORCH_NCCL:
        raise ValueError(
            "Topology-aware collective options require --collective-runner torch_nccl."
        )


def _reject_external_collective_args(args):
    external_args = args.p2p_command or args.all_reduce_command or args.nccl_tests_bin_dir
    if not external_args:
        return
    raise ValueError(
        "--collective-runner torch_nccl does not accept external collective commands."
    )


def _validate_profile_paths(args):
    profile_in_path = Path(args.profile_in)
    if not profile_in_path.is_file():
        raise ValueError(f"--profile-in does not exist: {profile_in_path}")
    profile_out_parent = Path(args.profile_out).parent
    if not profile_out_parent.exists():
        raise ValueError(f"--profile-out parent directory does not exist: {profile_out_parent}")


def _validate_calibration_args(args):
    if not args.calibration_template:
        raise ValueError("--calibration-template is required for --run-calibration.")
    if not args.config_path or not args.config_name:
        raise ValueError("--config-path and --config-name are required for --run-calibration.")
    config_dir = Path(args.config_path)
    if not config_dir.is_dir():
        raise ValueError(f"--config-path does not exist: {config_dir}")
    config_file = config_dir / f"{args.config_name}.yaml"
    if not config_file.is_file():
        raise ValueError(f"Calibration config does not exist: {config_file}")


def _build_profile_patch(args):
    patch = {}
    calibration_summary = None
    if args.fit_memory_bias:
        patch.update(_fit_bias_patch(args))
    if args.run_calibration:
        calibration_result = run_profile_calibration(
            profile_in=args.profile_in,
            template_name=args.calibration_template,
            config_path=args.config_path,
            config_name=args.config_name,
        )
        patch.update(calibration_result.profile_patch)
        calibration_summary = calibration_result.summary
    if args.measure_device_memory or args.measure_collectives:
        patch.update(_measurement_patch(args))
    return patch, calibration_summary


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
                    p2p_pair_classes=args.p2p_pair_classes,
                    all_reduce_group_sizes=args.all_reduce_group_sizes,
                )
            )
        )
    return patch


def _device_measurement_to_patch(measurement):
    return {"memory.total_memory_mb": int(measurement.metrics["total_memory_mb"])}


def _collective_measurements_to_patch(measurements):
    p2p = measurements["p2p"].metrics
    all_reduce = measurements["all_reduce"].metrics
    patch = {
        "interconnect.intra_node.p2p_bandwidth_gbps": p2p["bandwidth_gbps"],
        "interconnect.intra_node.p2p_latency_us": p2p["latency_us"],
        "interconnect.intra_node.all_reduce_bandwidth_gbps": all_reduce["bandwidth_gbps"],
        "interconnect.intra_node.all_reduce_latency_us": all_reduce["latency_us"],
    }
    patch.update(_p2p_classes_to_patch(measurements.get("p2p_classes", {})))
    patch.update(_all_reduce_profiles_to_patch(measurements.get("all_reduce_profiles", {})))
    return patch


def _parse_group_sizes(raw_value):
    if raw_value is None:
        return None
    parts = [part.strip() for part in raw_value.split(GROUP_SIZE_SEPARATOR)]
    if not all(parts):
        raise ValueError("--all-reduce-group-sizes must be comma-separated positive integers.")
    try:
        group_sizes = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            "--all-reduce-group-sizes must be comma-separated positive integers."
        ) from exc
    invalid = [group_size for group_size in group_sizes if group_size < MIN_GROUP_SIZE]
    if invalid:
        raise ValueError(f"--all-reduce-group-sizes must be positive integers: {invalid}")
    return group_sizes


def _p2p_classes_to_patch(p2p_classes):
    patch = {}
    for name, measurement in p2p_classes.items():
        prefix = f"interconnect.intra_node.p2p_classes.{name}"
        patch[f"{prefix}.bandwidth_gbps"] = measurement.metrics["bandwidth_gbps"]
        patch[f"{prefix}.latency_us"] = measurement.metrics["latency_us"]
        patch[f"{prefix}.gpu_pair"] = measurement.metadata["gpu_pair"]
    return patch


def _all_reduce_profiles_to_patch(all_reduce_profiles):
    patch = {}
    for group_size, measurement in all_reduce_profiles.items():
        prefix = (
            "interconnect.intra_node.collective_profiles."
            f"all_reduce.group_size_{group_size}"
        )
        patch[f"{prefix}.bandwidth_gbps"] = measurement.metrics["bandwidth_gbps"]
        patch[f"{prefix}.latency_us"] = measurement.metrics["latency_us"]
    return patch


def _print_calibration_summary(summary, profile_out):
    print(format_calibration_summary(summary))
    print(f"Derived profile path: {profile_out}")


if __name__ == "__main__":
    raise SystemExit(main())
