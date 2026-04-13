import argparse
import time

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.profile_acquisition import (
    build_acquisition_dataset,
    fit_memory_bias,
    write_profile_patch,
)
from flagscale.runner.auto_tuner.profile_acquisition.calibration_runner import (
    run_calibration,
)
from flagscale.runner.auto_tuner.profile_acquisition.backends import build_backend
from flagscale.runner.auto_tuner.profile_acquisition.templates import (
    get_calibration_template,
)
from flagscale.runner.auto_tuner.cost.memory_cost import estimate_memory_cost
from flagscale.runner.auto_tuner.generate import Generator
from flagscale.runner.auto_tuner.record.recorder import Recorder
from flagscale.runner.runner_base import JobStatus
from flagscale.runner.runner_train import SSHTrainRunner

RUNNER_NCCL_TESTS = 'nccl_tests'
RUNNER_TORCH_NCCL = 'torch_nccl'
CALIBRATION_TEMPLATES = ("dense-8", "moe-8")
CALIBRATION_POLL_INTERVAL_SECONDS = 10
CALIBRATION_STARTUP_WAIT_SECONDS = 3


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
    args = parser.parse_args(argv)
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
        calibration_result = _calibration_result(args)
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
                )
            )
        )
    return patch


def _calibration_result(args):
    template = get_calibration_template(args.calibration_template)
    return run_calibration(
        template=template,
        execute_task=_build_calibration_executor(args),
        gpu_memory_mb=_read_profile_gpu_memory_mb(args.profile_in),
    )


def _build_calibration_executor(args):
    runtime = {}

    def execute(strategy):
        if not runtime:
            config = _load_calibration_config(args)
            runtime["config"] = config
            runtime["generator"] = Generator(config)
            runtime["recorder"] = Recorder(config)
            runtime["max_time_per_task"] = _read_max_time_per_task(config)
        return _execute_calibration_task(
            strategy=strategy,
            config=runtime["config"],
            generator=runtime["generator"],
            recorder=runtime["recorder"],
            max_time_per_task=runtime["max_time_per_task"],
        )

    return execute


def _load_calibration_config(args):
    config_path = str(Path(args.config_path).resolve())
    with initialize_config_dir(version_base=None, config_dir=config_path):
        config = compose(config_name=args.config_name)
    OmegaConf.set_struct(config, False)
    _validate_calibration_config(config)
    config.experiment.auto_tuner.chip_profile = {"path": str(Path(args.profile_in).resolve())}
    return config


def _validate_calibration_config(config):
    if "experiment" not in config:
        raise ValueError("Warm-start calibration config is missing experiment section.")
    if config.experiment.task.type != "train":
        raise ValueError("Warm-start calibration only supports train tasks.")
    if "auto_tuner" not in config.experiment:
        raise ValueError("Warm-start calibration requires experiment.auto_tuner configuration.")


def _read_max_time_per_task(config):
    control = config.experiment.auto_tuner.get("control", {})
    return int(control.get("max_time_per_task", 300))


def _execute_calibration_task(*, strategy, config, generator, recorder, max_time_per_task):
    strategy_dict = _build_runtime_strategy(config, strategy)
    task = generator.gen(strategy_dict)
    runner = SSHTrainRunner(task)
    runner.run(enable_monitoring=task.experiment.runner.get("enable_monitoring", False))
    _monitor_calibration_task(runner, strategy_dict, max_time_per_task)
    recorder.record(task, strategy_dict)
    return _calibration_payload(task, strategy_dict, config)


def _build_runtime_strategy(config, strategy):
    space = config.experiment.auto_tuner.get("space", {})
    sequence_parallel = False if strategy.tp == 1 else _space_bool(space, "sequence_parallel")
    use_distributed_optimizer = False if strategy.dp == 1 else _space_bool(
        space, "use_distributed_optimizer"
    )
    return {
        "idx": strategy.strategy_idx,
        "data_parallel_size": strategy.dp,
        "use_distributed_optimizer": use_distributed_optimizer,
        "tensor_model_parallel_size": strategy.tp,
        "sequence_parallel": sequence_parallel,
        "pipeline_model_parallel_size": strategy.pp,
        "num_layers_per_virtual_pipeline_stage": None,
        "use_recompute": strategy.use_recompute,
        "recompute_method": strategy.recompute_method,
        "recompute_granularity": strategy.recompute_granularity,
        "recompute_num_layers": strategy.recompute_num_layers,
        "micro_batch_size": strategy.micro_batch_size,
        "context_parallel_size": _space_int(space, "context_parallel_size", 1),
        "expert_model_parallel_size": strategy.expert_model_parallel_size,
        "decoder_first_pipeline_num_layers": None,
        "decoder_last_pipeline_num_layers": None,
    }


def _space_bool(space, key):
    value = _space_value(space, key, False)
    return bool(value)


def _space_int(space, key, default):
    value = _space_value(space, key, default)
    return int(value)


def _space_value(space, key, default):
    values = space.get(key)
    if values in (None, "auto"):
        return default
    if not isinstance(values, list) or not values:
        return default
    first_value = values[0]
    if first_value == 0:
        return default
    return first_value


def _monitor_calibration_task(runner, strategy, max_time_per_task):
    time.sleep(CALIBRATION_STARTUP_WAIT_SECONDS)
    start_time = time.time()
    saw_running = False
    saw_subprocess = False
    while True:
        if time.time() - start_time > max_time_per_task:
            runner.stop()
            strategy["stopped_by_tuner"] = True
            return
        status = runner._query_status()
        if status == JobStatus.COMPLETED_OR_IDLE:
            return
        if status == JobStatus.RUNNING:
            saw_running = True
        if status == JobStatus.TRANSITIONAL and saw_running:
            runner.stop()
            strategy["stopped_by_tuner"] = True
            return
        sub_process = runner._query_sub_process_status()
        if sub_process:
            saw_subprocess = True
        elif saw_subprocess:
            runner.stop()
            strategy["stopped_by_tuner"] = True
            return
        time.sleep(CALIBRATION_POLL_INTERVAL_SECONDS)


def _calibration_payload(task, strategy, config):
    memory_cost = estimate_memory_cost(strategy, config)
    return {
        "status": _calibration_status(strategy),
        "memory_model_mb": memory_cost["memory_total_mb"],
        "max_mem_mb": _max_mem_value(strategy),
        "performance_ms": strategy.get("performance"),
        "log_path": str(task.train.system.logging.log_dir),
        "error": strategy.get("error") or "",
    }


def _calibration_status(strategy):
    if strategy.get("max_mem") == "OOM":
        return "oom"
    if strategy.get("error"):
        return "other_failure"
    return "success"


def _max_mem_value(strategy):
    max_mem = strategy.get("max_mem")
    if max_mem in (None, "OOM"):
        return None
    return float(max_mem)


def _read_profile_gpu_memory_mb(profile_in):
    profile = OmegaConf.to_container(OmegaConf.load(profile_in), resolve=True)
    memory = profile.get("memory", {})
    if "total_memory_mb" not in memory:
        raise ValueError("Profile is missing memory.total_memory_mb required for calibration.")
    return float(memory["total_memory_mb"])


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


def _print_calibration_summary(summary, profile_out):
    print(
        "Calibration summary: "
        f"template={summary.template_name} "
        f"sample_count={summary.sample_count} "
        f"success_count={summary.success_count} "
        f"oom_count={summary.oom_count} "
        f"other_failure_count={summary.other_failure_count} "
        f"oom_recall={summary.oom_recall} "
        f"false_prune_count={summary.false_prune_count} "
        f"reserved_memory_bias_mb={summary.reserved_memory_bias_mb} "
        f"peak_activation_bias_mb={summary.peak_activation_bias_mb}"
    )
    print(f"Derived profile path: {profile_out}")


if __name__ == "__main__":
    raise SystemExit(main())
