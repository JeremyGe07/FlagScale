import time

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.cost.memory_cost import estimate_memory_cost
from flagscale.runner.auto_tuner.generate import Generator
from flagscale.runner.auto_tuner.profile_acquisition.calibration_runner import (
    run_calibration,
)
from flagscale.runner.auto_tuner.record.recorder import Recorder
from flagscale.runner.auto_tuner.profile_acquisition.templates import (
    get_calibration_template,
    load_calibration_template_file,
)
from flagscale.runner.runner_base import JobStatus
from flagscale.runner.runner_train import SSHTrainRunner

CALIBRATION_POLL_INTERVAL_SECONDS = 10
CALIBRATION_STARTUP_WAIT_SECONDS = 3
FIRST_TASK_TIMEOUT_GRACE_MULTIPLIER = 2
OOM_LOG_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "cuda failure 2 'out of memory'",
    "failed to cuda calloc",
)
LOG_SCAN_LIMIT_BYTES = 2_000_000


def run_profile_calibration(
    *, profile_in, template_name=None, template_file=None, config_path, config_name
):
    template = _load_template(template_name=template_name, template_file=template_file)
    return run_calibration(
        template=template,
        execute_task=build_calibration_executor(
            profile_in=profile_in,
            config_path=config_path,
            config_name=config_name,
        ),
        gpu_memory_mb=read_profile_gpu_memory_mb(profile_in),
    )


def _load_template(*, template_name, template_file):
    if template_file is not None:
        return load_calibration_template_file(template_file)
    return get_calibration_template(template_name)


def build_calibration_executor(*, profile_in, config_path, config_name):
    runtime = {}

    def execute(strategy):
        if not runtime:
            runtime.update(
                _initialize_calibration_runtime(
                    profile_in=profile_in,
                    config_path=config_path,
                    config_name=config_name,
                )
            )
        return _execute_calibration_task(
            strategy=strategy,
            config=runtime["config"],
            generator=runtime["generator"],
            recorder=runtime["recorder"],
            max_time_per_task=runtime["max_time_per_task"],
        )

    return execute


def _initialize_calibration_runtime(*, profile_in, config_path, config_name):
    config = load_calibration_config(
        profile_in=profile_in,
        config_path=config_path,
        config_name=config_name,
    )
    return {
        "config": config,
        "generator": Generator(config),
        "recorder": Recorder(config),
        "max_time_per_task": read_max_time_per_task(config),
    }


def load_calibration_config(*, profile_in, config_path, config_name):
    resolved_path = str(Path(config_path).resolve())
    with initialize_config_dir(version_base=None, config_dir=resolved_path):
        config = compose(config_name=config_name)
    OmegaConf.set_struct(config, False)
    validate_calibration_config(config)
    config.experiment.auto_tuner.chip_profile = {"path": str(Path(profile_in).resolve())}
    return config


def validate_calibration_config(config):
    if "experiment" not in config:
        raise ValueError("Warm-start calibration config is missing experiment section.")
    if config.experiment.task.type != "train":
        raise ValueError("Warm-start calibration only supports train tasks.")
    if "auto_tuner" not in config.experiment:
        raise ValueError("Warm-start calibration requires experiment.auto_tuner configuration.")


def read_max_time_per_task(config):
    control = config.experiment.auto_tuner.get("control", {})
    return int(control.get("max_time_per_task", 300))


def build_calibration_runtime_strategy(config, strategy):
    space = config.experiment.auto_tuner.get("space", {})
    sequence_parallel = _resolve_sequence_parallel(space, strategy)
    use_distributed_optimizer = _resolve_use_distributed_optimizer(space, strategy)
    acc_step = _resolve_acc_step(config, strategy)
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
        "acc_step": acc_step,
        "context_parallel_size": _resolve_context_parallel_size(space, strategy),
        "expert_model_parallel_size": strategy.expert_model_parallel_size,
        "decoder_first_pipeline_num_layers": None,
        "decoder_last_pipeline_num_layers": None,
    }


def _resolve_sequence_parallel(space, strategy):
    if strategy.tp == 1:
        return False
    return bool(_resolve_strategy_or_space_value(space, strategy, "sequence_parallel"))


def _resolve_use_distributed_optimizer(space, strategy):
    if strategy.dp == 1:
        return False
    return bool(_resolve_strategy_or_space_value(space, strategy, "use_distributed_optimizer"))


def _resolve_context_parallel_size(space, strategy):
    value = _resolve_strategy_or_space_value(space, strategy, "context_parallel_size")
    return _coerce_positive_int(value, "context_parallel_size")


def _resolve_acc_step(config, strategy):
    global_batch_size = int(config.train.model.global_batch_size)
    data_parallel_size = int(strategy.dp)
    micro_batch_size = int(strategy.micro_batch_size)
    if data_parallel_size <= 0:
        raise ValueError("Warm-start calibration requires data_parallel_size to be > 0.")
    if micro_batch_size <= 0:
        raise ValueError("Warm-start calibration requires micro_batch_size to be > 0.")
    if global_batch_size % data_parallel_size != 0:
        raise ValueError(
            "Warm-start calibration requires train.model.global_batch_size to be "
            "divisible by data_parallel_size: "
            f"global_batch_size={global_batch_size}, data_parallel_size={data_parallel_size}."
        )
    local_batch_size = global_batch_size // data_parallel_size
    if local_batch_size % micro_batch_size != 0:
        raise ValueError(
            "Warm-start calibration requires local batch size to be divisible by "
            "micro_batch_size: "
            f"global_batch_size={global_batch_size}, data_parallel_size={data_parallel_size}, "
            f"local_batch_size={local_batch_size}, micro_batch_size={micro_batch_size}."
        )
    return local_batch_size // micro_batch_size


def _resolve_strategy_or_space_value(space, strategy, key):
    explicit_value = getattr(strategy, key, None)
    if explicit_value is not None:
        return explicit_value
    values = space.get(key)
    if values in (None, "auto"):
        raise ValueError(
            "Warm-start calibration requires a single explicit "
            f"experiment.auto_tuner.space.{key} value."
        )
    if not _is_single_value_sequence(values):
        raise ValueError(
            "Warm-start calibration requires an unambiguous single "
            f"experiment.auto_tuner.space.{key} value."
        )
    return values[0]


def _is_single_value_sequence(values):
    return isinstance(values, Sequence) and not isinstance(values, str) and len(values) == 1


def _coerce_positive_int(value, key):
    coerced = int(value)
    if coerced <= 0:
        raise ValueError(f"Warm-start calibration requires {key} to be > 0.")
    return coerced


def calibration_task_timeout_seconds(*, max_time_per_task, strategy_idx):
    if strategy_idx == 1:
        return max_time_per_task * FIRST_TASK_TIMEOUT_GRACE_MULTIPLIER
    return max_time_per_task


def _execute_calibration_task(*, strategy, config, generator, recorder, max_time_per_task):
    runtime_strategy = build_calibration_runtime_strategy(config, strategy)
    task = generator.gen(runtime_strategy)
    runner = SSHTrainRunner(task)
    runner.run(enable_monitoring=task.experiment.runner.get("enable_monitoring", False))
    _monitor_calibration_task(runner, runtime_strategy, max_time_per_task)
    recorder.record(task, runtime_strategy)
    return _build_calibration_payload(task, runtime_strategy, config)


def _monitor_calibration_task(runner, strategy, max_time_per_task):
    time.sleep(CALIBRATION_STARTUP_WAIT_SECONDS)
    start_time = time.time()
    timeout_seconds = calibration_task_timeout_seconds(
        max_time_per_task=max_time_per_task,
        strategy_idx=int(strategy["idx"]),
    )
    saw_running = False
    saw_subprocess = False
    while True:
        if time.time() - start_time > timeout_seconds:
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
        saw_subprocess = _read_subprocess_state(runner, strategy, saw_subprocess)
        time.sleep(CALIBRATION_POLL_INTERVAL_SECONDS)


def _read_subprocess_state(runner, strategy, saw_subprocess):
    if runner._query_sub_process_status():
        return True
    if saw_subprocess:
        runner.stop()
        strategy["stopped_by_tuner"] = True
    return False


def _build_calibration_payload(task, strategy, config):
    memory_cost = estimate_memory_cost(strategy, config)
    log_dir = Path(task.train.system.logging.log_dir)
    return {
        "status": _calibration_status(strategy, log_dir=log_dir),
        "memory_model_mb": memory_cost["memory_total_mb"],
        "max_mem_mb": _max_mem_value(strategy),
        "performance_ms": strategy.get("performance"),
        "log_path": str(log_dir),
        "error": strategy.get("error") or "",
    }


def _calibration_status(strategy, *, log_dir=None):
    if strategy.get("max_mem") == "OOM":
        return "oom"
    if _contains_oom_marker(strategy.get("error") or ""):
        return "oom"
    if log_dir is not None and _log_dir_contains_oom(log_dir):
        return "oom"
    if strategy.get("error"):
        return "other_failure"
    return "success"


def _log_dir_contains_oom(log_dir):
    root = Path(log_dir)
    if not root.exists():
        return False
    for path in root.rglob("*.log"):
        if _file_contains_oom(path):
            return True
    return False


def _file_contains_oom(path):
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")[-LOG_SCAN_LIMIT_BYTES:]
    except OSError:
        return False
    return _contains_oom_marker(content)


def _contains_oom_marker(content):
    lowered = str(content).lower()
    return any(marker in lowered for marker in OOM_LOG_MARKERS)


def _max_mem_value(strategy):
    max_mem = strategy.get("max_mem")
    if max_mem in (None, "OOM"):
        return None
    return float(max_mem)


def read_profile_gpu_memory_mb(profile_in):
    profile = OmegaConf.to_container(OmegaConf.load(profile_in), resolve=True)
    memory = profile.get("memory", {})
    if "total_memory_mb" not in memory:
        raise ValueError("Profile is missing memory.total_memory_mb required for calibration.")
    return float(memory["total_memory_mb"])


def format_calibration_summary(summary):
    return (
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
