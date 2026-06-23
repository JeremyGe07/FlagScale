from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.profile_acquisition.models import CalibrationStrategy
from flagscale.runner.auto_tuner.profile_acquisition.calibration_cli import (
    build_calibration_runtime_strategy,
    calibration_task_timeout_seconds,
    _calibration_status,
)
from tools.profile_acquisition.profile_acquire import main as profile_acquire_main


def _write_profile_yaml(path: Path):
    profile = {
        'schema_version': 'v1alpha1',
        'identity': {'name': 'nvidia_l20', 'vendor': 'nvidia', 'chip_class': 'gpu'},
        'memory': {'total_memory_mb': 46000, 'bandwidth_gbps': 864},
        'compute': {'bf16_tflops': 119.5, 'attention_tflops': 119.5},
        'interconnect': {
            'intra_node': {
                'fabric': 'pcie',
                'p2p_bandwidth_gbps': 64,
                'p2p_latency_us': 3,
                'all_reduce_bandwidth_gbps': 45,
                'all_reduce_latency_us': 8,
            },
            'host_device': {'bandwidth_gbps': 24, 'latency_us': 10},
        },
        'kernel_support': {
            'transformer_engine': True,
            'flash_attention': True,
            'fused_rmsnorm': True,
        },
        'topology': {'max_nodes': 1, 'devices_per_node': 2, 'homogeneous_only': True},
        'strategy_hints': {
            'default_search_priority': 'performance',
            'max_tensor_model_parallel_size': 2,
            'max_pipeline_model_parallel_size': 2,
            'disabled_dims': {},
        },
        'cost_model': {
            'reserved_memory_bias_mb': 0,
            'peak_activation_bias_mb': 0,
            'overlap': {
                'dp_comm_overlap_ratio': 0.0,
                'tp_comm_overlap_ratio': 0.0,
                'pp_comm_overlap_ratio': 0.0,
            },
        },
    }
    OmegaConf.save(config=OmegaConf.create(profile), f=path)


def _build_config(space_overrides, *, global_batch_size=32):
    return OmegaConf.create(
        {
            'train': {'model': {'global_batch_size': global_batch_size}},
            'experiment': {
                'task': {'type': 'train'},
                'auto_tuner': {'space': space_overrides},
            }
        }
    )


def _build_strategy(dp=2, tp=2, pp=1):
    return CalibrationStrategy(
        strategy_idx=1,
        task_name='tp2-mbs8',
        branch='tp2',
        dp=dp,
        tp=tp,
        pp=pp,
        micro_batch_size=8,
        use_recompute=False,
    )


def test_profile_acquire_cli_runs_dense_template_calibration(tmp_path, capsys):
    profile_in = tmp_path / 'nvidia_l20.yaml'
    profile_out = tmp_path / 'nvidia_l20.dense_warmstart.yaml'
    config_dir = tmp_path / 'conf'
    config_dir.mkdir()
    (config_dir / 'train_auto_tuner.yaml').write_text('defaults: []\n')
    _write_profile_yaml(profile_in)

    summary = SimpleNamespace(
        template_name='dense-8',
        sample_count=8,
        success_count=6,
        oom_count=1,
        other_failure_count=1,
        oom_recall=1.0,
        false_prune_count=0,
        reserved_memory_bias_mb=2048.0,
        peak_activation_bias_mb=512.0,
    )
    result = SimpleNamespace(
        summary=summary,
        profile_patch={
            'cost_model.reserved_memory_bias_mb': 2048.0,
            'cost_model.peak_activation_bias_mb': 512.0,
        },
    )

    with patch(
        'tools.profile_acquisition.profile_acquire.run_profile_calibration',
        return_value=result,
    ) as run_mock:
        exit_code = profile_acquire_main(
            [
                '--profile-in',
                str(profile_in),
                '--profile-out',
                str(profile_out),
                '--run-calibration',
                '--calibration-template',
                'dense-8',
                '--config-path',
                str(config_dir),
                '--config-name',
                'train_auto_tuner',
            ]
        )

    calibrated = OmegaConf.to_container(OmegaConf.load(profile_out), resolve=True)
    stdout = capsys.readouterr().out
    assert run_mock.call_args.kwargs['template_name'] == 'dense-8'
    assert run_mock.call_args.kwargs['config_name'] == 'train_auto_tuner'
    assert exit_code == 0
    assert calibrated['cost_model']['reserved_memory_bias_mb'] == 2048.0
    assert calibrated['cost_model']['peak_activation_bias_mb'] == 512.0
    assert 'Calibration summary:' in stdout
    assert 'template=dense-8' in stdout
    assert 'sample_count=8' in stdout
    assert f'Derived profile path: {profile_out}' in stdout


def test_profile_acquire_cli_runs_file_template_calibration(tmp_path, capsys):
    profile_in = tmp_path / 'nvidia_l20.yaml'
    profile_out = tmp_path / 'nvidia_l20.near_boundary.yaml'
    template_file = tmp_path / 'near_boundary_template.yaml'
    config_dir = tmp_path / 'conf'
    config_dir.mkdir()
    (config_dir / 'train_auto_tuner.yaml').write_text('defaults: []\n')
    template_file.write_text('name: near-boundary\ntasks: []\n')
    _write_profile_yaml(profile_in)

    summary = SimpleNamespace(
        template_name='near-boundary',
        sample_count=0,
        success_count=0,
        oom_count=0,
        other_failure_count=0,
        oom_recall=0.0,
        false_prune_count=0,
        reserved_memory_bias_mb=0.0,
        peak_activation_bias_mb=0.0,
    )
    result = SimpleNamespace(summary=summary, profile_patch={})

    with patch(
        'tools.profile_acquisition.profile_acquire.run_profile_calibration',
        return_value=result,
    ) as run_mock:
        exit_code = profile_acquire_main(
            [
                '--profile-in',
                str(profile_in),
                '--profile-out',
                str(profile_out),
                '--run-calibration',
                '--calibration-template-file',
                str(template_file),
                '--config-path',
                str(config_dir),
                '--config-name',
                'train_auto_tuner',
            ]
        )

    stdout = capsys.readouterr().out
    assert run_mock.call_args.kwargs['template_file'] == str(template_file)
    assert run_mock.call_args.kwargs['template_name'] is None
    assert exit_code == 0
    assert 'template=near-boundary' in stdout


def test_profile_acquire_cli_run_calibration_rejects_two_template_sources(tmp_path):
    profile_in = tmp_path / 'nvidia_l20.yaml'
    profile_out = tmp_path / 'nvidia_l20.near_boundary.yaml'
    template_file = tmp_path / 'near_boundary_template.yaml'
    config_dir = tmp_path / 'conf'
    config_dir.mkdir()
    (config_dir / 'train_auto_tuner.yaml').write_text('defaults: []\n')
    template_file.write_text('name: near-boundary\ntasks: []\n')
    _write_profile_yaml(profile_in)

    with pytest.raises(ValueError, match='exactly one'):
        profile_acquire_main(
            [
                '--profile-in',
                str(profile_in),
                '--profile-out',
                str(profile_out),
                '--run-calibration',
                '--calibration-template',
                'dense-8',
                '--calibration-template-file',
                str(template_file),
                '--config-path',
                str(config_dir),
                '--config-name',
                'train_auto_tuner',
            ]
        )


def test_profile_acquire_cli_run_calibration_requires_existing_config_yaml(tmp_path):
    profile_in = tmp_path / 'nvidia_l20.yaml'
    profile_out = tmp_path / 'nvidia_l20.dense_warmstart.yaml'
    config_dir = tmp_path / 'conf'
    config_dir.mkdir()
    _write_profile_yaml(profile_in)

    with pytest.raises(ValueError, match='train_auto_tuner.yaml'):
        profile_acquire_main(
            [
                '--profile-in',
                str(profile_in),
                '--profile-out',
                str(profile_out),
                '--run-calibration',
                '--calibration-template',
                'dense-8',
                '--config-path',
                str(config_dir),
                '--config-name',
                'train_auto_tuner',
            ]
        )


@pytest.mark.parametrize(
    ('space_overrides', 'message'),
    [
        (
            {
                'use_distributed_optimizer': [True, False],
                'sequence_parallel': [True],
                'context_parallel_size': [1],
            },
            'use_distributed_optimizer',
        ),
        (
            {
                'use_distributed_optimizer': [True],
                'sequence_parallel': [True, False],
                'context_parallel_size': [1],
            },
            'sequence_parallel',
        ),
        (
            {
                'use_distributed_optimizer': [True],
                'sequence_parallel': [True],
                'context_parallel_size': [1, 2],
            },
            'context_parallel_size',
        ),
    ],
)
def test_build_calibration_runtime_strategy_rejects_ambiguous_space(space_overrides, message):
    config = _build_config(space_overrides)

    with pytest.raises(ValueError, match=message):
        build_calibration_runtime_strategy(config, _build_strategy())


def test_build_calibration_runtime_strategy_includes_acc_step():
    config = _build_config(
        {
            'use_distributed_optimizer': [True],
            'sequence_parallel': [True],
            'context_parallel_size': [1],
        },
        global_batch_size=64,
    )

    runtime_strategy = build_calibration_runtime_strategy(config, _build_strategy(dp=4))

    assert runtime_strategy['acc_step'] == 2


@pytest.mark.parametrize(
    ('global_batch_size', 'dp', 'message'),
    [
        (30, 4, 'global_batch_size'),
        (30, 2, 'micro_batch_size'),
    ],
)
def test_build_calibration_runtime_strategy_rejects_indivisible_acc_step(
    global_batch_size,
    dp,
    message,
):
    config = _build_config(
        {
            'use_distributed_optimizer': [True],
            'sequence_parallel': [True],
            'context_parallel_size': [1],
        },
        global_batch_size=global_batch_size,
    )

    with pytest.raises(ValueError, match=message):
        build_calibration_runtime_strategy(config, _build_strategy(dp=dp))


def test_calibration_task_timeout_seconds_gives_first_task_grace():
    assert calibration_task_timeout_seconds(max_time_per_task=240, strategy_idx=1) == 480
    assert calibration_task_timeout_seconds(max_time_per_task=240, strategy_idx=2) == 240


def test_calibration_status_detects_nccl_oom_in_detail_logs(tmp_path):
    log_dir = tmp_path / 'logs'
    detail_dir = log_dir / 'details' / 'host_0_localhost' / 'attempt_0' / '0'
    detail_dir.mkdir(parents=True)
    (detail_dir / 'stdout.log').write_text(
        "NCCL WARN Cuda failure 2 'out of memory'\n",
        encoding='utf-8',
    )

    status = _calibration_status({'error': 'DistBackendError'}, log_dir=log_dir)

    assert status == 'oom'


def test_calibration_status_keeps_non_memory_errors_as_other_failure(tmp_path):
    log_dir = tmp_path / 'logs'
    detail_dir = log_dir / 'details' / 'host_0_localhost' / 'attempt_0' / '0'
    detail_dir.mkdir(parents=True)
    (detail_dir / 'stderr.log').write_text("NameError: name 'warnings' is not defined\n")

    status = _calibration_status({'error': 'ChildFailedError'}, log_dir=log_dir)

    assert status == 'other_failure'
