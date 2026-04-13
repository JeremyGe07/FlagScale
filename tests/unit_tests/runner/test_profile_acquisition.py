import csv
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.profile_acquisition.backends.nvidia import (
    NvidiaProfileBackend,
)
from flagscale.runner.auto_tuner.profile_acquisition.bias_calibration import fit_memory_bias
from flagscale.runner.auto_tuner.profile_acquisition.history_parser import build_acquisition_dataset
from flagscale.runner.auto_tuner.profile_acquisition.models import AcquisitionMeasurement
from flagscale.runner.auto_tuner.profile_acquisition.profile_patch import merge_profile_patch
from tools.profile_acquisition.profile_acquire import main as profile_acquire_main

def _write_history_csv(path: Path):
    rows = [
        {
            "idx": "1",
            "memory_model": "28414.0",
            "memory_breakdown": '{"parameters_mb": 1472.1, "model_states_mb": 17665.9, "activations_mb": 10748.0, "recompute_saved_mb": 0.0, "peak_mb": 28414.0, "reserved_mb": 0.0}',
            "max_mem": "33450.0",
            "performance": "9699.35",
            "error": "",
        },
        {
            "idx": "2",
            "memory_model": "44101.0",
            "memory_breakdown": '{"parameters_mb": 1472.1, "model_states_mb": 19208.0, "activations_mb": 24893.0, "recompute_saved_mb": 2288.0, "peak_mb": 44101.0, "reserved_mb": 0.0}',
            "max_mem": "OOM",
            "performance": "",
            "error": "OOM",
        },
        {
            "idx": "3",
            "memory_model": "37248.0",
            "memory_breakdown": '{"parameters_mb": 1472.2, "model_states_mb": 26499.9, "activations_mb": 10748.0, "recompute_saved_mb": 0.0, "peak_mb": 37248.0, "reserved_mb": 0.0}',
            "max_mem": "42388.0",
            "performance": "9829.85",
            "error": "",
        },
    ]
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'idx',
                'memory_model',
                'memory_breakdown',
                'max_mem',
                'performance',
                'error',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

def _write_tuner_log(path: Path):
    path.write_text(
        "\n".join(
            [
                "2026-04-08 11:35:00,000 - FlagScale-AutoTuner - INFO - Run task_1: {'idx': 1}",
                "2026-04-08 11:35:59,143 - FlagScale-AutoTuner - INFO - Record task_1:",
                "2026-04-08 11:35:59,146 - FlagScale-AutoTuner - INFO - task_1 error: {}",
                "2026-04-08 11:36:00,000 - FlagScale-AutoTuner - INFO - Run task_2: {'idx': 2}",
                "2026-04-08 11:36:59,143 - FlagScale-AutoTuner - INFO - Record task_2:",
                "2026-04-08 11:36:59,146 - FlagScale-AutoTuner - INFO - task_2 error: {'OOM': 'Tried to allocate 9.28 GiB. GPU 0 has a total capacity of 44.40 GiB of which 8.98 GiB is free. Of the allocated memory 25.13 GiB is allocated by PyTorch, and 9.79 GiB is reserved by PyTorch but unallocated.'}",
                "2026-04-08 11:37:00,000 - FlagScale-AutoTuner - INFO - Run task_3: {'idx': 3}",
                "2026-04-08 11:37:53,152 - FlagScale-AutoTuner - INFO - Record task_3:",
                "2026-04-08 11:37:53,156 - FlagScale-AutoTuner - INFO - task_3 error: {}",
            ]
        )
    )

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

def test_build_acquisition_dataset_parses_history_and_log(tmp_path):
    history_csv = tmp_path / 'history.csv'
    tuner_log = tmp_path / 'tuner.log'
    _write_history_csv(history_csv)
    _write_tuner_log(tuner_log)

    records = build_acquisition_dataset(history_csv, tuner_log)

    assert len(records) == 3
    assert records[0]['strategy_idx'] == 1
    assert records[0]['status'] == 'success'
    assert records[0]['max_mem_mb'] == 33450.0
    assert records[1]['status'] == 'oom'
    assert records[1]['oom_detail']['reserved_unallocated_mb'] == 10024.96

def test_fit_memory_bias_prefers_oom_recall_with_bounded_false_prunes(tmp_path):
    history_csv = tmp_path / 'history.csv'
    tuner_log = tmp_path / 'tuner.log'
    _write_history_csv(history_csv)
    _write_tuner_log(tuner_log)
    records = build_acquisition_dataset(history_csv, tuner_log)

    result = fit_memory_bias(
        records,
        gpu_memory_mb=46000,
        max_false_prunes=1,
    )

    assert result.reserved_memory_bias_mb > 1800
    assert result.summary['sample_count'] == 3
    assert result.summary['oom_recall'] == 1.0
    assert result.summary['false_prune_count'] <= 1
    assert result.peak_activation_bias_mb >= 0.0

def test_merge_profile_patch_updates_nested_fields_without_mutating_input():
    profile = {
        'memory': {'total_memory_mb': 46000, 'bandwidth_gbps': 864},
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
    patch = {
        'memory.total_memory_mb': 46068,
        'interconnect.intra_node.p2p_bandwidth_gbps': 71.5,
        'cost_model.reserved_memory_bias_mb': 4096,
    }

    merged = merge_profile_patch(profile, patch)

    assert profile['memory']['total_memory_mb'] == 46000
    assert merged['memory']['total_memory_mb'] == 46068
    assert merged['interconnect']['intra_node']['p2p_bandwidth_gbps'] == 71.5
    assert merged['cost_model']['reserved_memory_bias_mb'] == 4096

def test_profile_acquire_cli_fits_bias_and_writes_profile(tmp_path):
    history_csv = tmp_path / 'history.csv'
    tuner_log = tmp_path / 'tuner.log'
    profile_in = tmp_path / 'nvidia_l20.yaml'
    profile_out = tmp_path / 'nvidia_l20.calibrated.yaml'
    _write_history_csv(history_csv)
    _write_tuner_log(tuner_log)
    _write_profile_yaml(profile_in)

    exit_code = profile_acquire_main(
        [
            '--profile-in',
            str(profile_in),
            '--profile-out',
            str(profile_out),
            '--history-csv',
            str(history_csv),
            '--tuner-log',
            str(tuner_log),
            '--fit-memory-bias',
            '--gpu-memory-mb',
            '46000',
        ]
    )

    calibrated = OmegaConf.to_container(OmegaConf.load(profile_out), resolve=True)
    assert exit_code == 0
    assert calibrated['cost_model']['reserved_memory_bias_mb'] > 1800
    assert calibrated['memory']['total_memory_mb'] == 46000

def test_profile_acquire_cli_collects_device_memory_from_backend(tmp_path):
    profile_in = tmp_path / 'nvidia_l20.yaml'
    profile_out = tmp_path / 'nvidia_l20.measured.yaml'
    _write_profile_yaml(profile_in)

    class FakeBackend(NvidiaProfileBackend):
        def collect_device_memory(self):
            return AcquisitionMeasurement(
                kind='device_memory',
                source='nvidia',
                metrics={'total_memory_mb': 46068},
                metadata={'device_count': 2},
            )

    with patch('tools.profile_acquisition.profile_acquire.build_backend', return_value=FakeBackend()):
        exit_code = profile_acquire_main(
            [
                '--profile-in',
                str(profile_in),
                '--profile-out',
                str(profile_out),
                '--backend',
                'nvidia',
                '--measure-device-memory',
            ]
        )

    measured = OmegaConf.to_container(OmegaConf.load(profile_out), resolve=True)
    assert exit_code == 0
    assert measured['memory']['total_memory_mb'] == 46068

def test_profile_acquire_cli_collects_collectives_with_nccl_tests_dir(tmp_path):
    profile_in = tmp_path / 'nvidia_l20.yaml'
    profile_out = tmp_path / 'nvidia_l20.collectives.yaml'
    _write_profile_yaml(profile_in)

    class FakeBackend(NvidiaProfileBackend):
        def collect_collectives(
            self,
            p2p_command,
            all_reduce_command,
            runner=None,
            nccl_tests_bin_dir=None,
            ngpus=2,
        ):
            assert p2p_command is None
            assert all_reduce_command is None
            assert runner == 'nccl_tests'
            assert nccl_tests_bin_dir == '/opt/nccl-tests/build'
            assert ngpus == 2
            return {
                'p2p': AcquisitionMeasurement(
                    kind='collective_bw_latency',
                    source='nvidia',
                    metrics={'bandwidth_gbps': 71.5, 'latency_us': 2.8},
                    metadata={'collective': 'p2p'},
                ),
                'all_reduce': AcquisitionMeasurement(
                    kind='collective_bw_latency',
                    source='nvidia',
                    metrics={'bandwidth_gbps': 49.3, 'latency_us': 7.2},
                    metadata={'collective': 'all_reduce'},
                ),
            }

    with patch('tools.profile_acquisition.profile_acquire.build_backend', return_value=FakeBackend()):
        exit_code = profile_acquire_main(
            [
                '--profile-in',
                str(profile_in),
                '--profile-out',
                str(profile_out),
                '--backend',
                'nvidia',
                '--measure-collectives',
                '--collective-runner',
                'nccl_tests',
                '--nccl-tests-bin-dir',
                '/opt/nccl-tests/build',
            ]
        )

    measured = OmegaConf.to_container(OmegaConf.load(profile_out), resolve=True)
    assert exit_code == 0
    assert measured['interconnect']['intra_node']['p2p_bandwidth_gbps'] == 71.5
    assert measured['interconnect']['intra_node']['all_reduce_latency_us'] == 7.2
