from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from flagscale.runner.auto_tuner.profile_acquisition.models import AcquisitionMeasurement
from tools.profile_acquisition.profile_acquire import main as profile_acquire_main


def _write_profile_yaml(path):
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
            }
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


def test_measure_collectives_requires_explicit_runner(tmp_path):
    profile_in = tmp_path / 'nvidia_l20.yaml'
    profile_out = tmp_path / 'nvidia_l20.collectives.yaml'
    _write_profile_yaml(profile_in)

    with pytest.raises(ValueError, match='--collective-runner'):
        profile_acquire_main(
            [
                '--profile-in',
                str(profile_in),
                '--profile-out',
                str(profile_out),
                '--backend',
                'nvidia',
                '--measure-collectives',
                '--nccl-tests-bin-dir',
                '/opt/nccl-tests/build',
            ]
        )


def test_measure_collectives_torch_runner_does_not_require_nccl_tests_dir(tmp_path):
    profile_in = tmp_path / 'nvidia_l20.yaml'
    profile_out = tmp_path / 'nvidia_l20.collectives.yaml'
    _write_profile_yaml(profile_in)

    class FakeBackend:
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
            assert runner == 'torch_nccl'
            assert nccl_tests_bin_dir is None
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
                'torch_nccl',
            ]
        )

    measured = OmegaConf.to_container(OmegaConf.load(profile_out), resolve=True)
    assert exit_code == 0
    assert measured['interconnect']['intra_node']['p2p_bandwidth_gbps'] == 71.5
    assert measured['interconnect']['intra_node']['all_reduce_latency_us'] == 7.2


def test_measure_collectives_nccl_tests_runner_requires_bin_dir_or_commands(tmp_path):
    profile_in = tmp_path / 'nvidia_l20.yaml'
    profile_out = tmp_path / 'nvidia_l20.collectives.yaml'
    _write_profile_yaml(profile_in)

    with pytest.raises(ValueError, match='--nccl-tests-bin-dir'):
        profile_acquire_main(
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
            ]
        )
